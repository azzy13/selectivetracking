#!/usr/bin/env python3
"""
Simplified worker for GroundingDINO + ByteTrack + SceneGraph pipeline.

Deliberately minimal:
  - GroundingDINO detection only (no Florence2)
  - ByteTrack only (no CLIP tracker, no CLIP loading at all)
  - SceneGraphBuilder + SceneGraphMissionFilter as the sole post-track filter
  - No ReferringDetectionFilter, no TrackColorGate

The scene graph filter interprets per-frame track attributes (color_votes,
region, heading) to keep only tracks that match the mission prompt.
"""

from __future__ import annotations

import argparse
import importlib
import os
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import time
import torch
from PIL import Image
from torch.amp import autocast
from torchvision import transforms as T

from groundingdino.util.inference import load_model, predict
from scene_graph import SceneGraphBuilder, SceneGraphMissionFilter, ColorReIDMatcher

# ============================================================
# Defaults
# ============================================================
DEFAULT_CONFIG_PATH  = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
DEFAULT_WEIGHTS_PATH = "weights/groundingdino_swinb_cogcoor.pth"
DEFAULT_TEXT_PROMPT  = "car."
DEFAULT_MIN_BOX_AREA = 10
DEFAULT_FRAME_RATE   = 10


# ============================================================
# Utilities (duplicated from worker_clean to keep this file self-contained)
# ============================================================

def build_normalize_transform():
    def resize_if_needed(img):
        w, h = img.size
        short_side = min(w, h)
        if short_side > 800:
            scale = 800 / short_side
            return img.resize((int(w * scale), int(h * scale)))
        return img

    return T.Compose([
        T.Lambda(resize_if_needed),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def parse_frame_id(frame_name: str) -> int:
    stem = os.path.splitext(frame_name)[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse frame id from: {frame_name}")
    return int(digits)


def convert_dino_to_xyxy(boxes, logits, W: int, H: int) -> np.ndarray:
    dets = []
    for box, logit in zip(boxes, logits):
        cx, cy, w, h = box
        if w <= 0 or h <= 0:
            continue
        score = float(logit)
        x1 = (cx - w / 2.0) * W
        y1 = (cy - h / 2.0) * H
        x2 = (cx + w / 2.0) * W
        y2 = (cy + h / 2.0) * H
        dets.append([max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2), score])
    return np.array(dets, dtype=np.float32) if dets else np.empty((0, 5), dtype=np.float32)


def parse_kv_list(kv_list) -> dict:
    out = {}
    for kv in kv_list or []:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        try:
            if v.lower() in ("true", "false"):
                out[k] = v.lower() == "true"
            elif "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
        except ValueError:
            out[k] = v
    return out


# ============================================================
# Worker
# ============================================================

class Worker:
    """
    Simplified worker: GroundingDINO + ByteTrack + SceneGraphMissionFilter.

    Pipeline per frame:
        detect (DINO) → [scale-aware threshold] → track (ByteTrack)
        → sg_builder.update() → mission_filter.decide() → write kept tracks
    """

    def __init__(
        self,
        *,
        config_path: str = DEFAULT_CONFIG_PATH,
        weights_path: str = DEFAULT_WEIGHTS_PATH,
        text_prompt: str = DEFAULT_TEXT_PROMPT,
        box_thresh: float = 0.35,
        text_thresh: float = 0.25,
        use_fp16: bool = False,
        device: Optional[str] = None,
        # Tracker
        tracker_kwargs: Optional[dict] = None,
        frame_rate: int = DEFAULT_FRAME_RATE,
        # Scale-aware detection
        use_scale_aware_thresh: bool = True,
        small_box_area_thresh: int = 5000,
        # Scene-graph filter
        use_mission_filter: bool = True,
        mission_filter_hard: bool = False,
        mission_filter_thresh: float = 0.10,
        # Color re-ID (recovers track IDs after FOV/perspective changes)
        use_color_reid: bool = True,
        reid_max_lost_frames: int = 25,
        # Misc
        min_box_area: int = DEFAULT_MIN_BOX_AREA,
        save_video: bool = False,
    ):
        self.text_prompt            = text_prompt
        self.box_thresh             = float(box_thresh)
        self.text_thresh            = float(text_thresh)
        self.use_fp16               = bool(use_fp16)
        self.use_scale_aware_thresh = use_scale_aware_thresh
        self.small_box_area_thresh  = int(small_box_area_thresh)
        self.min_box_area           = int(min_box_area)
        self.save_video             = bool(save_video)
        self.frame_rate             = int(frame_rate)
        self.use_mission_filter     = use_mission_filter
        self.use_color_reid         = bool(use_color_reid)
        self._reid_max_lost_frames  = int(reid_max_lost_frames)

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Detector
        self.dino_model = load_model(config_path, weights_path)
        if hasattr(self.dino_model, "to"):
            self.dino_model = self.dino_model.to(self.device)
        self._transform = build_normalize_transform()

        # Tracker (ByteTrack only)
        tracker_kwargs = dict(tracker_kwargs or {})
        tracker_args = argparse.Namespace(
            track_thresh=tracker_kwargs.pop("track_thresh", 0.45),
            track_buffer=tracker_kwargs.pop("track_buffer", 120),
            match_thresh=tracker_kwargs.pop("match_thresh", 0.80),
            aspect_ratio_thresh=tracker_kwargs.pop("aspect_ratio_thresh", 10.0),
            lambda_weight=tracker_kwargs.pop("lambda_weight", 0.25),
            min_box_area=self.min_box_area,
            mot20=tracker_kwargs.pop("mot20", False),
        )
        mod = importlib.import_module("tracker.byte_tracker")
        self.tracker = mod.BYTETracker(tracker_args, frame_rate=frame_rate)

        # Mission filter (built lazily per sequence so prompt can be overridden)
        self._mf_hard   = mission_filter_hard
        self._mf_thresh = mission_filter_thresh

        # Fixed tracker type — always ByteTrack in this worker
        self.tracker_type = 'bytetrack'

        print(f"[WorkerSimple] prompt='{text_prompt}' | box_thresh={box_thresh} "
              f"| mission_filter={'on' if use_mission_filter else 'off'} "
              f"| color_reid={'on' if use_color_reid else 'off'} "
              f"| device={self.device}")

    # ------------------------------------------------------------------
    # Public per-frame API (used by the ROS2 node)
    # ------------------------------------------------------------------

    def preprocess_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        return self._preprocess(frame_bgr)

    def predict_detections(self, frame_bgr: np.ndarray, tensor_image: torch.Tensor,
                           orig_h: int, orig_w: int) -> np.ndarray:
        return self._detect(frame_bgr, tensor_image, orig_h, orig_w)

    def update_tracker(self, dets_xyxy: np.ndarray, orig_h: int, orig_w: int):
        if dets_xyxy.size == 0:
            dets_xyxy = np.empty((0, 5), dtype=np.float32)
        return self.tracker.update(dets_xyxy, [orig_h, orig_w], [orig_h, orig_w])

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        t = self._transform(img)
        if str(self.device).startswith("cuda"):
            t = t.cuda(non_blocking=True)
        return t.half() if self.use_fp16 else t

    def _detect(self, frame_bgr: np.ndarray, tensor: torch.Tensor,
                orig_h: int, orig_w: int) -> np.ndarray:
        init_thresh = self.box_thresh * 0.5 if self.use_scale_aware_thresh else self.box_thresh
        with torch.no_grad(), autocast(device_type=str(self.device).split(":")[0], enabled=self.use_fp16):
            boxes, logits, _ = predict(
                model=self.dino_model,
                image=tensor,
                caption=self.text_prompt,
                box_threshold=init_thresh,
                text_threshold=self.text_thresh,
            )
        dets = convert_dino_to_xyxy(boxes, logits, orig_w, orig_h)
        if self.use_scale_aware_thresh and dets.size > 0:
            dets = self._scale_aware_filter(dets)
        return dets

    def _scale_aware_filter(self, dets: np.ndarray) -> np.ndarray:
        kept = []
        tiny = self.small_box_area_thresh // 4   # e.g. 1250 px² — very distant objects
        for det in dets:
            x1, y1, x2, y2, score = det
            area = (x2 - x1) * (y2 - y1)
            if area < tiny:
                thresh = self.box_thresh * 0.45  # very lenient for tiny objects
            elif area < self.small_box_area_thresh:
                thresh = self.box_thresh * 0.60
            elif area < self.small_box_area_thresh * 3:
                thresh = self.box_thresh * 0.80
            else:
                thresh = self.box_thresh
            if score >= thresh:
                kept.append(det)
        return np.array(kept, dtype=np.float32).reshape(-1, 5) if kept else np.empty((0, 5), dtype=np.float32)

    # ------------------------------------------------------------------
    # Sequence processing
    # ------------------------------------------------------------------

    def process_sequence(
        self,
        *,
        seq: str,
        img_folder: str,
        gt_folder: str,
        out_path: str,
        sort_frames: bool = True,
        enable_scene_graph: bool = True,
    ):
        seq_path = os.path.join(img_folder, seq)
        if not os.path.isdir(seq_path):
            raise FileNotFoundError(f"Sequence path not found: {seq_path}")

        frame_files = [f for f in os.listdir(seq_path)
                       if os.path.isfile(os.path.join(seq_path, f))]
        if sort_frames:
            frame_files = sorted(frame_files, key=parse_frame_id)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        # Scene graph builder (always on in worker_simple)
        sg_builder = SceneGraphBuilder(text_prompt=self.text_prompt)

        # Mission filter
        mission_filter = None
        if self.use_mission_filter:
            mission_filter = SceneGraphMissionFilter(
                text_prompt=self.text_prompt,
                hard_mode=self._mf_hard,
                score_thresh=self._mf_thresh,
            )

        # Color re-ID: recovers track IDs lost during FOV/perspective changes
        color_reid = ColorReIDMatcher(max_lost_frames=self._reid_max_lost_frames) \
            if self.use_color_reid else None

        # Video writer
        video_writer = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        stats = {"det_total": 0, "track_total": 0, "kept_total": 0, "frames": 0}
        timings = {"load": 0.0, "preprocess": 0.0, "detect": 0.0,
                   "track": 0.0, "scene_graph": 0.0, "filter": 0.0}

        with open(out_path, "w") as f_res:
            for idx, frame_name in enumerate(frame_files):
                frame_id = parse_frame_id(frame_name)

                t0 = time.perf_counter()
                img = cv2.imread(os.path.join(seq_path, frame_name))
                if img is None:
                    continue
                orig_h, orig_w = img.shape[:2]
                timings["load"] += time.perf_counter() - t0

                if idx == 0:
                    print(f"[{seq}] F{frame_id}: {orig_h}x{orig_w} | ByteTrack | dino")

                if self.save_video and video_writer is None:
                    vpath = out_path.replace(".txt", ".mp4")
                    video_writer = cv2.VideoWriter(vpath, fourcc, self.frame_rate, (orig_w, orig_h))

                # Preprocess
                t0 = time.perf_counter()
                tensor = self._preprocess(img)
                timings["preprocess"] += time.perf_counter() - t0

                # Detect (GPU — sync before/after for accurate wall time)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                dets = self._detect(img, tensor, orig_h, orig_w)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timings["detect"] += time.perf_counter() - t0

                # Track
                t0 = time.perf_counter()
                if dets.size == 0:
                    dets_in = np.empty((0, 5), dtype=np.float32)
                else:
                    dets_in = dets
                tracks = self.tracker.update(dets_in, [orig_h, orig_w], [orig_h, orig_w])
                timings["track"] += time.perf_counter() - t0

                # Scene graph update
                t0 = time.perf_counter()
                frame_graph = sg_builder.update(frame_id, tracks, orig_h, orig_w, frame_bgr=img)
                timings["scene_graph"] += time.perf_counter() - t0

                # Mission filter: decide which tracks to keep
                t0 = time.perf_counter()
                if mission_filter is not None:
                    kept_ids = mission_filter.decide(frame_graph)
                    tracks_out = [t for t in tracks if t.track_id in kept_ids]
                else:
                    tracks_out = tracks
                timings["filter"] += time.perf_counter() - t0

                # Color re-ID: remap new IDs that match recently-lost tracks.
                # Use all ByteTrack IDs (not just tracks_out) so the graveyard
                # doesn't misfire when the mission filter removes a track.
                if color_reid is not None:
                    all_bt_ids = {t.track_id for t in tracks}
                    color_reid.update(frame_id, all_bt_ids, frame_graph)

                stats["det_total"]   += len(dets)
                stats["track_total"] += len(tracks)
                stats["kept_total"]  += len(tracks_out)
                stats["frames"]      += 1

                if idx % 20 == 0:
                    print(f"[{seq}] F{frame_id}: det={len(dets)} "
                          f"track={len(tracks)} kept={len(tracks_out)}")

                # Write MOT output (use resolved ID if re-ID is active)
                for t in tracks_out:
                    x, y, w, h = t.tlwh
                    if w * h > self.min_box_area:
                        out_id = color_reid.resolve(t.track_id) if color_reid else t.track_id
                        f_res.write(
                            f"{frame_id},{out_id},{x:.2f},{y:.2f},"
                            f"{w:.2f},{h:.2f},{t.score:.3f},-1,-1,-1\n"
                        )

                # Video
                if self.save_video and video_writer is not None:
                    vis = img.copy()
                    for t in tracks_out:
                        x, y, w, h = t.tlwh
                        if w * h > self.min_box_area:
                            out_id = color_reid.resolve(t.track_id) if color_reid else t.track_id
                            cv2.rectangle(vis, (int(x), int(y)),
                                          (int(x + w), int(y + h)), (0, 255, 0), 2)
                            cv2.putText(vis, f"ID:{out_id}", (int(x), int(y) - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    video_writer.write(vis)

        n = max(1, stats["frames"])
        total_ms = sum(timings.values()) * 1000 / n
        print(f"[{seq}] Done. frames={stats['frames']} "
              f"avg_det={stats['det_total']/n:.1f} "
              f"avg_track={stats['track_total']/n:.1f} "
              f"avg_kept={stats['kept_total']/n:.1f}")
        print(f"[{seq}] Latency per frame (avg over {stats['frames']} frames):")
        for stage, t in timings.items():
            ms = t * 1000 / n
            print(f"  {stage:<12} {ms:6.1f} ms  ({ms/total_ms*100:.0f}%)")
        print(f"  {'TOTAL':<12} {total_ms:6.1f} ms  → {1000/total_ms:.1f} FPS theoretical")
        print(f"[{seq}] Results → {out_path}")

        # Save scene graph
        if enable_scene_graph:
            sg_path = out_path.replace(".txt", "_scene_graphs.jsonl")
            sg_builder.save_jsonl(sg_path)
            summary = sg_builder.get_summary()
            print(f"[{seq}] Scene graph: {summary['total_frames']} frames, "
                  f"avg {summary['avg_nodes_per_frame']} nodes/frame, "
                  f"avg {summary['avg_edges_per_frame']} edges/frame")

        # Debug: show per-track color evidence from mission filter
        if mission_filter is not None:
            unique_ids = {n["track_id"] for fg in sg_builder.frames for n in fg["nodes"]}
            print(f"[{seq}] Mission filter color evidence:")
            for tid in sorted(unique_ids):
                ev = mission_filter.get_track_color_evidence(tid)
                if ev:
                    print(f"  track {tid}: {ev}")

        if video_writer is not None:
            video_writer.release()
            print(f"[{seq}] Video → {out_path.replace('.txt', '.mp4')}")

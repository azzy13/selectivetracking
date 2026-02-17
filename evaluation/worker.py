#!/usr/bin/env python3
"""
Clean refactored worker for GroundingDINO + Tracker evaluation.

Features:
  - Supports bytetrack, clip, and smartclip trackers
  - ReferKITTI GT visualization with YOLO-style labels
  - Referring detection filter for referring expression tracking
  - Optional video output with GT boxes
  - Multi-GPU dispatch support
"""
from __future__ import annotations

import os
import sys
import argparse
import importlib
import subprocess
from collections import deque
from typing import Dict, Tuple, Optional, Iterable, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torch.cuda.amp import autocast
import clip
import pandas as pd

from groundingdino.util.inference import load_model, predict
from demo.florence2_adapter import Florence2Detector

# ============================
# Configuration Defaults
# ============================
DEFAULT_CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
DEFAULT_WEIGHTS_PATH = "weights/groundingdino_swinb_cogcoor.pth"
DEFAULT_TEXT_PROMPT = "car. pedestrian."
DEFAULT_MIN_BOX_AREA = 10
DEFAULT_FRAME_RATE = 10

TRACKER_REGISTRY: Dict[str, Tuple[str, str]] = {
    "bytetrack": ("selectivetrack.byte_tracker", "BYTETracker"),
    "clip": ("selectivetrack.clip_tracker", "CLIPTracker"),
    "smartclip": ("selectivetrack.smart_clip_tracker", "SmartCLIPTracker"),
}


# ============================
# Utility Functions
# ============================
def build_normalize_transform():
    """Build image normalization transform with max 800px short side."""
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
    """Extract integer frame ID from filename (e.g., '000001.jpg' -> 1)."""
    stem = os.path.splitext(frame_name)[0]
    digits = ''.join(ch for ch in stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse frame id from: {frame_name}")
    return int(digits)


def convert_dino_to_xyxy(boxes: Iterable, logits: Iterable, W: int, H: int) -> np.ndarray:
    """Convert DINO boxes (cx,cy,w,h normalized) to [x1,y1,x2,y2,score]."""
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


def parse_kv_list(kv_list):
    """Parse --tracker_kv key=val arguments into typed dict."""
    out = {}
    for kv in kv_list or []:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        try:
            if v.lower() in ("true", "false"):
                out[k] = (v.lower() == "true")
            elif "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
        except ValueError:
            out[k] = v
    return out


# ============================
# ReferKITTI GT Helpers
# ============================
def load_referkitti_labels(label_path: str) -> List[Dict]:
    """
    Load YOLO-style labels from ReferKITTI.

    Format: class_id track_id x_left_norm y_top_norm width_norm height_norm

    NOTE: Despite field names, ReferKITTI stores TOP-LEFT coordinates, not center coordinates!

    Returns:
        List of dicts with keys: class_id, track_id, x_center, y_center, width, height
        (names kept as x_center/y_center for compatibility, but values are actually top-left)
    """
    if not os.path.isfile(label_path):
        return []

    labels = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                labels.append({
                    "class_id": int(parts[0]),
                    "track_id": int(float(parts[1])),
                    "x_center": float(parts[2]),
                    "y_center": float(parts[3]),
                    "width": float(parts[4]),
                    "height": float(parts[5]),
                })
            except ValueError:
                continue
    return labels


def draw_referkitti_gt_boxes(frame: np.ndarray, label_path: str, target_ids: Optional[set] = None) -> np.ndarray:
    """
    Draw ReferKITTI GT boxes on frame.

    Args:
        frame: BGR image
        label_path: Path to YOLO-style label file
        target_ids: Optional set of track IDs to highlight

    Returns:
        Frame with GT boxes drawn
    """
    labels = load_referkitti_labels(label_path)
    H, W = frame.shape[:2]

    for lab in labels:
        tid = lab["track_id"]
        if target_ids is not None and tid not in target_ids:
            continue

        # Convert normalized coords to pixel bbox
        # NOTE: Despite the dict key names, these are TOP-LEFT coords, not center!
        x_left = lab["x_center"] * W
        y_top = lab["y_center"] * H
        bw = lab["width"] * W
        bh = lab["height"] * H
        x1, y1 = int(x_left), int(y_top)
        x2, y2 = int(x1 + bw), int(y1 + bh)

        # Draw bbox and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"GT ID:{tid}", (int(x1 + bw), y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame


# ============================
# Referring Detection Filter
# ============================
class ReferringDetectionFilter:
    """
    Hybrid filter for referring expression tracking.

    Filters GroundingDINO detections using:
    1. Spatial position matching (left/right/top/bottom/center)
    2. Patch-based HSV color detection (black/white/red/blue/etc.)
    3. CLIP text-image similarity for general appearance matching

    The color detection uses patch-based histogram voting on HSV color space,
    which is more stable and accurate than CLIP for basic color attributes.
    """

    def __init__(
        self,
        clip_model,
        clip_preprocess,
        text_embedding: torch.Tensor,
        threshold: float = 0.25,
        pad: int = 4,
        device: str = "cuda",
        text_prompt: str = "",
        use_spatial_filter: bool = True,
        use_color_filter: bool = True
    ):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        # Ensure text_embedding is 1D [D] or 2D [1, D]
        text_embedding = text_embedding.to(device)
        if text_embedding.dim() == 3:
            text_embedding = text_embedding.squeeze(0)
        if text_embedding.dim() == 2 and text_embedding.size(0) == 1:
            text_embedding = text_embedding.squeeze(0)
        self.text_embedding = text_embedding
        self.threshold = float(threshold)
        self.pad = int(pad)
        self.device = device
        self.total_dets_in = 0
        self.total_dets_out = 0
        self.text_prompt = text_prompt.lower()
        self.use_spatial_filter = use_spatial_filter
        self.use_color_filter = use_color_filter

        # Parse spatial and color keywords from text
        self.spatial_region = self._parse_spatial_region(self.text_prompt)
        self.color_attribute = self._parse_color_attribute(self.text_prompt)

    def _parse_spatial_region(self, text: str):
        """Parse spatial keywords from referring expression."""
        if not text:
            return None

        # Check for spatial keywords (order matters - check specific before general)
        if "left" in text or "leftmost" in text:
            return "left"
        elif "right" in text or "rightmost" in text:
            return "right"
        elif "top" in text or "upper" in text or "above" in text:
            return "top"
        elif "bottom" in text or "lower" in text or "below" in text:
            return "bottom"
        elif "center" in text or "middle" in text or "central" in text:
            return "center"
        return None

    def _parse_color_attribute(self, text: str):
        """Parse color keywords from referring expression."""
        if not text:
            return None

        # Common colors in referring expressions
        colors = ["black", "white", "red", "blue", "green", "yellow", "gray", "grey", "silver", "dark", "light"]
        for color in colors:
            if color in text:
                return color
        return None

    def _spatial_score(self, bbox_xyxy: np.ndarray, img_width: int, img_height: int) -> float:
        """
        Compute spatial score for a bbox based on parsed spatial region.
        Returns 1.0 for perfect match, 0.0 for opposite region.
        """
        if not self.spatial_region:
            return 1.0  # No spatial constraint

        x1, y1, x2, y2 = bbox_xyxy[:4]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if self.spatial_region == "left":
            # Left half of image: score decreases linearly from left (1.0) to right (0.0)
            return 1.0 - (cx / img_width)
        elif self.spatial_region == "right":
            # Right half: score increases from left to right
            return cx / img_width
        elif self.spatial_region == "top":
            return 1.0 - (cy / img_height)
        elif self.spatial_region == "bottom":
            return cy / img_height
        elif self.spatial_region == "center":
            # Distance from center, normalized
            dist_x = abs(cx - img_width / 2) / (img_width / 2)
            dist_y = abs(cy - img_height / 2) / (img_height / 2)
            return 1.0 - np.sqrt(dist_x**2 + dist_y**2) / np.sqrt(2)

        return 1.0

    def filter(self, frame_bgr: np.ndarray, dets_xyxy: np.ndarray, verbose: bool = False) -> np.ndarray:
        """Filter detections by GroundingDINO score and spatial position.

        Color filtering is handled post-track by TrackColorGate.
        """
        if dets_xyxy.size == 0:
            return dets_xyxy

        self.total_dets_in += len(dets_xyxy)
        H, W = frame_bgr.shape[:2]

        dino_scores = dets_xyxy[:, 4].copy()

        if self.use_spatial_filter and self.spatial_region:
            spatial_scores = np.array([self._spatial_score(det, W, H) for det in dets_xyxy])
            combined_scores = dino_scores * spatial_scores
        else:
            spatial_scores = None
            combined_scores = dino_scores.copy()

        filtered = dets_xyxy[combined_scores >= self.threshold]

        if verbose and len(dets_xyxy) > 0:
            print(f"\n  [DINO scores] min={dino_scores.min():.3f}, max={dino_scores.max():.3f}, mean={dino_scores.mean():.3f}")
            if spatial_scores is not None:
                print(f"  [Spatial] region={self.spatial_region}, min={spatial_scores.min():.3f}, max={spatial_scores.max():.3f}")
            print(f"  [Pre-filter] kept {len(filtered)}/{len(dets_xyxy)} (thresh={self.threshold:.2f})")

        self.total_dets_out += len(filtered)
        return filtered

    def _get_patchwise_dominant_color(self, crop_rgb: np.ndarray, grid_size: int = 4):
        """
        Detect dominant color from a crop using patch-based histogram voting.

        Splits crop into grid_size x grid_size patches, computes dominant color
        per patch via HSV analysis, then votes for overall dominant color.

        Args:
            crop_rgb: RGB image crop (H x W x 3)
            grid_size: Number of patches per dimension (default 4x4 = 16 patches)

        Returns:
            Tuple of (dominant_color, votes_dict):
                dominant_color: 'dark', 'light', 'gray', 'red', 'orange',
                                'yellow', 'green', 'blue', or 'unknown'
                votes_dict: dict mapping color name -> patch count

        Note:
            - Saturated pixels (S >= 35) classified by hue first (red/blue/etc)
            - Achromatic pixels (S < 50) classified by brightness:
              'dark' (V < 80), 'light' (V > 180), 'gray' (80-180)
        """
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        h, w = hsv.shape[:2]

        # Handle very small crops
        if h < grid_size or w < grid_size:
            grid_size = min(h, w, 2)

        patch_h = max(1, h // grid_size)
        patch_w = max(1, w // grid_size)
        color_votes = {}

        def classify_pixel_color(hsv_pixel):
            """Classify a single HSV pixel into a color category.

            Checks saturation first: saturated pixels are classified by hue
            (chromatic), regardless of brightness. Only low-saturation pixels
            fall through to brightness-based (achromatic) classification.
            This prevents bright/dark reds from being misclassified as
            'light' or 'dark'.
            """
            h_val, s_val, v_val = hsv_pixel

            # Chromatic colors: if saturation is high enough, classify by hue
            # regardless of brightness. This correctly handles bright red,
            # dark red, etc.
            if s_val >= 35:
                if h_val < 15 or h_val >= 160:
                    return 'red'
                elif 15 <= h_val < 25:
                    return 'orange'
                elif 25 <= h_val < 35:
                    return 'yellow'
                elif 35 <= h_val < 85:
                    return 'green'
                elif 85 <= h_val < 160:
                    return 'blue'

            # Achromatic: low saturation, classify by brightness
            if v_val < 80:
                return 'dark'
            elif v_val > 180:
                return 'light'
            else:
                return 'gray'

        # Vote across patches
        for i in range(grid_size):
            for j in range(grid_size):
                # Extract patch
                y_start = i * patch_h
                y_end = min((i + 1) * patch_h, h)
                x_start = j * patch_w
                x_end = min((j + 1) * patch_w, w)

                patch = hsv[y_start:y_end, x_start:x_end]

                # Compute mean HSV for this patch
                if patch.size > 0:
                    mean_pixel = np.mean(patch.reshape(-1, 3), axis=0)
                    color = classify_pixel_color(mean_pixel)
                    color_votes[color] = color_votes.get(color, 0) + 1

        # Return dominant color AND votes dict
        if not color_votes:
            return 'unknown', {}

        dominant = max(color_votes.items(), key=lambda x: x[1])[0]
        return dominant, color_votes

    def _compute_color_similarities(self, frame_bgr: np.ndarray, dets_xyxy: np.ndarray) -> np.ndarray:
        """
        Compute color matching scores using patch-based histogram voting.

        This replaces CLIP-based color filtering with direct HSV color analysis.
        Returns binary-like scores: 1.0 for color match, 0.0 for mismatch.
        """
        if dets_xyxy.size == 0:
            return np.ones(0, dtype=np.float32)

        # If no color attribute in prompt, skip filtering (return all 1.0)
        if not self.color_attribute:
            return np.ones(len(dets_xyxy), dtype=np.float32)

        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        scores = []
        for (x1, y1, x2, y2, _) in dets_xyxy:
            # Clip to image bounds
            xi1 = max(0, int(x1))
            yi1 = max(0, int(y1))
            xi2 = min(W, int(x2))
            yi2 = min(H, int(y2))

            # Skip invalid or very small crops
            if xi2 <= xi1 or yi2 <= yi1 or (xi2 - xi1) < 10 or (yi2 - yi1) < 10:
                scores.append(0.5)  # Neutral score for invalid crops
                continue

            # Extract crop
            crop_rgb = rgb[yi1:yi2, xi1:xi2]

            # Get color votes via patch-based voting
            detected_color, votes = self._get_patchwise_dominant_color(crop_rgb, grid_size=4)

            # Presence-based scoring: if target color has enough patches,
            # count as match even if not the dominant color (handles aerial
            # views where windshield reflections can outvote car body paint).
            scores.append(self._color_score_with_presence(
                votes, self.color_attribute, min_target_patches=2))

        return np.array(scores, dtype=np.float32)

    def _match_color(self, detected_color: str, target_color: str) -> bool:
        """
        Check if detected color matches target color from prompt.
        Handles synonyms and related colors.
        """
        detected_color = detected_color.lower()
        target_color = target_color.lower()

        # Direct match
        if detected_color == target_color:
            return True

        # Handle synonyms and related colors
        color_groups = {
            'dark': ['dark', 'black'],  # "dark" now primary (brightness-based)
            'light': ['light', 'white'],  # "light" now primary (brightness-based)
            'gray': ['gray', 'grey', 'silver'],
            'red': ['red'],
            'orange': ['orange'],
            'yellow': ['yellow'],
            'green': ['green'],
            'blue': ['blue']
        }

        # Check if both colors are in the same group
        for group_colors in color_groups.values():
            if detected_color in group_colors and target_color in group_colors:
                return True

        return False

    def _color_score_three_way(self, detected_color: str, target_color: str) -> float:
        """Three-outcome color scoring for post-track gating.

        Returns:
            1.0: chromatic match (or achromatic match for achromatic targets)
            0.0: chromatic mismatch — positive evidence against target color
            0.5: achromatic dominant (dark/light/gray) — uninformative
        """
        ACHROMATIC = {'dark', 'light', 'gray', 'unknown'}
        detected_color = detected_color.lower()
        target_color = target_color.lower()

        # Match first (handles "black" target → detected "dark" → 1.0)
        if self._match_color(detected_color, target_color):
            return 1.0

        # Achromatic detected but didn't match target → uninformative
        if detected_color in ACHROMATIC:
            return 0.5

        # Chromatic color that doesn't match → evidence against
        return 0.0

    # Adjacent hue neighbors: when verifying a target color, patches of a
    # neighboring hue still provide supporting evidence (e.g., orange patches
    # support a "red" target because red cars often register as orange under
    # warm lighting or when viewed at an angle).
    _HUE_NEIGHBORS = {
        'red': ['orange'],
        'orange': ['red', 'yellow'],
        'yellow': ['orange', 'green'],
        'green': ['yellow'],
        'blue': [],
    }

    def _color_score_with_presence(self, votes: dict, target_color: str,
                                    min_target_patches: int = 3) -> float:
        """Score based on target color *presence* rather than dominance.

        From aerial views, windshield reflections (green/blue) can outvote the
        actual car body paint. This method checks whether the target color has
        enough patches to count as evidence, even if it's not the majority.

        Neighboring hues (e.g. orange for red) are counted as supporting
        evidence because lighting and viewing angle shift perceived hue.

        Returns:
            1.0  if target color (incl. synonyms + neighbors) has >= min_target_patches
            Otherwise falls back to three-way scoring on dominant color.
        """
        if not votes:
            return 0.5

        target_lc = target_color.lower()

        # Count patches matching target color (including synonyms)
        target_count = 0
        for color, count in votes.items():
            if self._match_color(color, target_lc):
                target_count += count

        if target_count >= min_target_patches:
            return 1.0

        # Also count neighboring hue patches as supporting evidence
        neighbor_count = target_count
        for neighbor in self._HUE_NEIGHBORS.get(target_lc, []):
            neighbor_count += votes.get(neighbor, 0)

        if neighbor_count >= min_target_patches:
            return 1.0

        # Fall back to standard three-way scoring
        dominant = max(votes, key=votes.get)
        return self._color_score_three_way(dominant, target_lc)

    def get_stats(self) -> dict:
        """Return filtering statistics."""
        retention = self.total_dets_out / self.total_dets_in if self.total_dets_in > 0 else 0.0
        return {
            "total_in": self.total_dets_in,
            "total_out": self.total_dets_out,
            "retention_rate": retention,
            "threshold": self.threshold
        }


# ============================
# Post-Track Color Gate
# ============================
class TrackColorGate:
    """Stateful post-track color gate with per-track EMA and hysteresis.

    Applied AFTER the tracker so that temporal identity is established first.
    Each track maintains a running color confidence; hysteresis prevents
    flicker from shadows or single-frame misclassifications.

    Three-outcome scoring per frame:
        1.0  supports target color (chromatic match)
        0.0  supports NOT target color (chromatic mismatch)
        0.5  uninformative (achromatic dominant — shadows, highlights)

    Hysteresis:
        Enter confirmed:  score > CONFIRM_THRESH for CONFIRM_COUNT of last WINDOW frames
        Exit confirmed:   FAIL_STREAK_TO_DROP consecutive 0.0 scores
        Unknown (0.5):    neutral — does not affect entry, exit, or fail streak
    """

    CONFIRM_THRESH = 0.65
    DROP_THRESH = 0.35
    CONFIRM_COUNT = 3
    WINDOW_SIZE = 5
    EMA_ALPHA = 0.3
    FAIL_STREAK_TO_DROP = 5

    def __init__(self, color_filter: ReferringDetectionFilter):
        self.color_filter = color_filter
        self.target_color = color_filter.color_attribute
        self._state: Dict[int, dict] = {}
        self.total_tracks_in = 0
        self.total_tracks_out = 0

    def _get_or_init(self, track_id: int) -> dict:
        if track_id not in self._state:
            self._state[track_id] = {
                'ema_score': 0.5,
                'recent_scores': deque(maxlen=self.WINDOW_SIZE),
                'fail_streak': 0,
                'confirmed': False,
            }
        return self._state[track_id]

    # Vertical padding fraction applied to bbox before color voting.
    # Compensates for tight detector bboxes that miss the car body in
    # aerial views (windshield reflections would otherwise dominate).
    COLOR_PAD_Y = 0.25

    def _score_bbox(self, frame_bgr: np.ndarray, x: float, y: float,
                    w: float, h: float) -> float:
        """Compute three-outcome color score for a single tlwh bbox."""
        H_img, W_img = frame_bgr.shape[:2]

        # Pad bbox vertically to capture more of the object body.
        # Aerial/elevated cameras produce tight bboxes where windshield
        # reflections dominate; padding recovers the actual paint area.
        pad_y = int(h * self.COLOR_PAD_Y)
        xi1 = max(0, int(x))
        yi1 = max(0, int(y) - pad_y // 3)       # small upward
        xi2 = min(W_img, int(x + w))
        yi2 = min(H_img, int(y + h) + pad_y)     # larger downward

        if xi2 <= xi1 or yi2 <= yi1 or (xi2 - xi1) < 10 or (yi2 - yi1) < 10:
            return 0.5  # too small → uninformative

        crop_rgb = cv2.cvtColor(frame_bgr[yi1:yi2, xi1:xi2], cv2.COLOR_BGR2RGB)
        _, votes = self.color_filter._get_patchwise_dominant_color(crop_rgb, grid_size=4)
        return self.color_filter._color_score_with_presence(
            votes, self.target_color, min_target_patches=3)

    def update(self, tracks: list, frame_bgr: np.ndarray,
               verbose: bool = False) -> list:
        """Evaluate color per track and return only confirmed tracks."""
        if not self.target_color:
            return tracks

        self.total_tracks_in += len(tracks)
        active_ids = set()
        confirmed = []

        for t in tracks:
            tid = t.track_id
            active_ids.add(tid)
            x, y, w, h = t.tlwh
            if w * h < 10:
                continue

            state = self._get_or_init(tid)
            score = self._score_bbox(frame_bgr, x, y, w, h)

            # EMA — skip unknown (0.5) so shadows don't drag confidence down
            if score != 0.5:
                state['ema_score'] = (
                    self.EMA_ALPHA * score
                    + (1.0 - self.EMA_ALPHA) * state['ema_score']
                )

            state['recent_scores'].append(score)

            # Fail streak: only 0.0 (chromatic mismatch) increments;
            # 1.0 resets; 0.5 (unknown) leaves unchanged
            if score == 0.0:
                state['fail_streak'] += 1
            elif score == 1.0:
                state['fail_streak'] = 0

            # --- hysteresis ---
            if not state['confirmed']:
                # Entry: need CONFIRM_COUNT frames > CONFIRM_THRESH in window
                n_high = sum(1 for s in state['recent_scores']
                             if s > self.CONFIRM_THRESH)
                if n_high >= self.CONFIRM_COUNT:
                    state['confirmed'] = True
            else:
                # Exit: FAIL_STREAK_TO_DROP consecutive 0.0 frames
                if state['fail_streak'] >= self.FAIL_STREAK_TO_DROP:
                    state['confirmed'] = False

            if state['confirmed']:
                confirmed.append(t)

            if verbose:
                tag = "CONFIRMED" if state['confirmed'] else "pending"
                print(f"    [ColorGate] T{tid}: score={score:.1f} "
                      f"ema={state['ema_score']:.2f} "
                      f"streak={state['fail_streak']} {tag}")

        # Purge stale tracks
        for tid in [k for k in self._state if k not in active_ids]:
            del self._state[tid]

        self.total_tracks_out += len(confirmed)

        if verbose and tracks:
            print(f"  [ColorGate] {len(confirmed)}/{len(tracks)} tracks confirmed")

        return confirmed

    def get_stats(self) -> dict:
        rate = self.total_tracks_out / self.total_tracks_in if self.total_tracks_in else 0.0
        return {
            'total_in': self.total_tracks_in,
            'total_out': self.total_tracks_out,
            'retention_rate': rate,
            'active_states': len(self._state),
            'target_color': self.target_color,
        }


# ============================
# Worker Class
# ============================
class Worker:
    """
    GroundingDINO + Tracker evaluation worker.

    Supports multiple tracker types, ReferKITTI GT visualization,
    referring expression filtering, and video output.
    """

    def __init__(
        self,
        *,
        # Detector config
        config_path: str = DEFAULT_CONFIG_PATH,
        weights_path: str = DEFAULT_WEIGHTS_PATH,
        text_prompt: str = DEFAULT_TEXT_PROMPT,
        detector: str = "dino",
        box_thresh: float = 0.35,
        text_thresh: float = 0.25,
        use_fp16: bool = False,
        device: Optional[str] = None,
        # Tracker config
        tracker_type: str = "bytetrack",
        tracker_kwargs: Optional[dict] = None,
        # Referring filter
        referring_mode: str = "none",
        referring_thresh: float = 0.25,
        use_spatial_filter: bool = True,
        use_color_filter: bool = True,
        # Scale-aware detection
        use_scale_aware_thresh: bool = True,
        small_box_area_thresh: int = 5000,
        # Misc
        frame_rate: int = DEFAULT_FRAME_RATE,
        min_box_area: int = DEFAULT_MIN_BOX_AREA,
        verbose_first_n_frames: int = 5,
        save_video: bool = False,
        show_gt_boxes: bool = False,
        dataset_type: str = "mot",  # "mot" or "referkitti"
        referkitti_data_root: Optional[str] = None,  # Path to ReferKITTI root (for GT labels)
        target_object_ids: Optional[List[int]] = None,  # For referring expressions - which object IDs to show as GT
    ):
        self.text_prompt = text_prompt
        self.box_thresh = float(box_thresh)
        self.text_thresh = float(text_thresh)
        self.use_scale_aware_thresh = use_scale_aware_thresh
        self.small_box_area_thresh = small_box_area_thresh
        self.use_fp16 = bool(use_fp16)
        self.frame_rate = int(frame_rate)
        self.min_box_area = int(min_box_area)
        self.save_video = bool(save_video)
        self.show_gt_boxes = bool(show_gt_boxes)
        self.dataset_type = dataset_type
        self.referkitti_data_root = referkitti_data_root
        self.target_object_ids = set(target_object_ids) if target_object_ids else None
        self.verbose_first_n_frames = int(verbose_first_n_frames)

        # Device
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Detector
        self.detector_kind = detector
        if self.detector_kind == "dino":
            self.dino_model = load_model(config_path, weights_path)
            if hasattr(self.dino_model, "to"):
                self.dino_model = self.dino_model.to(self.device)
        else:
            self.florence = Florence2Detector(
                model_id="microsoft/Florence-2-large",
                device=self.device,
                fp16=self.use_fp16
            )

        self._transform = build_normalize_transform()

        # Tracker
        tracker_kwargs = dict(tracker_kwargs or {})
        tracker_args = argparse.Namespace(
            track_thresh=tracker_kwargs.pop("track_thresh", 0.5),
            track_buffer=tracker_kwargs.pop("track_buffer", 30),
            match_thresh=tracker_kwargs.pop("match_thresh", 0.8),
            aspect_ratio_thresh=tracker_kwargs.pop("aspect_ratio_thresh", 10.0),
            lambda_weight=tracker_kwargs.pop("lambda_weight", 0.25),
            text_sim_thresh=tracker_kwargs.pop("text_sim_thresh", 0.15),
            min_box_area=self.min_box_area,
            mot20=tracker_kwargs.pop("mot20", False),
            **tracker_kwargs,
        )
        self.tracker = self._build_tracker(tracker_type, tracker_args, frame_rate=self.frame_rate)
        self.tracker_type = tracker_type

        # CLIP setup
        self.class_names = [c.strip() for c in self.text_prompt.split(".") if c.strip()] or ["object"]
        self.text_embedding = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_pad = int(tracker_kwargs.pop("clip_pad", 4))

        need_clip = self.tracker_type in ("clip", "smartclip") or referring_mode != "none"
        if need_clip:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()
            with torch.no_grad():
                tokens = clip.tokenize(self.class_names).to(self.device)
                self.text_embedding = F.normalize(
                    self.clip_model.encode_text(tokens).float(), dim=-1
                ).contiguous()

        # Referring filter
        self.referring_filter = None
        if referring_mode != "none" and self.clip_model is not None:
            self.referring_filter = ReferringDetectionFilter(
                clip_model=self.clip_model,
                clip_preprocess=self.clip_preprocess,
                text_embedding=self.text_embedding,
                threshold=referring_thresh,
                pad=self.clip_pad,
                device=self.device,
                text_prompt=text_prompt,
                use_spatial_filter=use_spatial_filter,
                use_color_filter=use_color_filter
            )
            spatial_region = self.referring_filter.spatial_region
            color_attr = self.referring_filter.color_attribute
            filters = []
            if spatial_region:
                filters.append(f"spatial={spatial_region}")
            if color_attr:
                filters.append(f"color={color_attr}")
            filter_str = ", ".join(filters) if filters else "none"
            print(f"[Worker] Referring filter: thresh={referring_thresh:.2f}, filters=[{filter_str}]")

        # Post-track color gate (uses color methods from referring_filter)
        self.color_gate = None
        if use_color_filter and self.referring_filter is not None and self.referring_filter.color_attribute:
            self.color_gate = TrackColorGate(color_filter=self.referring_filter)
            print(f"[Worker] Post-track color gate: target={self.color_gate.target_color}, "
                  f"confirm={TrackColorGate.CONFIRM_COUNT}/{TrackColorGate.WINDOW_SIZE} frames, "
                  f"exit_streak={TrackColorGate.FAIL_STREAK_TO_DROP}")

    @staticmethod
    def _build_tracker(tracker_type: str, tracker_args: argparse.Namespace, *, frame_rate: int):
        """Build tracker from registry."""
        if tracker_type not in TRACKER_REGISTRY:
            raise ValueError(f"Unknown tracker: {tracker_type}. Available: {list(TRACKER_REGISTRY.keys())}")
        module_path, class_name = TRACKER_REGISTRY[tracker_type]
        module = importlib.import_module(module_path)
        TrackerCls = getattr(module, class_name)
        return TrackerCls(tracker_args, frame_rate=frame_rate)

    def preprocess_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess frame for DINO."""
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        tensor = self._transform(img)
        if str(self.device).startswith("cuda"):
            tensor = tensor.cuda(non_blocking=True)
        return tensor.half() if self.use_fp16 else tensor

    def predict_detections(self, frame_bgr: np.ndarray, tensor_image: Optional[torch.Tensor],
                          orig_h: int, orig_w: int) -> np.ndarray:
        """Run object detection with optional scale-aware thresholding."""
        if self.detector_kind == "dino":
            # Use lower threshold initially if scale-aware is enabled
            # This ensures we don't miss distant/small objects
            initial_box_thresh = self.box_thresh * 0.5 if self.use_scale_aware_thresh else self.box_thresh

            with torch.no_grad(), autocast(enabled=self.use_fp16):
                boxes, logits, _ = predict(
                    model=self.dino_model,
                    image=tensor_image,
                    caption=self.text_prompt,
                    box_threshold=initial_box_thresh,
                    text_threshold=self.text_thresh,
                )
            dets = convert_dino_to_xyxy(boxes, logits, orig_w, orig_h)

            # Apply scale-aware filtering if enabled
            if self.use_scale_aware_thresh and dets.size > 0:
                dets = self._apply_scale_aware_filtering(dets, orig_w, orig_h)

            return dets
        else:
            return self.florence.predict(
                frame_bgr=frame_bgr,
                text_prompt=self.text_prompt,
                box_threshold=self.box_thresh
            )

    def _apply_scale_aware_filtering(self, dets_xyxy: np.ndarray, img_w: int = 0, img_h: int = 0) -> np.ndarray:
        """
        Apply scale-aware thresholding: lower threshold for small/distant objects.

        Small boxes (distant objects) are harder to detect and get lower scores,
        so we use a more lenient threshold for them.
        """
        if dets_xyxy.size == 0:
            return dets_xyxy

        kept = []
        for det in dets_xyxy:
            x1, y1, x2, y2, score = det
            box_area = (x2 - x1) * (y2 - y1)

            # Compute adaptive threshold based on box size
            # Small boxes (distant): use lower threshold
            # Large boxes (close): use full threshold
            if box_area < self.small_box_area_thresh:
                # Small box: use 60% of original threshold
                adaptive_thresh = self.box_thresh * 0.6
            elif box_area < self.small_box_area_thresh * 3:
                # Medium box: use 80% of original threshold
                adaptive_thresh = self.box_thresh * 0.8
            else:
                # Large box: use full threshold
                adaptive_thresh = self.box_thresh

            # Keep detection if score >= adaptive threshold
            if score >= adaptive_thresh:
                kept.append(det)

        return np.array(kept, dtype=np.float32).reshape(-1, 5) if kept else np.empty((0, 5), dtype=np.float32)

    def update_tracker(self, dets_xyxy: np.ndarray, orig_h: int, orig_w: int):
        """Update tracker with detections."""
        if dets_xyxy.size == 0:
            dets_xyxy = np.empty((0, 5), dtype=np.float32)
        return self.tracker.update(dets_xyxy, [orig_h, orig_w], [orig_h, orig_w])

    def update_tracker_clip(self, dets_xyxy: np.ndarray, frame_bgr: np.ndarray,
                           orig_h: int, orig_w: int):
        """Update CLIP-aware tracker."""
        dets = dets_xyxy if dets_xyxy.size else np.empty((0, 5), dtype=np.float32)
        det_embs = self._compute_detection_embeddings(frame_bgr, dets)
        return self.tracker.update(
            detections=dets,
            detection_embeddings=det_embs,
            img_info=(orig_h, orig_w),
            text_embedding=self.text_embedding,
            class_names=self.class_names,
        )

    def _compute_detection_embeddings(self, frame_bgr: np.ndarray, dets_xyxy: np.ndarray) -> List[Optional[torch.Tensor]]:
        """
        Compute CLIP embeddings for detections using spatial masking.

        Instead of cropping, this masks out everything outside the bbox while
        preserving spatial context. This helps CLIP understand spatial expressions
        like "left car" vs "right car" because the position within the full frame
        is maintained.

        Approach:
        1. Start with full image
        2. Create mask: everything outside bbox → black/invisible
        3. Encode the full masked image (bbox position preserved)
        4. Combine with full image embedding for robustness
        """
        if dets_xyxy.size == 0:
            return []

        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Compute full image embedding once
        full_img_pil = Image.fromarray(rgb)
        full_img_tensor = self.clip_preprocess(full_img_pil).unsqueeze(0).to(self.device, non_blocking=True)
        with torch.no_grad():
            full_img_emb = F.normalize(self.clip_model.encode_image(full_img_tensor), dim=-1).float().cpu().squeeze(0)

        # Compute masked embeddings (preserves spatial position)
        masked_images = []
        for (x1, y1, x2, y2, _) in dets_xyxy.tolist():
            xi1 = max(0, int(x1) - self.clip_pad)
            yi1 = max(0, int(y1) - self.clip_pad)
            xi2 = min(W, int(x2) + self.clip_pad)
            yi2 = min(H, int(y2) + self.clip_pad)

            if xi2 > xi1 and yi2 > yi1 and (xi2 - xi1) >= 10 and (yi2 - yi1) >= 10:
                # Create masked image: black everywhere except bbox region
                masked = np.zeros_like(rgb)
                masked[yi1:yi2, xi1:xi2] = rgb[yi1:yi2, xi1:xi2]
                masked_images.append(Image.fromarray(masked))
            else:
                masked_images.append(None)

        batch = [self.clip_preprocess(m).unsqueeze(0) for m in masked_images if m is not None]
        if not batch:
            return [None] * len(masked_images)

        batch_t = torch.cat(batch, 0).to(self.device, non_blocking=True)
        with torch.no_grad():
            masked_embs = F.normalize(self.clip_model.encode_image(batch_t), dim=-1).float().cpu()

        # Combine masked embedding with full image embedding
        # masked_emb: object + spatial position, full_img_emb: global context
        out, j = [], 0
        for m in masked_images:
            if m is None:
                out.append(None)
            else:
                # Weighted average: more weight on masked (spatial info) than full image
                combined_emb = F.normalize((0.7 * masked_embs[j] + 0.3 * full_img_emb), dim=-1)
                out.append(combined_emb)
                j += 1
        return out

    @staticmethod
    def _write_mot_line(fh, frame_id: int, track_id: int, x: float, y: float, w: float, h: float):
        """Write MOTChallenge format line."""
        fh.write(f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

    def process_sequence(
        self,
        *,
        seq: str,
        img_folder: str,
        gt_folder: str,
        out_path: str,
        sort_frames: bool = True,
        video_out_path: Optional[str] = None,
    ):
        """
        Process a sequence and generate tracking results.

        Args:
            seq: Sequence name
            img_folder: Root folder containing sequence images
            gt_folder: Root folder containing ground truth
            out_path: Output path for tracking results
            sort_frames: Whether to sort frames by ID
            video_out_path: Optional video output path
        """
        seq_path = os.path.join(img_folder, seq)
        if not os.path.isdir(seq_path):
            raise FileNotFoundError(f"Sequence path not found: {seq_path}")

        # Load GT data
        gt_pandas_data = None
        if self.dataset_type == "mot":
            gt_txt_file = os.path.join(gt_folder, "gt", seq + ".txt")
            if os.path.isfile(gt_txt_file):
                gt_pandas_data = pd.read_csv(
                    gt_txt_file, header=None,
                    names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "x1", "x2", "x3", "x4"],
                    sep=","
                )
                gt_pandas_data.sort_values(by="frame", inplace=True)

        # Get frame files
        frame_files = [f for f in os.listdir(seq_path) if os.path.isfile(os.path.join(seq_path, f))]
        if sort_frames:
            frame_files = sorted(frame_files, key=parse_frame_id)

        # Setup video writer
        video_writer = None
        if self.save_video:
            if video_out_path is None:
                video_out_path = out_path.replace(".txt", ".mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            if self.target_object_ids:
                print(f"[{seq}] Tracking {len(self.target_object_ids)} target object IDs: {sorted(self.target_object_ids)}")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "w") as f_res:
            for idx, frame_name in enumerate(frame_files):
                frame_id = parse_frame_id(frame_name)
                img = cv2.imread(os.path.join(seq_path, frame_name))
                if img is None:
                    continue
                orig_h, orig_w = img.shape[:2]

                if self.save_video and video_writer is None:
                    video_writer = cv2.VideoWriter(video_out_path, fourcc, self.frame_rate, (orig_w, orig_h))
                    print(f"[{seq}] Saving video to: {video_out_path}")

                # Preprocess
                if self.detector_kind == "dino":
                    tensor = self.preprocess_frame(img)
                    if idx == 0:
                        print(f"[{seq}] F{frame_id}: {orig_h}x{orig_w} | {type(self.tracker).__name__} | {self.detector_kind}")
                else:
                    tensor = None
                    if idx == 0:
                        print(f"[{seq}] F{frame_id}: {orig_h}x{orig_w} | {type(self.tracker).__name__} | {self.detector_kind}")

                # Detect
                dets = self.predict_detections(img, tensor, orig_h, orig_w)
                show_detail = (idx % 20 == 0)
                if show_detail:
                    print(f"[{seq}] F{frame_id}: det={len(dets)}", end="")

                # Filter
                if self.referring_filter is not None:
                    dets_before = len(dets)
                    dets = self.referring_filter.filter(img, dets, verbose=show_detail)
                    if show_detail:
                        print(f" → filt={len(dets)}", end="")

                # Track
                if self.tracker_type in ("clip", "smartclip"):
                    tracks = self.update_tracker_clip(dets, img, orig_h, orig_w)
                else:
                    tracks = self.update_tracker(dets, orig_h, orig_w)

                if show_detail:
                    print(f" → track={len(tracks)}", end="")

                # Post-track color gate
                if self.color_gate is not None:
                    tracks = self.color_gate.update(tracks, img, verbose=show_detail)
                    if show_detail:
                        print(f" → color={len(tracks)}", end="")

                if show_detail:
                    print()

                # Write results
                for t in tracks:
                    x, y, w, h = t.tlwh
                    if w * h > self.min_box_area:
                        self._write_mot_line(f_res, frame_id, t.track_id, float(x), float(y), float(w), float(h))

                # Video output
                if self.save_video and video_writer is not None:
                    vis_frame = img.copy()

                    # Draw predicted tracks (green)
                    for t in tracks:
                        x, y, w, h = t.tlwh
                        if w * h > self.min_box_area:
                            x1, y1 = int(x), int(y)
                            x2, y2 = int(x + w), int(y + h)
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(vis_frame, f"ID:{t.track_id}", (x1, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Draw GT boxes if enabled
                    if self.show_gt_boxes:
                        if self.dataset_type == "referkitti":
                            # ReferKITTI: YOLO-style labels (only show target objects for referring expressions)
                            if self.referkitti_data_root:
                                label_path = os.path.join(
                                    self.referkitti_data_root, "KITTI", "training",
                                    "labels_with_ids", "image_02", seq, f"{frame_id:06d}.txt"
                                )
                                vis_frame = draw_referkitti_gt_boxes(vis_frame, label_path, target_ids=self.target_object_ids)
                        elif gt_pandas_data is not None:
                            # MOT: pandas format
                            gt_frame_data = gt_pandas_data[gt_pandas_data["frame"] == frame_id]
                            for _, row in gt_frame_data.iterrows():
                                x1 = int(row["bb_left"])
                                y1 = int(row["bb_top"])
                                w = int(row["bb_width"])
                                h = int(row["bb_height"])
                                x2, y2 = x1 + w, y1 + h
                                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(vis_frame, f"GT ID:{int(row['id'])}", (x1 + w, y1 - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    video_writer.write(vis_frame)

        print(f"[{seq}] Saved results to: {out_path}")

        if video_writer is not None:
            video_writer.release()
            print(f"[{seq}] Saved video to: {video_out_path}")

        if self.referring_filter is not None:
            stats = self.referring_filter.get_stats()
            print(f"[{seq}] Referring filter: {stats['total_in']} → {stats['total_out']} "
                  f"({stats['retention_rate']*100:.1f}% retention)")

        if self.color_gate is not None:
            cg = self.color_gate.get_stats()
            print(f"[{seq}] Color gate: {cg['total_in']} → {cg['total_out']} "
                  f"({cg['retention_rate']*100:.1f}% retention, "
                  f"active_states={cg['active_states']})")

    def process_many(self, *, seqs: Iterable[str], img_folder: str, res_folder: str,
                    gt_folder: str, suffix: str = ".txt"):
        """Process multiple sequences."""
        os.makedirs(res_folder, exist_ok=True)
        for seq in seqs:
            out_path = os.path.join(res_folder, f"{seq}{suffix}")
            self.process_sequence(seq=seq, img_folder=img_folder, gt_folder=gt_folder, out_path=out_path)


# ============================
# CLI
# ============================
if __name__ == "__main__":
    import glob as _glob
    from datetime import datetime

    def list_sequences(img_root: str):
        return sorted([d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))])

    def collect_sequences(args) -> List[str]:
        seqs = set()
        if args.seq:
            seqs.update(args.seq)
        if args.seq_file:
            with open(args.seq_file) as fh:
                seqs.update(line.strip() for line in fh if line.strip() and not line.startswith("#"))
        if args.seq_glob:
            for pat in args.seq_glob:
                for p in _glob.glob(os.path.join(args.img_folder, pat)):
                    if os.path.isdir(p):
                        seqs.add(os.path.basename(p))
        if args.all or not seqs:
            seqs.update(list_sequences(args.img_folder))
        return sorted(seqs)

    def resolve_single_out(seq: str, out_arg: Optional[str], out_dir: Optional[str], timestamp: bool) -> str:
        if out_arg and out_arg.lower().endswith(".txt"):
            os.makedirs(os.path.dirname(out_arg), exist_ok=True)
            return out_arg
        root = out_arg or out_dir or "outputs"
        if timestamp:
            root = os.path.join(root, datetime.now().strftime("%Y-%m-%d_%H%M"))
        os.makedirs(root, exist_ok=True)
        return os.path.join(root, f"{seq}.txt")

    def dispatch_multi_gpu(seqs: List[str], args, tracker_kv: dict):
        devices = [d.strip() for d in (args.devices or "0").split(",") if d.strip()]
        jobs = max(1, int(args.jobs))
        procs = []

        root = args.out_dir or "outputs"
        if args.timestamp:
            root = os.path.join(root, datetime.now().strftime("%Y-%m-%d_%H%M"))
        os.makedirs(root, exist_ok=True)

        this_script = os.path.abspath(__file__)
        for i, seq in enumerate(seqs):
            gpu_id = devices[i % len(devices)]
            out_path = os.path.join(root, f"{seq}.txt")

            if args.save_video:
                video_folder = root.replace("/results", "/videos").replace("\\results", "\\videos")
                if "results" not in root:
                    video_folder = os.path.join(os.path.dirname(root), "videos")
                os.makedirs(video_folder, exist_ok=True)
                video_path = os.path.join(video_folder, f"{seq}.mp4")

            cmd = [
                sys.executable, "-u", this_script,
                "--seq", seq,
                "--img_folder", args.img_folder,
                "--out", out_path,
                "--tracker", args.tracker,
                "--box_thresh", str(args.box_thresh),
                "--text_thresh", str(args.text_thresh),
                "--track_thresh", str(args.track_thresh),
                "--match_thresh", str(args.match_thresh),
                "--track_buffer", str(args.track_buffer),
                "--text_prompt", args.text_prompt,
                "--detector", args.detector,
                "--config", args.config,
                "--weights", args.weights,
                "--min_box_area", str(args.min_box_area),
                "--frame_rate", str(args.frame_rate),
                "--dataset_type", args.dataset_type,
                "--child"
            ]
            if args.use_fp16:
                cmd.append("--use_fp16")
            if args.save_video:
                cmd.append("--save_video")
                cmd.extend(["--video_out", video_path])
            if args.show_gt_boxes:
                cmd.append("--show_gt_boxes")
            for k, v in (tracker_kv or {}).items():
                cmd.extend(["--tracker_kv", f"{k}={v}"])

            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
                "PYTHONWARNINGS": "ignore::UserWarning,ignore::FutureWarning",
                "TRANSFORMERS_VERBOSITY": "error",
                "MPLBACKEND": "Agg",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            })

            p = subprocess.Popen(cmd, env=env)
            procs.append(p)

            if len(procs) >= jobs:
                procs[0].wait()
                procs = procs[1:]

        for p in procs:
            p.wait()

    parser = argparse.ArgumentParser(description="Clean worker for GroundingDINO + Tracker evaluation")

    # Sequence selection
    parser.add_argument("--seq", nargs="*", help="Sequence names")
    parser.add_argument("--seq_file", type=str, help="File with sequence names")
    parser.add_argument("--seq_glob", action="append", help="Glob patterns for sequences")
    parser.add_argument("--all", action="store_true", help="Process all sequences")
    parser.add_argument("--img_folder", required=True, type=str, help="Image root folder")

    # Output
    parser.add_argument("--out", type=str, help="Output file/folder")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--timestamp", action="store_true", help="Add timestamp to output")
    parser.add_argument("--video_out", type=str, help="Video output path")
    parser.add_argument("--save_video", action="store_true", help="Save tracking video")
    parser.add_argument("--show_gt_boxes", action="store_true", help="Show GT boxes in video")

    # Dataset
    parser.add_argument("--dataset_type", choices=["mot", "referkitti"], default="mot", help="Dataset type")

    # Detector
    parser.add_argument("--detector", choices=["dino", "florence2"], default="dino", help="Detector type")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Model config path")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS_PATH, help="Model weights path")
    parser.add_argument("--box_thresh", type=float, default=0.35, help="Box threshold")
    parser.add_argument("--text_thresh", type=float, default=0.25, help="Text threshold")
    parser.add_argument("--text_prompt", type=str, default=DEFAULT_TEXT_PROMPT, help="Text prompt")
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16")

    # Tracker
    parser.add_argument("--tracker", default="bytetrack", choices=list(TRACKER_REGISTRY.keys()), help="Tracker type")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="Track threshold")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="Match threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="Track buffer")
    parser.add_argument("--tracker_kv", action="append", help="Tracker key=value args")

    # CLIP
    parser.add_argument("--lambda_weight", type=float, default=0.25, help="CLIP fusion weight")
    parser.add_argument("--low_thresh", type=float, default=0.1, help="Low detection threshold")
    parser.add_argument("--text_sim_thresh", type=float, default=0.0, help="Min CLIP text similarity")
    parser.add_argument("--use_clip_in_high", action="store_true", help="Use CLIP in high conf stage")
    parser.add_argument("--use_clip_in_low", action="store_true", help="Use CLIP in low conf stage")
    parser.add_argument("--use_clip_in_unconf", action="store_true", help="Use CLIP in unconf stage")

    # Misc
    parser.add_argument("--min_box_area", type=int, default=DEFAULT_MIN_BOX_AREA, help="Min box area")
    parser.add_argument("--frame_rate", type=int, default=DEFAULT_FRAME_RATE, help="Frame rate")

    # Multi-GPU
    parser.add_argument("--devices", type=str, help="GPU IDs (comma-separated)")
    parser.add_argument("--jobs", type=int, default=1, help="Max concurrent jobs")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    tracker_kwargs = {
        "track_thresh": args.track_thresh,
        "track_buffer": args.track_buffer,
        "match_thresh": args.match_thresh,
        "lambda_weight": args.lambda_weight,
        "low_thresh": args.low_thresh,
        "text_sim_thresh": args.text_sim_thresh,
        "use_clip_in_high": args.use_clip_in_high,
        "use_clip_in_low": args.use_clip_in_low,
        "use_clip_in_unconf": args.use_clip_in_unconf,
    }
    tracker_kwargs.update(parse_kv_list(args.tracker_kv))

    # Child mode
    if args.child:
        if not args.seq or len(args.seq) != 1 or not args.out:
            raise SystemExit("Child mode needs exactly one --seq and --out")
        worker = Worker(
            tracker_type=args.tracker,
            tracker_kwargs=tracker_kwargs,
            box_thresh=args.box_thresh,
            text_thresh=args.text_thresh,
            use_fp16=args.use_fp16,
            text_prompt=args.text_prompt,
            detector=args.detector,
            frame_rate=args.frame_rate,
            save_video=args.save_video,
            show_gt_boxes=args.show_gt_boxes,
            dataset_type=args.dataset_type,
            min_box_area=args.min_box_area,
            config_path=args.config,
            weights_path=args.weights,
        )
        out_path = args.out if args.out.lower().endswith(".txt") else os.path.join(args.out, f"{args.seq[0]}.txt")
        video_out = args.video_out if hasattr(args, 'video_out') and args.video_out else None
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        worker.process_sequence(
            seq=args.seq[0],
            img_folder=args.img_folder,
            gt_folder=os.path.join(args.img_folder, ".."),
            out_path=out_path,
            video_out_path=video_out
        )
        raise SystemExit(0)

    # Parent mode
    seqs = collect_sequences(args)

    if args.devices and len(seqs) > 1:
        dispatch_multi_gpu(seqs, args, tracker_kwargs)
    else:
        worker = Worker(
            tracker_type=args.tracker,
            tracker_kwargs=tracker_kwargs,
            box_thresh=args.box_thresh,
            text_thresh=args.text_thresh,
            use_fp16=args.use_fp16,
            text_prompt=args.text_prompt,
            detector=args.detector,
            frame_rate=args.frame_rate,
            save_video=args.save_video,
            show_gt_boxes=args.show_gt_boxes,
            dataset_type=args.dataset_type,
            min_box_area=args.min_box_area,
            config_path=args.config,
            weights_path=args.weights,
        )

        if len(seqs) == 1:
            out_path = resolve_single_out(seqs[0], args.out, args.out_dir, args.timestamp)
            worker.process_sequence(
                seq=seqs[0],
                img_folder=args.img_folder,
                gt_folder=os.path.join(args.img_folder, ".."),
                out_path=out_path
            )
        else:
            root = args.out_dir
            if args.timestamp:
                root = os.path.join(root, datetime.now().strftime("%Y-%m-%d_%H%M"))
            os.makedirs(root, exist_ok=True)

            if worker.save_video:
                video_folder = root.replace("/results", "/videos").replace("\\results", "\\videos")
                if "results" not in root:
                    video_folder = os.path.join(os.path.dirname(root), "videos")
                os.makedirs(video_folder, exist_ok=True)

            for s in seqs:
                out_path = os.path.join(root, f"{s}.txt")
                video_path = os.path.join(video_folder, f"{s}.mp4") if worker.save_video else None
                worker.process_sequence(
                    seq=s,
                    img_folder=args.img_folder,
                    out_path=out_path,
                    gt_folder=os.path.join(args.img_folder, ".."),
                    video_out_path=video_path
                )

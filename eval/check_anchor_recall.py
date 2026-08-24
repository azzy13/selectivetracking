#!/usr/bin/env python3
"""
Week 3 validation: does the anchor now appear as a candidate box / graph node?

Runs the Week 2 sweep frames (dataset/sweep) through the detector under three
captions and scores anchor recall exactly as 02_evaluate_grounding.ipynb does —
a clip counts as recalled when ANY detection matches the bus GT box at IoU >= 0.5:

    BEFORE (Week 2)   "red car"                 — what the pipeline actually
                                                  prompted with: target class only
    raw prompt        "red car behind the bus"  — notebook variant 3, for reference
    AFTER  (Week 3)   "red car . bus"           — build_detector_prompt(parse(prompt))

Nothing else differs.  Same frames, same weights, same thresholds, same metric.

The BEFORE row is the ~0 baseline: with only the target class in the caption the
bus is never proposed, so it can never be a node, so the relation has nothing to
attach to.  The raw-prompt row is worth keeping in view — GroundingDINO does
ground a bus box from the full sentence, but the pipeline never used that caption
and its hard filter dropped the box anyway.

Also reported, because they are the rest of the Week 3 contract:
  - how many detections get tagged ROLE_ANCHOR by assign_detection_roles
  - that anchors are excluded from the emitted set (emitted_tracks)

Usage:
    python eval/check_anchor_recall.py
    python eval/check_anchor_recall.py --prompt "red car behind the bus" --json_out out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPT_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.ops import box_convert  # noqa: E402

import groundingdino.datasets.transforms as T  # noqa: E402
from groundingdino.util.inference import load_model, predict  # noqa: E402

from query_grounding import (ROLE_ANCHOR, assign_detection_roles,  # noqa: E402
                             build_detector_prompt)
from query_parser import parse  # noqa: E402

CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
WEIGHTS_PATH = "weights/groundingdino_swinb_cogcoor.pth"
SWEEP_ROOT = os.path.join(ROOT, "dataset", "sweep")
RECORDED_ROOT = "runs/sweep"      # what the manifest paths were written with

_TRANSFORM = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Sweep loading — same rebasing the notebook does
# ---------------------------------------------------------------------------

def load_clips(sweep_root: str):
    with open(os.path.join(sweep_root, "manifest.json")) as fh:
        manifest = json.load(fh)
    for clip in manifest["clips"]:
        for key in ("rgb_dir", "gt_dir"):
            if not os.path.isabs(clip[key]):
                clip[key] = os.path.join(sweep_root,
                                         os.path.relpath(clip[key], RECORDED_ROOT))
    return manifest, manifest["clips"]


def load_frame(clip):
    i = clip["frames"] // 2
    img = np.array(Image.open(os.path.join(clip["rgb_dir"], f"{i:06d}.png")).convert("RGB"))
    with open(os.path.join(clip["gt_dir"], f"{i:06d}.json")) as fh:
        graph = json.load(fh)
    return img, graph


def gt_boxes_by_role(clip, graph):
    """{'bus': [x1,y1,x2,y2], 'target': ..., 'distractor': ...}"""
    role_of = {v: k for k, v in clip["ids"].items()}
    return {role_of[n["id"]]: n["box2d"] for n in graph["nodes"] if n["id"] in role_of}


# ---------------------------------------------------------------------------
# Detection + the notebook's scoring
# ---------------------------------------------------------------------------

def run_inference(model, image, caption, box_thresh, text_thresh, device):
    """-> list of (box_xyxy_pixels, score, phrase)."""
    h, w = image.shape[:2]
    tensor, _ = _TRANSFORM(Image.fromarray(image), None)
    boxes, scores, phrases = predict(
        model, tensor, caption, box_threshold=box_thresh,
        text_threshold=text_thresh, device=device, remove_combined=True,
    )
    if len(boxes) == 0:
        return []
    xyxy = box_convert(boxes * torch.tensor([w, h, w, h]), "cxcywh", "xyxy").numpy()
    return [(b.tolist(), float(s), p) for b, s, p in zip(xyxy, scores, phrases)]


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_role(box, gt, iou_thresh: float):
    best_role, best = None, iou_thresh
    for role, gbox in gt.items():
        v = iou(box, gbox)
        if v >= best:
            best_role, best = role, v
    return best_role


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep_root", default=SWEEP_ROOT)
    ap.add_argument("--prompt", default=None,
                    help="defaults to the sweep manifest's own prompt")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--box_thresh", type=float, default=0.25)
    ap.add_argument("--text_thresh", type=float, default=0.25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args(argv)

    manifest, clips = load_clips(args.sweep_root)
    prompt = args.prompt or manifest["prompt"]
    query = parse(prompt)
    if query.anchor is None:
        ap.error(f"{prompt!r} parses to anchor=None — no anchor to measure.")

    from query_grounding import entity_phrase
    captions = {
        "BEFORE (Week 2)": entity_phrase(query.target),   # target class only
        "raw prompt":      prompt,                        # reference
        "AFTER  (Week 3)": build_detector_prompt(query),  # target . anchor
    }

    print(f"prompt   \"{prompt}\"")
    print(f"clips    {len(clips)} from {args.sweep_root}")
    print(f"metric   anchor recall = any detection matches the bus at IoU >= {args.iou}")

    model = load_model(CONFIG_PATH, WEIGHTS_PATH, device=args.device).to(args.device)

    results = {}
    for label, caption in captions.items():
        hits = 0
        anchor_tagged = 0
        n_det = 0
        per_clip = []
        for clip in clips:
            img, graph = load_frame(clip)
            gt = gt_boxes_by_role(clip, graph)
            dets = run_inference(model, img, caption, args.box_thresh,
                                 args.text_thresh, args.device)
            roles = [match_role(b, gt, args.iou) for b, _, _ in dets]
            det_roles = assign_detection_roles([p for _, _, p in dets], query)

            hit = "bus" in roles
            hits += hit
            anchor_tagged += sum(1 for r in det_roles if r == ROLE_ANCHOR)
            n_det += len(dets)
            per_clip.append({
                "config_id": clip["config_id"],
                "condition": clip["condition"],
                "anchor_recall": bool(hit),
                "n_det": len(dets),
            })

        results[label] = {
            "caption": caption,
            "anchor_recall": hits / len(clips),
            "clips_hit": hits,
            "clips": len(clips),
            "dets_per_clip": round(n_det / len(clips), 2),
            "anchor_tagged_per_clip": round(anchor_tagged / len(clips), 2),
            "per_clip": per_clip,
        }
        print(f"\n{label}  caption=\"{caption}\"")
        print(f"   anchor recall {hits}/{len(clips)} = {hits / len(clips):.3f}"
              f"   |  {results[label]['dets_per_clip']} det/clip,"
              f" {results[label]['anchor_tagged_per_clip']} tagged anchor")

    before = results["BEFORE (Week 2)"]["anchor_recall"]
    after = results["AFTER  (Week 3)"]["anchor_recall"]
    print("\n" + "=" * 70)
    print(f"WEEK 3 RESULT   anchor recall {before:.3f} → {after:.3f}   ({after - before:+.3f})")
    print(f"anchors emitted: no — emitted_tracks() returns target candidates only")
    print("=" * 70)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"prompt": prompt, "query": query.to_dict(),
                       "iou_thresh": args.iou, "runs": results}, fh, indent=2)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Refer-KITTI → prompt-compliance ground truth.

The SP / SR / PCR / DCR / SID metrics (``carla_sim/metrics.md``) are defined
over a ground truth in which **every box carries a per-frame prompt-validity
flag**.  A box that is not prompt-valid is a *distractor* — that is what makes
DCR different from a plain false-positive rate.

Refer-KITTI already holds exactly that information, split across two files:

    KITTI/<split>/labels_with_ids/image_02/<seq>/<frame>.txt
        every annotated object in the frame  (class, track id, normalised box)

    expression/<seq>/<expression>.json
        {"label": {"<frame>": [track_id, ...]}, "sentence": "cars in left"}
        which of those track ids the sentence refers to, **frame by frame**

This module joins them into the same dict schema that
``carla_sim/evaluate_prompt_metrics.py`` consumes for CARLA, so both datasets
are scored by one implementation of the metrics rather than two.

    gt_data = {
        "meta": {"prompt": str, ...},
        "annotations": [
            {"image_id": int,        # frame number, as in the image filename
             "gt_id":    int,        # unique per annotation
             "track_id": int,        # Refer-KITTI object id
             "bbox_xyxy": [x1, y1, x2, y2],
             "is_target": bool},     # prompt-valid *in this frame*
            ...
        ],
    }

Two details worth knowing:

* **Per-frame validity, not per-sequence.**  ``eval_referkitti.py`` unions the
  referred ids over the whole clip and treats them as valid in every frame.
  That is looser than the annotation: an object referred to only in frames
  10–38 would score as valid in frame 200 as well.  Here the ``label`` map is
  read frame by frame, which is what "prompt-valid" means in the metrics doc.

* **Boxes are top-left, not centre.**  Despite the YOLO-looking layout, the
  Refer-KITTI ``labels_with_ids`` columns are ``x_left y_top w h`` normalised
  by image size.  ``eval_referkitti.py`` carries the same note; getting it
  wrong shifts every box by half its size and IoU matching quietly collapses.

Standalone use (prints a summary of one expression, no model involved):

    python eval/referkitti_prompt_gt.py \
        --data_root dataset/referkitti --sequence 0005 \
        --expression cars-in-left
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

# Written by Refer-KITTI as <frame>.png; the number is the frame id the
# Worker also writes into its MOT output (worker_clean.parse_frame_id).
_IMAGE_EXTS = (".png", ".jpg", ".jpeg")


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
def sequence_image_dir(data_root: str, seq: str, split: str = "training") -> str:
    return os.path.join(data_root, "KITTI", split, "image_02", seq)


def sequence_label_dir(data_root: str, seq: str, split: str = "training") -> str:
    return os.path.join(data_root, "KITTI", split, "labels_with_ids", "image_02", seq)


def expression_dir(data_root: str, seq: str) -> str:
    return os.path.join(data_root, "expression", seq)


def list_sequences(data_root: str, split: str = "training") -> List[str]:
    """Sequences that have both images and expressions."""
    images_root = os.path.join(data_root, "KITTI", split, "image_02")
    if not os.path.isdir(images_root):
        raise FileNotFoundError(f"image_02 not found: {images_root}")
    seqs = sorted(
        d for d in os.listdir(images_root)
        if os.path.isdir(os.path.join(images_root, d))
        and os.path.isdir(expression_dir(data_root, d))
    )
    return seqs


def list_expressions(data_root: str, seq: str) -> List[str]:
    """Expression *names* (JSON stems) available for a sequence, sorted."""
    paths = sorted(glob.glob(os.path.join(expression_dir(data_root, seq), "*.json")))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def sequence_image_size(data_root: str, seq: str, split: str = "training"):
    """(width, height) of the sequence, read from its first frame.

    KITTI frames are a constant size within a sequence but differ between
    sequences (1242x375 for 0005, others vary), so this is read once per
    sequence rather than once per frame.
    """
    img_dir = sequence_image_dir(data_root, seq, split)
    frames = sorted(
        f for f in os.listdir(img_dir) if f.lower().endswith(_IMAGE_EXTS)
    )
    if not frames:
        raise FileNotFoundError(f"No images in {img_dir}")

    from PIL import Image  # local import: keeps this module importable without cv2

    with Image.open(os.path.join(img_dir, frames[0])) as im:
        return im.size  # (W, H)


def load_expression(data_root: str, seq: str, name: str) -> dict:
    """Load one expression JSON.

    Returns::

        {"name":     "cars-in-left",
         "sequence": "0005",
         "sentence": "cars in left",
         "valid_ids_by_frame": {frame_id: {track_id, ...}}}

    The ``ignore`` field is read but asserted empty: it is empty in all 818
    expression files of this release, so no ignore-region handling exists
    downstream.  A non-empty one would silently be scored as a distractor,
    hence the loud failure.
    """
    path = os.path.join(expression_dir(data_root, seq), f"{name}.json")
    with open(path) as f:
        data = json.load(f)

    ignore = {k: v for k, v in (data.get("ignore") or {}).items() if v}
    if ignore:
        raise NotImplementedError(
            f"{path} has a non-empty 'ignore' map ({list(ignore)[:3]}...); "
            "ignore regions are not handled by the prompt metrics."
        )

    valid_by_frame: Dict[int, set] = {}
    for frame, ids in (data.get("label") or {}).items():
        if isinstance(ids, (int, str)):
            ids = [ids]
        valid_by_frame[int(frame)] = {int(i) for i in ids}

    return {
        "name": name,
        "sequence": seq,
        "sentence": data["sentence"],
        "valid_ids_by_frame": valid_by_frame,
    }


def load_sequence_boxes(
    data_root: str, seq: str, split: str = "training"
) -> Dict[int, List[dict]]:
    """Every annotated object in the sequence, as ``{frame_id: [box, ...]}``.

    Each box is ``{"track_id": int, "class_id": int, "bbox_xyxy": [...]}`` in
    absolute pixels.
    """
    label_dir = sequence_label_dir(data_root, seq, split)
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"labels_with_ids not found: {label_dir}")

    W, H = sequence_image_size(data_root, seq, split)

    boxes_by_frame: Dict[int, List[dict]] = defaultdict(list)
    for label_path in sorted(glob.glob(os.path.join(label_dir, "*.txt"))):
        stem = os.path.splitext(os.path.basename(label_path))[0]
        try:
            frame_id = int(stem)
        except ValueError:
            continue

        # A frame with no objects still exists as an empty file; keep the key
        # so "frames with no valid GT" is distinguishable from "frame missing".
        boxes_by_frame.setdefault(frame_id, [])

        with open(label_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 6:
                    continue
                # NOTE: top-left coords despite the YOLO-style layout.
                class_id = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x1 = float(parts[2]) * W
                y1 = float(parts[3]) * H
                bw = float(parts[4]) * W
                bh = float(parts[5]) * H
                boxes_by_frame[frame_id].append({
                    "track_id": track_id,
                    "class_id": class_id,
                    "bbox_xyxy": [x1, y1, x1 + bw, y1 + bh],
                })

    return dict(boxes_by_frame)


# ----------------------------------------------------------------------
# The join
# ----------------------------------------------------------------------
def build_prompt_gt(
    boxes_by_frame: Dict[int, List[dict]],
    expression: dict,
    *,
    restrict_to_labelled_frames: bool = True,
) -> dict:
    """Join sequence boxes with one expression into CARLA-schema ``gt_data``.

    ``restrict_to_labelled_frames`` keeps only frames the expression actually
    annotates.  Refer-KITTI's ``label`` map covers a contiguous span that can
    be shorter than the clip (e.g. 284 of 297 frames for ``0005``); the frames
    outside it are *unannotated*, not "annotated as containing nothing", so
    scoring predictions there would charge the tracker for GT that was never
    written.  Set it False to score the whole clip instead.
    """
    valid_by_frame = expression["valid_ids_by_frame"]

    if restrict_to_labelled_frames:
        frame_ids = sorted(set(boxes_by_frame) & set(valid_by_frame))
    else:
        frame_ids = sorted(boxes_by_frame)

    annotations: List[dict] = []
    referred_missing = 0  # ids the expression names that have no box that frame
    gt_id = 0

    for frame_id in frame_ids:
        frame_boxes = boxes_by_frame.get(frame_id, [])
        valid_ids = valid_by_frame.get(frame_id, set())
        present_ids = {b["track_id"] for b in frame_boxes}
        referred_missing += len(valid_ids - present_ids)

        for box in frame_boxes:
            annotations.append({
                "image_id": frame_id,
                "gt_id": gt_id,
                "track_id": box["track_id"],
                "class_id": box["class_id"],
                "bbox_xyxy": box["bbox_xyxy"],
                "is_target": box["track_id"] in valid_ids,
            })
            gt_id += 1

    num_valid = sum(a["is_target"] for a in annotations)
    return {
        "meta": {
            "prompt": expression["sentence"],
            "dataset": "referkitti",
            "sequence": expression["sequence"],
            "expression": expression["name"],
            "frames": len(frame_ids),
            "frame_range": [frame_ids[0], frame_ids[-1]] if frame_ids else None,
            "num_annotations": len(annotations),
            "num_valid": num_valid,
            "num_distractors": len(annotations) - num_valid,
            "referred_ids_without_box": referred_missing,
        },
        "annotations": annotations,
    }


def load_prompt_gt(
    data_root: str,
    seq: str,
    expression_name: str,
    split: str = "training",
    *,
    boxes_by_frame: Optional[Dict[int, List[dict]]] = None,
    restrict_to_labelled_frames: bool = True,
) -> dict:
    """``load_sequence_boxes`` + ``load_expression`` + ``build_prompt_gt``.

    Pass ``boxes_by_frame`` to reuse one sequence load across expressions —
    the label files are re-read otherwise, once per expression.
    """
    if boxes_by_frame is None:
        boxes_by_frame = load_sequence_boxes(data_root, seq, split)
    expression = load_expression(data_root, seq, expression_name)
    return build_prompt_gt(
        boxes_by_frame, expression,
        restrict_to_labelled_frames=restrict_to_labelled_frames,
    )


# ----------------------------------------------------------------------
# CLI — inspect one expression without loading any model
# ----------------------------------------------------------------------
def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data_root", default="dataset/referkitti")
    ap.add_argument("--split", default="training")
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--expression", help="expression name; omit to list them all")
    args = ap.parse_args()

    names = list_expressions(args.data_root, args.sequence)
    if args.expression is None:
        print(f"{len(names)} expressions for sequence {args.sequence}:\n")
        boxes = load_sequence_boxes(args.data_root, args.sequence, args.split)
        total = sum(len(v) for v in boxes.values())
        print(f"  sequence GT: {len(boxes)} frames, {total} boxes\n")
        for name in names:
            gt = load_prompt_gt(args.data_root, args.sequence, name, args.split,
                                boxes_by_frame=boxes)
            m = gt["meta"]
            print(f"  {name:52} {m['num_valid']:5d} valid / "
                  f"{m['num_distractors']:5d} distractor  "
                  f"over {m['frames']:4d} frames   \"{m['prompt']}\"")
        return

    gt = load_prompt_gt(args.data_root, args.sequence, args.expression, args.split)
    print(json.dumps(gt["meta"], indent=2))
    print(f"\nfirst 5 annotations:")
    for ann in gt["annotations"][:5]:
        print(f"  {ann}")


if __name__ == "__main__":
    _main()

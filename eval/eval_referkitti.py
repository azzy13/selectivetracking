#!/usr/bin/env python3
"""
Refer-KITTI RMOT-style evaluation for your GroundingDINO + (ByteTrack / CLIPTracker) pipeline.

For each sequence:
  - load expression/<seq>.json
  - for each expression:
      * use the expression text as text_prompt
      * run tracking on that sequence
      * build GT filtered to the referenced object ids
      * evaluate tracking only on those ids

Usage (example):
    python3 eval/eval_referkitti_rmot.py \
        --data_root dataset/referkitti \
        --tracker bytetrack \
        --detector dino \
        --devices 0,1 \
        --jobs 2
"""

import os
import cv2
import argparse
import json
from datetime import datetime
import shutil

import glob as _glob

import pandas as pd
import torch

from compute_metrics import MotMetricsEvaluator
# Import your Worker + helper from worker.py (same folder)
from worker_clean import Worker, parse_kv_list

# ----------------------------------------------------------------------
# Dataset defaults
# ----------------------------------------------------------------------
DATASET_DEFAULTS = {
    "referkitti": {
        "frame_rate": 10,   # KITTI is 10 FPS
    }
}

# ----------------------------------------------------------------------
# Utility: symlink KITTI image folders (image_02/0000 → run/images/0000)
# ----------------------------------------------------------------------
def create_kitti_image_symlinks(data_root, kitti_split, out_folder):
    """
    data_root: path to referkitti root (contains KITTI/, expression/)
    kitti_split: usually 'training'
    out_folder: where we make seq symlinks
    """
    images_root = os.path.join(data_root, "KITTI", kitti_split, "image_02")
    if not os.path.isdir(images_root):
        raise FileNotFoundError(f"image_02 folder not found: {images_root}")

    os.makedirs(out_folder, exist_ok=True)
    sequences = sorted(
        d for d in os.listdir(images_root)
        if os.path.isdir(os.path.join(images_root, d))
    )

    print(f"\n🔗 Creating image symlinks for {len(sequences)} KITTI sequences...")
    for seq in sequences:
        src = os.path.abspath(os.path.join(images_root, seq))
        dst = os.path.abspath(os.path.join(out_folder, seq))

        if not os.path.exists(src):
            print(f"   ✗ Missing image dir: {src}")
            continue
        if os.path.exists(dst):
            continue

        try:
            os.symlink(src, dst, target_is_directory=True)
            print(f"   ✓ {seq}")
        except OSError:
            shutil.copytree(src, dst)
            print(f"   ✓ {seq} (copied)")

    return sequences


# ----------------------------------------------------------------------
# Utility: build sequence-level MOT-style GT from labels_with_ids
# ----------------------------------------------------------------------
def build_referkitti_seq_gt(data_root, kitti_split, out_folder, seq_whitelist=None):
    """
    Convert YOLO-style labels_with_ids per-frame KITTI files to MOTChallenge-style
    <seq>.txt with rows:
        frame, id, bb_left, bb_top, bb_width, bb_height, 1, -1, -1, -1

    Assumes each labels_with_ids line is:
        class_id track_id x_center y_center width height   (all normalized 0..1)
    """
    labels_root = os.path.join(
        data_root, "KITTI", kitti_split, "labels_with_ids", "image_02"
    )
    images_root = os.path.join(
        data_root, "KITTI", kitti_split, "image_02"
    )
    if not os.path.isdir(labels_root):
        raise FileNotFoundError(f"labels_with_ids/image_02 not found: {labels_root}")

    os.makedirs(out_folder, exist_ok=True)

    num_seqs = 0
    for seq in sorted(os.listdir(labels_root)):
        if seq_whitelist is not None and seq not in seq_whitelist:
            continue

        seq_dir = os.path.join(labels_root, seq)
        if not os.path.isdir(seq_dir):
            continue

        gt_out = os.path.join(out_folder, f"{seq}.txt")
        with open(gt_out, "w") as fout:
            frame_files = [
                f for f in os.listdir(seq_dir)
                if os.path.isfile(os.path.join(seq_dir, f)) and f.lower().endswith(".txt")
            ]
            frame_files = sorted(frame_files)

            for fname in frame_files:
                stem = os.path.splitext(fname)[0]
                try:
                    frame_id = int(stem)
                except ValueError:
                    continue

                # Get image size (375x1242), try .png then .jpg
                img_path_png = os.path.join(images_root, seq, f"{frame_id:06d}.png")
                img_path_jpg = os.path.join(images_root, seq, f"{frame_id:06d}.jpg")
                img = cv2.imread(img_path_png)
                if img is None:
                    img = cv2.imread(img_path_jpg)
                if img is None:
                    print(f"   ⚠ Could not read image for {seq}/{frame_id:06d}, skipping frame.")
                    continue

                H, W = img.shape[:2]

                label_path = os.path.join(seq_dir, fname)
                with open(label_path, "r") as fin:
                    for line in fin:
                        line = line.strip()
                        if not line:
                            continue

                        # whitespace-separated
                        parts = line.split()
                        if len(parts) < 6:
                            continue

                        try:
                            # class_id = int(parts[0])   # unused
                            track_id = int(float(parts[1]))
                            # NOTE: Despite field names, ReferKITTI stores TOP-LEFT coords, not center!
                            x_left = float(parts[2]) * W
                            y_top  = float(parts[3]) * H
                            bw  = float(parts[4]) * W
                            bh  = float(parts[5]) * H
                        except ValueError:
                            continue

                        bb_left = x_left
                        bb_top  = y_top

                        fout.write(
                            f"{frame_id},{track_id},{bb_left:.2f},{bb_top:.2f},"
                            f"{bw:.2f},{bh:.2f},1,-1,-1,-1\n"
                        )

        num_seqs += 1

    print(f"\n📋 Built MOT-style sequence GT for {num_seqs} sequences → {out_folder}")


# ----------------------------------------------------------------------
# Utility: load referring expressions for a sequence
# ----------------------------------------------------------------------
def load_expressions_for_sequence(expr_root: str, seq: str):
    """
    Load all referring expressions for a sequence.

    Supports two layouts:

      A) Folder with many JSONs per seq (Refer-KITTI):
         expression/0001/*.json
         Each file looks like:
           {
             "label": { "1": [11], "2": [11], ... },
             "ignore": { ... },
             "video_name": "0001",
             "sentence": "black cars in the right"
           }

      B) Generic formats (list/dict with 'expressions', 'annotations', etc.)

    Returns a list of dicts:
        { "expr_id": int, "text": str, "obj_ids": List[int] }
    """
    # --- Case A: folder per sequence with many JSONs ---
    seq_dir = os.path.join(expr_root, seq)
    json_paths = []
    if os.path.isdir(seq_dir):
        json_paths = sorted(_glob.glob(os.path.join(seq_dir, "*.json")))

    # --- Case B fallback: single file expression/0001.json ---
    single_path = os.path.join(expr_root, f"{seq}.json")
    if os.path.isfile(single_path):
        json_paths.append(single_path)

    if not json_paths:
        print(f"   ⚠ No expression JSON for sequence {seq}: {seq_dir} or {single_path}")
        return []

    TEXT_KEYS = ["sentence", "expression", "exp", "text", "caption"]
    ID_KEYS = ["obj_ids", "object_ids", "track_ids", "track_id_list", "ids"]

    result = []
    expr_counter = 0

    for jp in json_paths:
        with open(jp, "r") as f:
            data = json.load(f)

        # --- Special case: Refer-KITTI style (label + sentence) ---
        if isinstance(data, dict) and "sentence" in data and "label" in data:
            text = data["sentence"]
            label_map = data.get("label", {})

            id_set = set()
            for _, ids in label_map.items():
                if isinstance(ids, (int, str)):
                    id_set.add(int(ids))
                else:
                    for tid in ids:
                        id_set.add(int(tid))

            if id_set:
                result.append(
                    {
                        "expr_id": expr_counter,
                        "text": text,
                        "obj_ids": sorted(id_set),
                    }
                )
                expr_counter += 1
            continue  # next JSON file

        # --- Generic fallback (for other datasets / formats) ---
        if isinstance(data, dict):
            candidates = (
                data.get("expressions")
                or data.get("annotations")
                or data.get("refs")
                or data.get("data")
            )
            if candidates is None:
                candidates = [data]
            items = candidates
        elif isinstance(data, list):
            items = data
        else:
            print(f"   ⚠ Unsupported JSON structure in {jp}, skipping.")
            continue

        for item in items:
            if not isinstance(item, dict):
                continue

            text = None
            for k in TEXT_KEYS:
                if k in item:
                    text = item[k]
                    break

            obj_ids = None
            for k in ID_KEYS:
                if k in item:
                    obj_ids = item[k]
                    break

            if text is None or obj_ids is None:
                continue

            if isinstance(obj_ids, (int, str)):
                obj_ids = [obj_ids]
            obj_ids = [int(x) for x in obj_ids]

            result.append(
                {
                    "expr_id": expr_counter,
                    "text": text,
                    "obj_ids": obj_ids,
                }
            )
            expr_counter += 1

    print(f"   ✓ Loaded {len(result)} expressions for seq {seq}")
    return result

# ----------------------------------------------------------------------
# Utility: build expression-specific GT from sequence GT
# ----------------------------------------------------------------------
def build_expr_gt_from_seq(gt_seq_dir: str, seq: str, expr, out_dir: str):
    """
    gt_seq_dir: directory containing <seq>.txt (sequence-level GT)
    seq: e.g. '0000'
    expr: dict with keys 'expr_id', 'obj_ids'
    out_dir: where to write <seq>_exprXXXX.txt
    """
    os.makedirs(out_dir, exist_ok=True)
    seq_gt_path = os.path.join(gt_seq_dir, f"{seq}.txt")
    if not os.path.isfile(seq_gt_path):
        raise FileNotFoundError(seq_gt_path)

    df = pd.read_csv(
        seq_gt_path,
        header=None,
        names=[
            "frame",
            "id",
            "bb_left",
            "bb_top",
            "bb_width",
            "bb_height",
            "conf",
            "x",
            "y",
            "z",
        ],
    )
    target_ids = set(expr["obj_ids"])
    df_expr = df[df["id"].isin(target_ids)]

    expr_name = f"{seq}_expr{expr['expr_id']:04d}"
    out_path = os.path.join(out_dir, f"{expr_name}.txt")
    df_expr.to_csv(out_path, header=False, index=False)
    return expr_name, out_path


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Refer-KITTI RMOT-style evaluation using Worker + motmetrics"
    )
    ap.add_argument(
        "--data_root",
        required=True,
        help="Path to referkitti root (contains KITTI/ and expression/).",
    )
    ap.add_argument(
        "--kitti_split",
        default="training",
        choices=["training"],
        help="Currently only 'training' is defined in Refer-KITTI.",
    )
    ap.add_argument(
        "--box_threshold", type=float, default=0.40, help="GroundingDINO box threshold"
    )
    ap.add_argument(
        "--text_threshold", type=float, default=0.80, help="GroundingDINO text threshold"
    )
    ap.add_argument(
        "--use_scale_aware_thresh",
        action="store_true",
        default=True,
        help="Enable scale-aware thresholding (lower threshold for small/distant objects)"
    )
    ap.add_argument(
        "--no_scale_aware_thresh",
        dest="use_scale_aware_thresh",
        action="store_false",
        help="Disable scale-aware thresholding"
    )
    ap.add_argument(
        "--small_box_area_thresh",
        type=int,
        default=5000,
        help="Area threshold (pixels) for small boxes in scale-aware filtering (default: 5000)"
    )
    ap.add_argument(
        "--track_thresh", type=float, default=0.45, help="Tracker high-conf threshold"
    )
    ap.add_argument(
        "--match_thresh", type=float, default=0.85, help="Tracker matching threshold"
    )
    ap.add_argument(
        "--track_buffer", type=int, default=120, help="Tracker buffer length"
    )
    ap.add_argument(
        "--tracker", choices=["bytetrack", "clip", "smartclip"], default="bytetrack"
    )
    ap.add_argument(
        "--detector", choices=["dino", "florence2"], default="dino"
    )
    ap.add_argument(
        "--config",
        type=str,
        default="groundingdino/config/GroundingDINO_SwinB_cfg.py",
    )
    ap.add_argument(
        "--weights",
        type=str,
        default="weights/swinb_light_visdrone_ft_best.pth",
    )
    ap.add_argument("--min_box_area", type=int, default=10)
    ap.add_argument(
        "--frame_rate", type=int, default=10, help="Override frame rate (default 10)."
    )
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--show_gt_boxes", action="store_true", help="Show GT boxes in saved videos")
    ap.add_argument(
        "--outdir", type=str, default=None, help="Root output dir (default: auto)"
    )

    # Tracker extra args (same semantics as worker.py CLI)
    ap.add_argument(
        "--lambda_weight",
        type=float,
        default=0.25,
        help="Weight for CLIP embedding cost in IoU+CLIP fusion (0=IoU only, 1=CLIP only)",
    )
    ap.add_argument(
        "--low_thresh",
        type=float,
        default=0.1,
        help="Low detection score threshold for second-stage association",
    )
    ap.add_argument(
        "--text_sim_thresh",
        type=float,
        default=0.0,
        help="Minimum CLIP text similarity for detection gating (0 disables gating)",
    )
    ap.add_argument(
        "--use_clip_in_high",
        action="store_true",
        help="Use CLIP fusion in high-confidence association stage",
    )
    ap.add_argument(
        "--use_clip_in_low",
        action="store_true",
        help="Use CLIP fusion in low-confidence association stage",
    )
    ap.add_argument(
        "--use_clip_in_unconf",
        action="store_true",
        help="Use CLIP fusion in unconfirmed stage",
    )

    # Text-grounding at matching args
    ap.add_argument(
        "--use_text_gate_matching",
        type=lambda x: x.lower() == 'true',
        default=True,
        help="Enable text-grounding cost at matching time (default: True)",
    )
    ap.add_argument(
        "--text_gate_mode",
        type=str,
        choices=["penalty", "hard"],
        default="penalty",
        help="Text-grounding mode: 'penalty' (soft gating) or 'hard' (block bad matches)",
    )
    ap.add_argument(
        "--text_gate_weight",
        type=float,
        default=0.5,
        help="Weight for text-grounding cost in penalty mode (default: 0.5)",
    )

    # Referring detection filter args
    ap.add_argument(
        "--referring_mode",
        type=str,
        choices=["none", "threshold"],
        default="threshold",
        help="Referring filter mode: 'threshold' (keep above similarity) or 'none' (disabled)",
    )
    ap.add_argument(
        "--referring_thresh",
        type=float,
        default=0.0,
        help="Minimum CLIP similarity when referring_mode='threshold' (default: 0.0)",
    )
    # Note: Both filters are enabled by default (True), so no flag means both are ON
    # Use --no_color_filter or --no_spatial_filter to disable
    color_group = ap.add_mutually_exclusive_group()
    color_group.add_argument(
        "--use_color_filter", "--use_colour_filter",
        dest="use_color_filter",
        action="store_true",
        default=None,
        help="Enable patch-based color filtering for color keywords (black/white/red/etc.)",
    )
    color_group.add_argument(
        "--no_color_filter", "--no_colour_filter",
        dest="use_color_filter",
        action="store_false",
        help="Disable patch-based color filtering",
    )

    spatial_group = ap.add_mutually_exclusive_group()
    spatial_group.add_argument(
        "--use_spatial_filter",
        action="store_true",
        default=None,
        help="Enable spatial filtering for spatial keywords (left/right/top/bottom/center)",
    )
    spatial_group.add_argument(
        "--no_spatial_filter",
        dest="use_spatial_filter",
        action="store_false",
        help="Disable spatial filtering",
    )

    # Pass-through tracker_kv if you want (optional)
    ap.add_argument(
        "--tracker_kv",
        action="append",
        help="extra tracker args as key=val (repeatable, passed to Worker)",
    )

    # Multi-GPU compatibility (similar CLI to eval_visdrone)
    ap.add_argument(
        "--devices",
        type=str,
        default="0",
        help="Comma-separated GPU ids to use (e.g. '0,1'). "
             "Expressions are assigned round-robin to these devices.",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Max concurrent jobs (placeholder; currently runs sequentially).",
    )

    args = ap.parse_args()

    # Set defaults for filter flags (True if not specified)
    if args.use_color_filter is None:
        args.use_color_filter = True
    if args.use_spatial_filter is None:
        args.use_spatial_filter = True
    if not hasattr(args, 'use_scale_aware_thresh'):
        args.use_scale_aware_thresh = True
    if not hasattr(args, 'small_box_area_thresh'):
        args.small_box_area_thresh = 5000

    dataset_name = "referkitti"
    defaults = DATASET_DEFAULTS[dataset_name]
    frame_rate = args.frame_rate or defaults["frame_rate"]

    # Parse devices list
    if args.devices:
        devices = [d.strip() for d in args.devices.split(",") if d.strip() != ""]
    else:
        devices = ["0"]
    if not devices:
        devices = ["0"]

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    if args.outdir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        run_outdir = os.path.join("outputs", f"{dataset_name}_rmot_{timestamp}")
    else:
        run_outdir = args.outdir

    os.makedirs(run_outdir, exist_ok=True)
    gt_seq_dir = os.path.join(run_outdir, "gt_seq")
    gt_expr_dir = os.path.join(run_outdir, "gt_expr")
    res_expr_dir = os.path.join(run_outdir, "results_expr")
    temp_images = os.path.join(run_outdir, "images")
    os.makedirs(gt_seq_dir, exist_ok=True)
    os.makedirs(gt_expr_dir, exist_ok=True)
    os.makedirs(res_expr_dir, exist_ok=True)
    os.makedirs(temp_images, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Refer-KITTI RMOT Evaluation (split: {args.kitti_split})")
    print(f"Output directory: {run_outdir}")
    print(f"Using devices: {devices} (round-robin per expression)")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Build sequence-level GT + images symlinks
    # ------------------------------------------------------------------

    images_root = os.path.join(args.data_root, "KITTI", args.kitti_split, "image_02")
    all_seqs = sorted(
        d for d in os.listdir(images_root)
        if os.path.isdir(os.path.join(images_root, d))
    )

    # Final eval: ALL sequences, first 4 expressions each
    seq_whitelist = all_seqs
    build_referkitti_seq_gt(args.data_root, args.kitti_split, gt_seq_dir, seq_whitelist=seq_whitelist)

    sequences = create_kitti_image_symlinks(args.data_root, args.kitti_split, temp_images)

    sequences = [s for s in sequences if s in seq_whitelist]
    print(f"Running on {len(sequences)} sequences, 4 expressions each (FINAL EVAL)")

    expr_root = os.path.join(args.data_root, "expression")

    # ------------------------------------------------------------------
    # Tracker kwargs (same mapping as worker.py main)
    # ------------------------------------------------------------------
    tracker_kwargs = dict(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
    )
    tracker_kwargs.update(
        {
            "lambda_weight": args.lambda_weight,
            "low_thresh": args.low_thresh,
            "text_sim_thresh": args.text_sim_thresh,
            "use_clip_in_high": args.use_clip_in_high,
            "use_clip_in_low": args.use_clip_in_low,
            "use_clip_in_unconf": args.use_clip_in_unconf,
            "use_text_gate_matching": args.use_text_gate_matching,
            "text_gate_mode": args.text_gate_mode,
            "text_gate_weight": args.text_gate_weight,
        }
    )
    tracker_kwargs.update(parse_kv_list(args.tracker_kv))

    # ------------------------------------------------------------------
    # Loop over sequences & expressions
    # ------------------------------------------------------------------
    total_expr = 0
    for seq in sequences:
        print(f"\n=== Sequence {seq} ===")
        expr_list = load_expressions_for_sequence(expr_root, seq)
        if not expr_list:
            print(f"   ⚠ No valid expressions for {seq}, skipping.")
            continue

        # Using first 2 expressions per sequence (TEST RUN)
        max_expr = 4
        expr_list = expr_list[:max_expr]
        print(f"   Using first {len(expr_list)} expressions for {seq}")

        # Load sequence-level GT once for this seq
        seq_gt_path = os.path.join(gt_seq_dir, f"{seq}.txt")
        if not os.path.isfile(seq_gt_path):
            print(f"   ⚠ Missing GT for {seq}, skipping.")
            continue

        seq_img_root = temp_images  # contains subdir per seq

        for expr in expr_list:
            # Choose GPU in round-robin
            gpu_id = devices[total_expr % len(devices)]
            if torch.cuda.is_available():
                device_str = f"cuda:{gpu_id}"
            else:
                device_str = "cpu"

            total_expr += 1
            expr_name = f"{seq}_expr{expr['expr_id']:04d}"
            print(f"\n   ▶ Expression {expr['expr_id']} ({expr_name}) on device {device_str}")
            print(f"      Text: {expr['text']}")
            print(f"      Obj IDs: {expr['obj_ids']}")

            # 1) Build expression-specific GT
            expr_name, expr_gt_path = build_expr_gt_from_seq(
                gt_seq_dir, seq, expr, gt_expr_dir
            )

            print(expr["text"])

            # 2) Instantiate a fresh Worker with this expression as text_prompt
            worker = Worker(
                tracker_type=args.tracker,
                tracker_kwargs=tracker_kwargs,
                box_thresh=args.box_threshold,
                text_thresh=args.text_threshold,
                use_fp16=args.fp16,
                text_prompt=expr["text"],
                detector=args.detector,
                frame_rate=frame_rate,
                save_video=args.save_video,
                show_gt_boxes=args.show_gt_boxes,
                dataset_type="referkitti",
                referkitti_data_root=args.data_root,
                target_object_ids=expr["obj_ids"],  # Pass target IDs for GT visualization
                min_box_area=args.min_box_area,
                config_path=args.config,
                weights_path=args.weights,
                device=device_str,
                referring_mode=args.referring_mode,
                referring_thresh=args.referring_thresh,
                use_spatial_filter=args.use_spatial_filter,
                use_color_filter=args.use_color_filter,
                use_scale_aware_thresh=args.use_scale_aware_thresh,
                small_box_area_thresh=args.small_box_area_thresh,
            )

            # 3) Run tracking on this sequence; write expression-specific result
            out_path = os.path.join(res_expr_dir, f"{expr_name}.txt")

            # worker.process_sequence expects gt_folder such that gt_folder/gt/<seq>.txt exists.
            worker_gt_root = os.path.join(run_outdir, "worker_gt")
            worker_gt_dir = os.path.join(worker_gt_root, "gt")
            os.makedirs(worker_gt_dir, exist_ok=True)
            dest_seq_gt = os.path.join(worker_gt_dir, f"{seq}.txt")
            if not os.path.isfile(dest_seq_gt):
                shutil.copy(seq_gt_path, dest_seq_gt)

            worker.process_sequence(
                seq=seq,
                img_folder=seq_img_root,
                gt_folder=worker_gt_root,
                out_path=out_path,
                video_out_path=None,
            )

    print(f"\nTotal expressions processed: {total_expr}")

    # ------------------------------------------------------------------
    # Evaluate per-expression with motmetrics
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("📊 Evaluating RMOT-style Results (per expression)")
    print(f"{'=' * 60}\n")

    evaluator = MotMetricsEvaluator(distth=0.5, fmt="mot15-2D")
    df = evaluator.evaluate(gt_expr_dir, res_expr_dir, verbose=True)

    if df is not None:
        if "AVG" in df.index:
            mota = float(df.loc["AVG"].get("mota", float("nan")))
            idf1 = float(df.loc["AVG"].get("idf1", float("nan")))
        else:
            mota = df["mota"].mean()
            idf1 = df["idf1"].mean()

        print(f"\n{'=' * 60}")
        print(f"📈 RMOT Summary over expressions: MOTA={mota*100:.2f}% | IDF1={idf1*100:.2f}%")
        print(f"{'=' * 60}\n")
        print(f"OPTUNA_RMOT:MOTA={mota:.6f} IDF1={idf1:.6f}")
    else:
        print("⚠ No matching GT/RES files for evaluation.")

    print(f"\n✅ Complete! Results saved to: {run_outdir}")


if __name__ == "__main__":
    main()

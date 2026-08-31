#!/usr/bin/env python3
"""Score the Week-2 relational sweep with SP / SR / DCR.

The sweep is the one dataset here built to test *relational* grounding: its
distractor matches the target on class, colour and rough size, so a prompt can
only be answered by reasoning about the relation. This runs the pipeline over
all 26 clips and scores them with the same metric implementation used for
Refer-KITTI and CARLA.

Read **SP, SR and DCR** from this. PCR and SID are not reported: every clip is a
single frame, so PCR collapses to 0/1 and SID has no time axis to switch along.

Two modes, and the comparison between them is the point:

    --plain      the pre-Week-3 path: detect the target class, emit everything
    --grounded   parse the query, detect target AND anchor, build the scene
                 graph over all candidates, score, then select answers

Thresholds default to the sweep's own (0.25 / 0.25 / 0.20) rather than the
eval_carla defaults (0.40 / 0.80 / 0.45): a clip is one frame, so a detection
that misses track_thresh never becomes a track at all, and the eval defaults
silently drop the distractor — which is the object the benchmark is about.

Usage::

    python eval/eval_sweep_prompt.py --grounded
    python eval/eval_sweep_prompt.py --plain
    python eval/eval_sweep_prompt.py --grounded --prompt "red car behind the bus"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sweep_prompt_gt as spg

CARLA_SIM_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "carla_sim")
)


def _import_prompt_metrics():
    if CARLA_SIM_DIR not in sys.path:
        sys.path.insert(0, CARLA_SIM_DIR)
    from evaluate_prompt_metrics import compute_metrics, load_predictions_mot
    return compute_metrics, load_predictions_mot


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dataset/sweep")
    ap.add_argument("--prompt", default="red car in front of the bus",
                    help="The prompt to evaluate. Default is the non-degenerate "
                         "direction — see sweep_prompt_gt for why the manifest's "
                         "own 'behind' prompt has two valid answers in most clips.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--grounded", action="store_true", default=True)
    mode.add_argument("--plain", dest="grounded", action="store_false")
    ap.add_argument("--no_answer_selection", dest="answer_selection",
                    action="store_false", default=True)

    ap.add_argument("--box_threshold", type=float, default=0.25)
    ap.add_argument("--text_threshold", type=float, default=0.25)
    ap.add_argument("--track_thresh", type=float, default=0.20)
    ap.add_argument("--match_thresh", type=float, default=0.85)
    ap.add_argument("--track_buffer", type=int, default=30)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--config", default="groundingdino/config/GroundingDINO_SwinB_cfg.py")
    ap.add_argument("--weights", default="weights/groundingdino_swinb_cogcoor.pth")
    ap.add_argument("--tracker", default="bytetrack",
                    choices=["bytetrack", "clip", "smartclip"])
    ap.add_argument("--device", default="0")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--skip_run", action="store_true")
    return ap


def run_clips(args, per_clip, results_dir):
    """Run the pipeline once per clip; return {clip_index: [prediction dicts]}."""
    import torch
    from worker_clean import Worker

    device = "cpu"
    if args.device != "cpu" and torch.cuda.is_available():
        device = f"cuda:{args.device}"

    query = None
    if args.grounded:
        from query_parser import parse
        query = parse(args.prompt)
        print(f"  parsed: target={query.target.get('class')} "
              f"anchor={(query.anchor or {}).get('class')} "
              f"relation={(query.relation or {}).get('name')}\n")

    # One Worker for the whole sweep: the model load is ~5.5 s and the prompt
    # does not change between clips, so building it 26 times is pure waste.
    worker = Worker(
        tracker_type=args.tracker,
        tracker_kwargs=dict(track_thresh=args.track_thresh,
                            track_buffer=args.track_buffer,
                            match_thresh=args.match_thresh),
        box_thresh=args.box_threshold,
        text_thresh=args.text_threshold,
        use_fp16=args.fp16,
        text_prompt=args.prompt,
        query=query,
        answer_selection=args.answer_selection,
        detector="dino",
        frame_rate=10,
        config_path=args.config,
        weights_path=args.weights,
        device=device,
        referring_mode="none",
        use_spatial_filter=False,
        use_color_filter=False,
    )

    for clip in per_clip:
        out_path = os.path.join(results_dir, f"{clip['name']}.txt")
        # img_folder/seq must resolve to the directory of frames.
        worker.process_sequence(
            seq="rgb",
            img_folder=clip["img_folder"],
            gt_folder="",
            out_path=out_path,
        )
        print(f"    {clip['name']:16} -> {out_path}")


def pool_predictions(per_clip, results_dir, load_predictions_mot):
    """Merge per-clip MOT files into one prediction dict keyed by clip index."""
    pooled = defaultdict(list)
    for clip in per_clip:
        path = os.path.join(results_dir, f"{clip['name']}.txt")
        if not os.path.isfile(path):
            continue
        for frame_preds in load_predictions_mot(path).values():
            pooled[clip["index"]].extend(frame_preds)
    return pooled


def main() -> None:
    args = build_argparser().parse_args()
    compute_metrics, load_predictions_mot = _import_prompt_metrics()

    from query_parser import parse
    relation = (parse(args.prompt).relation or {}).get("name")
    if relation is None:
        raise SystemExit(f"Prompt {args.prompt!r} carries no relation; "
                         "the sweep is a relational benchmark.")

    gt_data, per_clip = spg.build_pooled_gt(args.root, relation, args.prompt)

    if args.outdir is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        mode = "grounded" if args.grounded else "plain"
        args.outdir = os.path.join("outputs", f"sweep_prompt_{mode}_{stamp}")
    results_dir = os.path.join(args.outdir, "results")
    os.makedirs(results_dir, exist_ok=True)

    key = spg.answer_key_summary(per_clip)
    print(f"\n{'=' * 72}")
    print("Week-2 relational sweep · prompt-compliance (SP / SR / DCR)")
    print(f"{'=' * 72}")
    print(f"  prompt        \"{args.prompt}\"   (relation: {relation})")
    print(f"  mode          {'grounded' if args.grounded else 'plain'}"
          f"{'' if args.answer_selection or not args.grounded else ' (selection off)'}")
    print(f"  clips         {len(per_clip)}")
    print(f"  GT            {gt_data['meta']['num_valid']} valid / "
          f"{gt_data['meta']['num_annotations'] - gt_data['meta']['num_valid']} "
          f"non-valid boxes")
    print(f"  answer key    " + ", ".join(f"{v} clips {k}" for k, v in key.items() if v))
    print(f"  output        {args.outdir}")
    print(f"{'=' * 72}\n")

    if not args.skip_run:
        run_clips(args, per_clip, results_dir)

    predictions = pool_predictions(per_clip, results_dir, load_predictions_mot)
    sp, sr, pcr, dcr, stats = compute_metrics(gt_data, predictions, args.iou,
                                              mode="single_target")

    total = stats["total_predictions"]
    hit_valid = stats["predictions_matching_valid"]
    hit_distr = stats["predictions_matching_distractor"]
    hit_none = total - hit_valid - hit_distr

    print(f"\n{'=' * 72}")
    print(f"  SP   Semantic Precision       {sp * 100:6.2f}%   "
          f"{hit_valid}/{total} predictions on a valid target")
    print(f"  SR   Semantic Recall          {sr * 100:6.2f}%   "
          f"{stats['valid_gt_matched']}/{stats['total_valid_gt']} valid GT boxes found")
    print(f"  DCR  Distractor Confusion     {dcr * 100:6.2f}%   "
          f"{hit_distr}/{total} predictions on a distractor")
    print(f"\n  where the {total} predictions went: {hit_valid} valid · "
          f"{hit_distr} distractor · {hit_none} matched nothing")
    print(f"{'=' * 72}\n")

    # Per-clip: emitted vs the clip's own answer key. This is the reading that
    # matters on a relational benchmark — "returned nothing when nothing was
    # correct" is a success, and an aggregate rate hides it.
    by_clip = defaultdict(list)
    for a in gt_data["annotations"]:
        by_clip[a["image_id"]].append(a)

    exact = 0
    rows = []
    for clip in per_clip:
        preds = predictions.get(clip["index"], [])
        gts = by_clip[clip["index"]]
        matched_valid = set()
        for p in preds:
            best, best_iou = None, 0.0
            for g in gts:
                from evaluate_prompt_metrics import compute_iou
                i = compute_iou(p["bbox_xyxy"], g["bbox_xyxy"])
                if i > best_iou:
                    best, best_iou = g, i
            if best is not None and best_iou >= args.iou and best["is_target"]:
                matched_valid.add(best["gt_id"])
        want = clip["num_valid"]
        got_right = len(matched_valid) == want and len(preds) == want
        exact += got_right
        rows.append((clip["name"], clip["condition"], want, len(preds),
                     len(matched_valid), got_right))

    print(f"  Exact-answer clips: {exact}/{len(per_clip)}"
          f"   (emitted precisely the correct set, including the empty set)\n")
    print(f"  {'clip':16} {'cond':7} {'want':>5} {'emitted':>8} {'right':>6}  ok")
    print("  " + "-" * 52)
    for name, cond, want, npred, nright, ok in rows:
        print(f"  {name:16} {cond:7} {want:5d} {npred:8d} {nright:6d}  "
              f"{'yes' if ok else 'no'}")

    summary = {
        "prompt": args.prompt, "relation": relation,
        "mode": "grounded" if args.grounded else "plain",
        "answer_selection": args.answer_selection,
        "semantic_precision": sp, "semantic_recall": sr,
        "distractor_confusion_rate": dcr,
        "exact_answer_clips": exact, "clips": len(per_clip),
        "answer_key": key,
        "stats": {k: v for k, v in stats.items() if k != "frame_stats"},
        "per_clip": [dict(zip(("clip", "condition", "want", "emitted",
                               "right", "exact"), r)) for r in rows],
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  written: {os.path.join(args.outdir, 'summary.json')}\n")


if __name__ == "__main__":
    main()

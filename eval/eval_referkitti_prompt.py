#!/usr/bin/env python3
"""Refer-KITTI evaluation under the prompt-compliance metrics (SP / SR / DCR).

This is the SP/SR/PCR/DCR/SID counterpart to ``eval_referkitti.py``, which
reports MOTA/IDF1.  The pipeline it runs is the same; only the ground truth
and the scoring differ:

    eval_referkitti.py   GT = referred objects only   ->  MOTA, IDF1
    this script          GT = every object, flagged   ->  SP, SR, PCR, DCR, SID

Keeping the distractors in the GT is the whole point.  MOTA cannot tell a
prediction that landed on a parked car nobody asked about from one that landed
on empty road — both are false positives.  DCR can, and that is the error mode
a referring tracker actually has.

The metric implementations are imported from ``carla_sim/evaluate_prompt_metrics.py``
rather than reimplemented, so a Refer-KITTI number and a CARLA number are
produced by the same code.  ``referkitti_prompt_gt.py`` does the joining.

Solo test on one scenario (one sequence, one expression):

    python eval/eval_referkitti_prompt.py --sequence 0005 --expression cars-in-left --fp16

All expressions of a sequence:

    python eval/eval_referkitti_prompt.py --sequence 0005 --max_expressions 8 --fp16

Re-score results that are already on disk (no model load, seconds not minutes):

    python eval/eval_referkitti_prompt.py --sequence 0005 --expression cars-in-left \
        --outdir outputs/referkitti_prompt_2026-08-31_1200 --skip_run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

# eval/ is not a package; siblings are imported by name.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import referkitti_prompt_gt as rkgt
from referkitti_prompt_report import (
    render_expression_report,
    render_run_report,
    summarise,
)

# The prompt-compliance metrics live with the CARLA generator that defined
# them (carla_sim/metrics.md).  Same relative hop as eval_carla.py uses.
CARLA_SIM_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "carla_sim")
)


def _import_prompt_metrics():
    if CARLA_SIM_DIR not in sys.path:
        sys.path.insert(0, CARLA_SIM_DIR)
    try:
        from evaluate_prompt_metrics import (  # noqa: F401
            compute_metrics,
            compute_semantic_id_switches,
            load_predictions_mot,
        )
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise ImportError(
            f"Could not import evaluate_prompt_metrics from {CARLA_SIM_DIR}. "
            "The SP/SR/PCR/DCR/SID implementations live there; pass --carla_sim_dir "
            "if that checkout is somewhere else."
        ) from exc
    return compute_metrics, compute_semantic_id_switches, load_predictions_mot


# ----------------------------------------------------------------------
# Defaults — Trial 532 (see TRIAL_532_BEST_PARAMS.txt), the tuned Refer-KITTI
# configuration.  Kept here so a bare invocation reproduces the tuned method
# rather than an arbitrary one.
# ----------------------------------------------------------------------
DEFAULTS = dict(
    config="groundingdino/config/GroundingDINO_SwinB_cfg.py",
    weights="weights/swinb_light_visdrone_ft_best.pth",
    tracker="clip",
    detector="dino",
    box_threshold=0.455,
    text_threshold=0.363,
    track_thresh=0.159,
    match_thresh=0.880,
    track_buffer=110,
    lambda_weight=0.568,
    text_gate_mode="penalty",
    text_gate_weight=0.736,
    referring_mode="threshold",
    referring_thresh=0.263,
    small_box_area_thresh=4000,
    frame_rate=10,
    min_box_area=10,
)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )

    sel = ap.add_argument_group("scenario selection")
    sel.add_argument("--data_root", default="dataset/referkitti",
                     help="Refer-KITTI root (contains KITTI/ and expression/)")
    sel.add_argument("--split", default="training", choices=["training"])
    sel.add_argument("--sequence", default="0005",
                     help="KITTI sequence id, or a comma-separated list. "
                          "The standard Refer-KITTI test split is 0005,0011,0013 "
                          "— what the ByteTrack/FairMOT baselines in this repo "
                          "were scored on.")
    sel.add_argument("--test_split", action="store_true",
                     help="Shorthand for --sequence 0005,0011,0013")
    sel.add_argument("--expression", action="append", default=None,
                     help="Expression name (JSON stem, e.g. cars-in-left). "
                          "Repeatable. Default: the first --max_expressions.")
    sel.add_argument("--max_expressions", type=int, default=1,
                     help="How many expressions per sequence when --expression "
                          "is not given (default 1: the solo test)")
    sel.add_argument("--all_expressions", action="store_true",
                     help="Every expression of every selected sequence")
    sel.add_argument("--list", action="store_true",
                     help="List the sequence's expressions with their GT "
                          "valid/distractor counts, then exit")

    ev = ap.add_argument_group("evaluation")
    ev.add_argument("--iou", type=float, default=0.5,
                    help="IoU threshold for Hungarian matching (default 0.5)")
    ev.add_argument("--score_whole_clip", action="store_true",
                    help="Score every frame of the clip, not only the frames "
                         "the expression annotates (default: annotated only)")
    ev.add_argument("--carla_sim_dir", default=None,
                    help="Override the location of evaluate_prompt_metrics.py")

    run = ap.add_argument_group("run control")
    run.add_argument("--outdir", default=None, help="Run directory (default: auto)")
    run.add_argument("--skip_run", action="store_true",
                     help="Do not run the model; score the result files already "
                          "in <outdir>/results")
    run.add_argument("--resume", action="store_true",
                     help="Re-score expressions whose result file already exists "
                          "and run only the rest. Lets a long sweep be split "
                          "across GPUs or restarted without losing work.")
    run.add_argument("--device", default="0", help="GPU id, or 'cpu'")
    run.add_argument("--fp16", action="store_true")
    run.add_argument("--save_video", action="store_true")
    run.add_argument("--show_gt_boxes", action="store_true")

    mdl = ap.add_argument_group("method (defaults = tuned Trial 532)")
    mdl.add_argument("--config", default=DEFAULTS["config"])
    mdl.add_argument("--weights", default=DEFAULTS["weights"])
    mdl.add_argument("--tracker", default=DEFAULTS["tracker"],
                     choices=["bytetrack", "clip", "smartclip"])
    mdl.add_argument("--detector", default=DEFAULTS["detector"],
                     choices=["dino", "florence2"])
    mdl.add_argument("--box_threshold", type=float, default=DEFAULTS["box_threshold"])
    mdl.add_argument("--text_threshold", type=float, default=DEFAULTS["text_threshold"])
    mdl.add_argument("--track_thresh", type=float, default=DEFAULTS["track_thresh"])
    mdl.add_argument("--match_thresh", type=float, default=DEFAULTS["match_thresh"])
    mdl.add_argument("--track_buffer", type=int, default=DEFAULTS["track_buffer"])
    mdl.add_argument("--lambda_weight", type=float, default=DEFAULTS["lambda_weight"])
    mdl.add_argument("--text_gate_mode", default=DEFAULTS["text_gate_mode"],
                     choices=["penalty", "hard"])
    mdl.add_argument("--text_gate_weight", type=float, default=DEFAULTS["text_gate_weight"])
    # worker_clean only reads referring_mode as an on/off switch and applies
    # referring_thresh; there is no top-k path in it, so none is offered here.
    mdl.add_argument("--referring_mode", default=DEFAULTS["referring_mode"],
                     choices=["none", "threshold"])
    mdl.add_argument("--referring_thresh", type=float, default=DEFAULTS["referring_thresh"])
    mdl.add_argument("--small_box_area_thresh", type=int,
                     default=DEFAULTS["small_box_area_thresh"])
    mdl.add_argument("--min_box_area", type=int, default=DEFAULTS["min_box_area"])
    mdl.add_argument("--frame_rate", type=int, default=DEFAULTS["frame_rate"])
    mdl.add_argument("--no_color_filter", dest="use_color_filter",
                     action="store_false", default=True)
    mdl.add_argument("--no_spatial_filter", dest="use_spatial_filter",
                     action="store_false", default=True)
    mdl.add_argument("--no_scale_aware_thresh", dest="use_scale_aware_thresh",
                     action="store_false", default=True)
    # Off by default: the motion cue is weak on a moving camera (balanced
    # accuracy 0.589 on GT boxes). See eval/check_motion_classifier.py.
    mdl.add_argument("--use_motion_filter", action="store_true", default=False,
                     help="Enable the post-track motion gate for moving/parked "
                          "prompts (weak cue — measure before trusting)")
    mdl.add_argument("--tracker_kv", action="append",
                     help="Extra tracker args as key=val (repeatable)")
    return ap


# ----------------------------------------------------------------------
def selected_sequences(args) -> list:
    """Sequences to run, in order. ``--sequence`` accepts a comma-separated list."""
    if args.test_split:
        return ["0005", "0011", "0013"]
    seqs = [t.strip() for t in str(args.sequence).split(",") if t.strip()]
    known = set(rkgt.list_sequences(args.data_root, args.split))
    unknown = [s for s in seqs if s not in known]
    if unknown:
        raise SystemExit(f"Unknown sequence(s): {unknown}\n"
                         f"Available: {sorted(known)}")
    return seqs


def select_expressions(args, seq: str) -> list:
    """Expression names to run for one sequence."""
    names = rkgt.list_expressions(args.data_root, seq)
    if not names:
        raise SystemExit(f"No expressions for sequence {seq} "
                         f"under {args.data_root}/expression")

    if args.expression:
        # An explicit list is only meaningful against a single sequence; when
        # several are selected, keep the ones this sequence actually has rather
        # than failing the whole run on the first sequence that lacks one.
        chosen = [e for e in args.expression if e in names]
        if not chosen:
            raise SystemExit(
                f"None of {args.expression} exist for sequence {seq}.\n"
                f"Run with --sequence {seq} --list to see the {len(names)} available."
            )
        return chosen

    if args.all_expressions:
        return names
    return names[: args.max_expressions]


def run_tracking(args, seq: str, expression: dict, out_path: str) -> None:
    """Run the pipeline over one sequence with this expression as the prompt."""
    import torch
    from worker_clean import Worker, parse_kv_list

    device = "cpu"
    if args.device != "cpu" and torch.cuda.is_available():
        device = f"cuda:{args.device}"

    tracker_kwargs = dict(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        lambda_weight=args.lambda_weight,
        use_text_gate_matching=True,
        text_gate_mode=args.text_gate_mode,
        text_gate_weight=args.text_gate_weight,
    )
    tracker_kwargs.update(parse_kv_list(args.tracker_kv))

    # Every referred id in the clip — used only for GT overlay in --save_video.
    target_ids = sorted({
        tid
        for ids in expression["valid_ids_by_frame"].values()
        for tid in ids
    })

    worker = Worker(
        tracker_type=args.tracker,
        tracker_kwargs=tracker_kwargs,
        box_thresh=args.box_threshold,
        text_thresh=args.text_threshold,
        use_fp16=args.fp16,
        text_prompt=expression["sentence"],
        detector=args.detector,
        frame_rate=args.frame_rate,
        save_video=args.save_video,
        show_gt_boxes=args.show_gt_boxes,
        dataset_type="referkitti",
        referkitti_data_root=args.data_root,
        target_object_ids=target_ids,
        min_box_area=args.min_box_area,
        config_path=args.config,
        weights_path=args.weights,
        device=device,
        referring_mode=args.referring_mode,
        referring_thresh=args.referring_thresh,
        use_spatial_filter=args.use_spatial_filter,
        use_color_filter=args.use_color_filter,
        use_motion_filter=args.use_motion_filter,
        use_scale_aware_thresh=args.use_scale_aware_thresh,
        small_box_area_thresh=args.small_box_area_thresh,
    )

    worker.process_sequence(
        seq=seq,
        img_folder=os.path.join(args.data_root, "KITTI", args.split, "image_02"),
        gt_folder="",  # only read for dataset_type="mot"
        out_path=out_path,
        video_out_path=None,
    )


# ----------------------------------------------------------------------
def main() -> None:
    args = build_argparser().parse_args()

    if args.carla_sim_dir:
        global CARLA_SIM_DIR
        CARLA_SIM_DIR = os.path.abspath(args.carla_sim_dir)

    if args.list:
        for seq in selected_sequences(args):
            boxes = rkgt.load_sequence_boxes(args.data_root, seq, args.split)
            total = sum(len(v) for v in boxes.values())
            print(f"\nSequence {seq}: {len(boxes)} frames, {total} GT boxes\n")
            for name in rkgt.list_expressions(args.data_root, seq):
                gt = rkgt.load_prompt_gt(args.data_root, seq, name,
                                         args.split, boxes_by_frame=boxes)
                m = gt["meta"]
                print(f"  {name:52} {m['num_valid']:5d} valid / "
                      f"{m['num_distractors']:5d} distractor   \"{m['prompt']}\"")
        return

    compute_metrics, compute_sid, load_predictions_mot = _import_prompt_metrics()

    sequences = selected_sequences(args)
    plan = [(seq, select_expressions(args, seq)) for seq in sequences]
    total_jobs = sum(len(names) for _, names in plan)

    if args.outdir is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        args.outdir = os.path.join("outputs", f"referkitti_prompt_{stamp}")
    results_dir = os.path.join(args.outdir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'=' * 72}")
    print("Refer-KITTI · prompt-compliance evaluation (SP / SR / PCR / DCR / SID)")
    print(f"{'=' * 72}")
    for seq, names in plan:
        print(f"  {seq}         {len(names)} expression(s)")
    print(f"  total        {total_jobs} runs")
    print(f"  IoU match    {args.iou}")
    print(f"  scored on    "
          f"{'the whole clip' if args.score_whole_clip else 'annotated frames only'}")
    print(f"  output       {args.outdir}")
    print(f"{'=' * 72}\n")

    rows = []
    job = 0
    for seq, names in plan:
        # One sequence load, reused across that sequence's expressions.
        boxes_by_frame = rkgt.load_sequence_boxes(args.data_root, seq, args.split)

        for name in names:
            job += 1
            expression = rkgt.load_expression(args.data_root, seq, name)
            gt_data = rkgt.build_prompt_gt(
                boxes_by_frame, expression,
                restrict_to_labelled_frames=not args.score_whole_clip,
            )
            out_path = os.path.join(results_dir, f"{seq}_{name}.txt")

            print(f"[{job}/{total_jobs}] {seq} · {name}   \"{expression['sentence']}\"")

            if args.skip_run or (args.resume and os.path.isfile(out_path)):
                if not os.path.isfile(out_path):
                    print(f"    ⚠ no results at {out_path}; skipping\n")
                    continue
                print(f"    reusing {out_path}")
            else:
                try:
                    run_tracking(args, seq, expression, out_path)
                except Exception as exc:
                    # One bad expression must not cost a multi-hour sweep.
                    print(f"    ✗ FAILED: {type(exc).__name__}: {exc}\n")
                    continue

            predictions = load_predictions_mot(out_path)
            sp, sr, pcr, dcr, stats = compute_metrics(
                gt_data, predictions, args.iou, mode="single_target"
            )
            sid, sid_events = compute_sid(
                gt_data, predictions, args.iou, mode="single_target"
            )

            row = {
                "sequence": seq,
                "expression": name,
                "prompt": expression["sentence"],
                "semantic_precision": sp,
                "semantic_recall": sr,
                "prompt_coverage_ratio": pcr,
                "distractor_confusion_rate": dcr,
                "semantic_id_switches": sid,
                "gt": gt_data["meta"],
                "stats": {k: v for k, v in stats.items() if k != "frame_stats"},
                "result_file": os.path.relpath(out_path, args.outdir),
            }
            rows.append(row)

            print(render_expression_report(row, indent="    "))

            with open(os.path.join(args.outdir,
                                   f"metrics_{seq}_{name}.json"), "w") as f:
                json.dump({**row, "frame_stats": stats["frame_stats"],
                           "sid_events": sid_events}, f, indent=2)

            # Checkpoint after every expression: a sweep this long should not
            # lose everything to a crash in its last hour.
            with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
                json.dump({"config": {k: v for k, v in vars(args).items()},
                           "summary": summarise(rows), "expressions": rows},
                          f, indent=2)

    if not rows:
        print("No expressions were scored.")
        return

    summary = summarise(rows)
    report = render_run_report(rows, summary, args)
    print(report)

    with open(os.path.join(args.outdir, "report.md"), "w") as f:
        f.write(report)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"config": {k: v for k, v in vars(args).items()},
                   "summary": summary, "expressions": rows}, f, indent=2)

    print(f"\nWritten:")
    print(f"  {os.path.join(args.outdir, 'report.md')}")
    print(f"  {os.path.join(args.outdir, 'metrics.json')}")


if __name__ == "__main__":
    main()

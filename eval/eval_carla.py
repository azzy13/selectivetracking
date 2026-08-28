#!/usr/bin/env python3
"""
CARLA Scenario Evaluation for GroundingDINO + Tracker pipeline.

Evaluates tracking on CARLA-generated scenarios with ground truth.
For each scenario:
  - reads gt.json (with prompt, bbox annotations, is_target flags)
  - uses the prompt as text_prompt for GroundingDINO
  - runs detection + tracking via Worker
  - evaluates with prompt-compliance metrics (Prompt Precision, Prompt Recall, SID)

Usage:
    python eval/eval_carla.py \
        --carla_scenarios /path/to/carla_follow_scenarios \
        --tracker clip \
        --fp16
"""

import os
import sys
import json
import argparse
import shutil
from datetime import datetime
from threading import Thread


class TeeStream:
    """Write to both a file and the original stream (stdout/stderr)."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

import numpy as np
import torch

from worker_clean import Worker as WorkerClean, parse_kv_list
from worker_simple import Worker as WorkerSimple

# Prompt-compliance evaluator (lazy import in function)
CARLA_SIM_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "carla_sim")

# ----------------------------------------------------------------------
# Defaults (matching eval_referkitti.py)
# ----------------------------------------------------------------------
DEFAULT_CONFIG = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
#DEFAULT_WEIGHTS = "weights/swinb_light_visdrone_ft_best.pth"
DEFAULT_WEIGHTS = "weights/groundingdino_swinb_cogcoor.pth"
DEFAULT_FRAME_RATE = 10
DEFAULT_MIN_BOX_AREA = 10
    
# ----------------------------------------------------------------------
# CARLA scenario loading
# ----------------------------------------------------------------------
def load_carla_scenarios(scenarios_dir):
    """
    Scan carla_follow_scenarios/ for valid scenario directories.

    Returns:
        dict mapping scenario_name -> {
            'gt_json_path': str,
            'gt_data': dict,
            'images_dir': str,
            'text_prompt': str,
        }
    """
    scenario_info = {}

    for entry in sorted(os.listdir(scenarios_dir)):
        scenario_path = os.path.join(scenarios_dir, entry)
        gt_json_path = os.path.join(scenario_path, "gt.json")
        images_dir = os.path.join(scenario_path, "images")

        if not os.path.isfile(gt_json_path) or not os.path.isdir(images_dir):
            continue

        with open(gt_json_path) as f:
            gt_data = json.load(f)

        prompt = gt_data.get("meta", {}).get("prompt", "object")
        num_images = len(gt_data.get("images", []))
        num_anns = len(gt_data.get("annotations", []))

        scenario_info[entry] = {
            "gt_json_path": gt_json_path,
            "gt_data": gt_data,
            "images_dir": images_dir,
            "text_prompt": prompt,
        }
        print(f"  {entry}: {num_images} frames, {num_anns} annotations, prompt='{prompt}'")

    return scenario_info


def setup_image_symlinks(scenario_info, temp_images_dir):
    """
    Create symlinks so Worker.process_sequence can find images.

    Worker expects: temp_images_dir/<scenario_name>/<frame>.png
    CARLA has:      scenario_XXX/images/<frame>.png

    Creates: temp_images_dir/<scenario_name> -> scenario_XXX/images/
    """
    os.makedirs(temp_images_dir, exist_ok=True)

    for scenario_name, info in sorted(scenario_info.items()):
        src = os.path.abspath(info["images_dir"])
        dst = os.path.abspath(os.path.join(temp_images_dir, scenario_name))

        if os.path.exists(dst):
            continue

        try:
            os.symlink(src, dst, target_is_directory=True)
        except OSError:
            # Fallback: copy if symlinks not supported
            shutil.copytree(src, dst)

    return temp_images_dir


# ----------------------------------------------------------------------
# Prompt-compliance evaluation
# ----------------------------------------------------------------------
def run_prompt_compliance_eval(scenario_info, res_folder, iou_threshold=0.5,
                               mode="single_target"):
    """
    Run prompt-compliance evaluation for all CARLA scenarios.

    Uses functions from carla_sim/evaluate_prompt_metrics.py.

    Returns:
        dict mapping scenario_name -> metrics dict
    """
    # Import prompt metrics evaluator
    carla_sim_path = os.path.abspath(CARLA_SIM_DIR)
    if carla_sim_path not in sys.path:
        sys.path.insert(0, carla_sim_path)

    from evaluate_prompt_metrics import (
        load_predictions_mot,
        compute_metrics,
        compute_semantic_id_switches,
    )

    results = {}

    for scenario_name, info in sorted(scenario_info.items()):
        gt_data = info["gt_data"]
        pred_path = os.path.join(res_folder, f"{scenario_name}.txt")

        if not os.path.isfile(pred_path):
            print(f"  Warning: No predictions for {scenario_name}, skipping.")
            continue

        predictions = load_predictions_mot(pred_path)

        sp, sr, pcr, dcr, pr_stats = compute_metrics(
            gt_data, predictions, iou_threshold, mode
        )
        sid_count, sid_events = compute_semantic_id_switches(
            gt_data, predictions, iou_threshold, mode
        )

        prompt = gt_data["meta"]["prompt"]
        results[scenario_name] = {
            "semantic_precision": sp,
            "semantic_recall": sr,
            "prompt_coverage_ratio": pcr,
            "distractor_confusion_rate": dcr,
            "semantic_id_switches": sid_count,
            "prompt": prompt,
            "pr_stats": pr_stats,
            "sid_events": sid_events,
        }

        print(f"\n  === {scenario_name} (prompt: '{prompt}') ===")
        print(f"    Semantic Precision:        {sp:.4f}"
              f"  ({pr_stats['predictions_matching_valid']}/{pr_stats['total_predictions']})")
        print(f"    Semantic Recall:           {sr:.4f}"
              f"  ({pr_stats['valid_gt_matched']}/{pr_stats['total_valid_gt']})")
        print(f"    Prompt Coverage Ratio:     {pcr:.4f}")
        print(f"    Distractor Confusion Rate: {dcr:.4f}")
        print(f"    Semantic ID Switches:      {sid_count}")

    return results


def save_prompt_results(results, out_folder):
    """Save per-scenario prompt-compliance results as JSON."""
    os.makedirs(out_folder, exist_ok=True)

    for scenario_name, metrics in results.items():
        out_path = os.path.join(out_folder, f"metrics_prompt_{scenario_name}.json")
        save_data = {
            "scenario": scenario_name,
            "prompt": metrics["prompt"],
            "semantic_precision": metrics["semantic_precision"],
            "semantic_recall": metrics["semantic_recall"],
            "prompt_coverage_ratio": metrics["prompt_coverage_ratio"],
            "distractor_confusion_rate": metrics["distractor_confusion_rate"],
            "semantic_id_switches": metrics["semantic_id_switches"],
        }
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2)


# ----------------------------------------------------------------------
# Query grounding (Week 3) — text prompt -> parsed Query
# ----------------------------------------------------------------------
def resolve_prompt(args, info):
    """The prompt this scenario will actually be run with.

    ``--text_prompt`` overrides the per-scenario prompt from gt.json.  Returned
    without a trailing "." — the detector caption convention is applied later,
    and the parser reads the sentence, not the caption.
    """
    prompt = args.text_prompt if args.text_prompt else info["text_prompt"]
    return prompt.strip().rstrip(".").strip()


def build_query(prompt):
    """Parse ``prompt`` into a ``query_parser.Query``.

    Raises whatever the parser raises — callers validate up front (see
    ``validate_queries``) so a run never silently degrades to plain MOT after
    being asked for grounding.
    """
    from query_parser import parse
    return parse(prompt)


def validate_queries(scenario_info, args):
    """Parse every scenario's prompt before any model is loaded.

    Grounding that fails halfway through a multi-GPU run wastes the whole run,
    and falling back to plain MOT would produce output that looks grounded but
    is not.  So every prompt is parsed here first and the run aborts if any of
    them cannot be.

    Returns:
        ``{scenario_name: Query}``.
    """
    queries, failures = {}, []
    for scenario_name, info in sorted(scenario_info.items()):
        prompt = resolve_prompt(args, info)
        try:
            queries[scenario_name] = build_query(prompt)
        except Exception as exc:
            failures.append((scenario_name, prompt, type(exc).__name__, str(exc)))

    if failures:
        print("\nERROR: --grounded was requested but these prompts do not parse:")
        for scenario_name, prompt, kind, msg in failures:
            print(f"  {scenario_name}: '{prompt}'\n      {kind}: {msg}")
        print("\nTeach the parser the phrasing (PHRASE_RELATIONS / VERB_RELATIONS /")
        print("MODIFIER_RELATIONS in eval/query_parser.py) or drop --grounded.")
        sys.exit(1)

    from query_grounding import describe_grounding
    print("\nQuery grounding enabled:")
    for scenario_name, query in queries.items():
        print(f"  {scenario_name}: {describe_grounding(query)}")
        for note in query.notes:
            print(f"      note: {note}")
    return queries


# ----------------------------------------------------------------------
# Multi-GPU worker
# ----------------------------------------------------------------------
def _run_on_device(device_str, scenario_items, args, tracker_kwargs,
                   temp_images_dir, res_dir, run_outdir, queries=None):
    """Process a batch of scenarios on a single GPU device.

    ``queries`` maps scenario name -> ``query_parser.Query`` when --grounded is
    on.  A scenario with a query gets the grounded pipeline (caption carries the
    anchor class, graph over all candidates, soft scorer); one without gets the
    pre-Week-3 hard-filter pipeline, unchanged.
    """
    gpu_tag = device_str.split(":")[-1] if "cuda" in device_str else "cpu"

    dummy_gt_root = os.path.join(run_outdir, "_gt_stub")
    os.makedirs(os.path.join(dummy_gt_root, "gt"), exist_ok=True)

    for scenario_name, info in scenario_items:
        text_prompt = resolve_prompt(args, info)
        query = (queries or {}).get(scenario_name)
        if not text_prompt.endswith("."):
            text_prompt += "."

        print(f"\n[GPU {gpu_tag}] {'=' * 50}")
        print(f"[GPU {gpu_tag}] Scenario: {scenario_name}")
        print(f"[GPU {gpu_tag}] Prompt:   '{text_prompt}'")
        # The Worker rebuilds the caption from the query, so print what the
        # detector will actually see rather than the sentence it came from.
        if query is not None:
            from query_grounding import build_detector_prompt
            print(f"[GPU {gpu_tag}] Caption:  '{build_detector_prompt(query, dotted=True)}'")
        print(f"[GPU {gpu_tag}] Images:   {info['images_dir']}")
        print(f"[GPU {gpu_tag}] {'=' * 50}")

        if args.worker == "simple":
            # WorkerSimple has no query-grounding path; validate_queries only
            # runs for --worker clean, so this is belt-and-braces.
            if query is not None:
                raise ValueError("--grounded requires --worker clean")
            worker = WorkerSimple(
                config_path=args.config,
                weights_path=args.weights,
                text_prompt=text_prompt,
                box_thresh=args.box_threshold,
                text_thresh=args.text_threshold,
                use_fp16=args.fp16,
                device=device_str,
                tracker_kwargs=dict(tracker_kwargs),
                frame_rate=args.frame_rate,
                use_scale_aware_thresh=args.use_scale_aware_thresh,
                small_box_area_thresh=args.small_box_area_thresh,
                use_mission_filter=True,
                min_box_area=args.min_box_area,
                save_video=args.save_video,
            )
        else:
            worker = WorkerClean(
                tracker_type=args.tracker,
                tracker_kwargs=dict(tracker_kwargs),
                box_thresh=args.box_threshold,
                text_thresh=args.text_threshold,
                use_fp16=args.fp16,
                text_prompt=text_prompt,
                detector=args.detector,
                frame_rate=args.frame_rate,
                save_video=args.save_video,
                show_gt_boxes=args.show_gt_boxes,
                dataset_type="mot",
                min_box_area=args.min_box_area,
                config_path=args.config,
                weights_path=args.weights,
                device=device_str,
                referring_mode="threshold",
                referring_thresh=args.referring_thresh,
                use_spatial_filter=args.use_spatial_filter,
                use_color_filter=args.use_color_filter,
                use_scale_aware_thresh=args.use_scale_aware_thresh,
                small_box_area_thresh=args.small_box_area_thresh,
                query=query,
                debug_draw_anchors=args.debug_draw_anchors,
                answer_selection=args.answer_selection,
            )

        out_path = os.path.join(res_dir, f"{scenario_name}.txt")
        worker.process_sequence(
            seq=scenario_name,
            img_folder=temp_images_dir,
            gt_folder=dummy_gt_root,
            out_path=out_path,
            enable_scene_graph=args.scene_graph,
        )

        if args.visualize_scene_graph:
            jsonl_path = out_path.replace(".txt", "_scene_graphs.jsonl")
            if os.path.isfile(jsonl_path):
                viz_dir = os.path.join(run_outdir, "viz", scenario_name)
                import subprocess
                script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "visualize_scene_graph.py")
                subprocess.run([
                    sys.executable, script,
                    "--jsonl", jsonl_path,
                    "--images", info["images_dir"],
                    "--out", viz_dir,
                ], check=False)
                print(f"[GPU {gpu_tag}] Scene graph viz → {viz_dir}/")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="CARLA scenario evaluation using GroundingDINO + Tracker"
    )

    # Worker selection
    ap.add_argument(
        "--worker", choices=["clean", "simple"], default="clean",
        help="'clean' = full worker_clean pipeline (CLIP+color+spatial filters); "
             "'simple' = ByteTrack + SceneGraphMissionFilter only (no CLIP).",
    )

    # CARLA-specific
    ap.add_argument(
        "--carla_scenarios", required=True,
        help="Path to carla_follow_scenarios/ directory.",
    )
    ap.add_argument(
        "--scenarios", nargs="+", default=None, metavar="SCENARIO",
        help="Run only these scenario names, e.g. --scenarios scenario_001 scenario_003",
    )
    ap.add_argument(
        "--text_prompt", "--prompt", type=str, default=None, dest="text_prompt",
        help="Override text prompt for all scenarios (default: read from gt.json).",
    )
    ap.add_argument(
        "--prompt_eval_mode", type=str, default="single_target",
        choices=["single_target", "all_red_sedans"],
        help="Mode for prompt-compliance evaluation.",
    )

    # Query grounding (Week 3) — see eval/query_grounding.py
    ap.add_argument(
        "--grounded", action=argparse.BooleanOptionalAction, default=False,
        help="Parse each prompt into a Query and run the grounded pipeline: the "
             "detector caption carries the anchor class too, the scene graph is "
             "built over all candidates, and the soft subgraph scorer replaces "
             "the referring/colour hard filters.  Requires --worker clean.",
    )
    ap.add_argument(
        "--answer_selection", action=argparse.BooleanOptionalAction, default=True,
        help="With --grounded, emit only the candidates the parsed relation "
             "actually holds for, instead of every target candidate. A "
             "threshold, not an argmax: a plural prompt can return several, and "
             "a frame where nothing satisfies the relation returns nothing. "
             "Prompts the relation layer cannot judge (plain, ego-anchored, or "
             "no scene-graph equivalent) are passed through untouched. "
             "--no-answer_selection restores the pre-selection behaviour.")
    ap.add_argument(
        "--debug_draw_anchors", action="store_true",
        help="With --grounded and --save_video, draw anchor boxes dotted orange. "
             "Anchors are still never written to the MOT output.",
    )

    # Detector (defaults from eval_referkitti.py)
    ap.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    ap.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    ap.add_argument("--detector", choices=["dino", "florence2"], default="dino")
    ap.add_argument("--box_threshold", type=float, default=0.40)
    ap.add_argument("--text_threshold", type=float, default=0.80)
    ap.add_argument("--use_scale_aware_thresh", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--small_box_area_thresh", type=int, default=5000)
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)

    # Tracker (defaults from eval_referkitti.py)
    ap.add_argument(
        "--tracker", choices=["bytetrack", "clip", "smartclip"], default="bytetrack",
    )
    ap.add_argument("--track_thresh", type=float, default=0.45)
    ap.add_argument("--match_thresh", type=float, default=0.80)
    ap.add_argument("--track_buffer", type=int, default=120)
    ap.add_argument("--lambda_weight", type=float, default=0.25)
    ap.add_argument("--low_thresh", type=float, default=0.1)
    ap.add_argument("--text_sim_thresh", type=float, default=0.0)
    ap.add_argument("--use_clip_in_high", action="store_true")
    ap.add_argument("--use_clip_in_low", action="store_true")
    ap.add_argument("--use_clip_in_unconf", action="store_true")

    # Text-grounding at matching
    ap.add_argument(
        "--use_text_gate_matching", type=lambda x: x.lower() == "true",
        default=True,
    )
    ap.add_argument(
        "--text_gate_mode", type=str, choices=["penalty", "hard"], default="penalty",
    )
    ap.add_argument("--text_gate_weight", type=float, default=0.5)

    # Referring detection filter
    ap.add_argument("--referring_thresh", type=float, default=0.25)

    ap.add_argument("--use_color_filter", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use_spatial_filter", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument(
        "--tracker_kv", action="append",
        help="Extra tracker args as key=val (repeatable).",
    )

    # Output / misc
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--min_box_area", type=int, default=DEFAULT_MIN_BOX_AREA)
    ap.add_argument("--frame_rate", type=int, default=DEFAULT_FRAME_RATE)
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--show_gt_boxes", action="store_true")
    ap.add_argument("--devices", type=str, default="0,1")

    # Scene graph
    ap.add_argument(
        "--scene_graph", action="store_true", default=True,
        help="Build scene graphs and save JSONL to results/ alongside MOT output.",
    )
    ap.add_argument(
        "--visualize_scene_graph", action="store_true", default=True,
        help="Render per-frame scene graph PNGs to viz/<scenario>/ (implies --scene_graph).",
    )

    args = ap.parse_args()

    # Visualizing scene graphs requires building them first
    if args.visualize_scene_graph:
        args.scene_graph = True

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    if args.outdir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        run_outdir = os.path.join("outputs", f"carla_eval_{timestamp}")
    else:
        run_outdir = args.outdir

    os.makedirs(run_outdir, exist_ok=True)
    res_dir = os.path.join(run_outdir, "results")
    temp_images_dir = os.path.join(run_outdir, "images")

    # Tee stdout/stderr to a log file in the output directory
    log_path = os.path.join(run_outdir, "run_log.txt")
    _log_fh = open(log_path, "w")
    sys.stdout = TeeStream(sys.__stdout__, _log_fh)
    sys.stderr = TeeStream(sys.__stderr__, _log_fh)

    print(f"\n{'=' * 60}")
    print(f"CARLA Scenario Evaluation")
    print(f"Scenarios: {args.carla_scenarios}")
    print(f"Tracker:   {args.tracker}")
    print(f"Color filter: {args.use_color_filter}")
    print(f"Spatial filter: {args.use_spatial_filter}")
    print(f"Grounded:  {args.grounded}")
    print(f"Output:    {run_outdir}")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Step 1: Load scenarios
    # ------------------------------------------------------------------
    print("Loading CARLA scenarios...")
    scenario_info = load_carla_scenarios(args.carla_scenarios)

    if args.scenarios:
        unknown = set(args.scenarios) - set(scenario_info)
        if unknown:
            print(f"ERROR: Unknown scenarios: {sorted(unknown)}")
            sys.exit(1)
        scenario_info = {k: v for k, v in scenario_info.items() if k in args.scenarios}

    if not scenario_info:
        print("ERROR: No valid CARLA scenarios found.")
        sys.exit(1)

    print(f"\nFound {len(scenario_info)} scenarios.\n")

    # Query grounding: parse every prompt before loading a model, so a bad
    # prompt fails in seconds rather than after the detector is on the GPU.
    queries = None
    if args.grounded:
        if args.worker != "clean":
            print("ERROR: --grounded requires --worker clean "
                  f"(got --worker {args.worker}).")
            sys.exit(1)
        queries = validate_queries(scenario_info, args)

    print("Setting up image paths...")
    setup_image_symlinks(scenario_info, temp_images_dir)

    # ------------------------------------------------------------------
    # Step 2: Build tracker kwargs
    # ------------------------------------------------------------------
    tracker_kwargs = dict(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        lambda_weight=args.lambda_weight,
        low_thresh=args.low_thresh,
        text_sim_thresh=args.text_sim_thresh,
        use_clip_in_high=args.use_clip_in_high,
        use_clip_in_low=args.use_clip_in_low,
        use_clip_in_unconf=args.use_clip_in_unconf,
        use_text_gate_matching=args.use_text_gate_matching,
        text_gate_mode=args.text_gate_mode,
        text_gate_weight=args.text_gate_weight,
    )
    tracker_kwargs.update(parse_kv_list(args.tracker_kv))

    # ------------------------------------------------------------------
    # Step 3: Run inference per scenario (parallel across GPUs)
    # ------------------------------------------------------------------
    os.makedirs(res_dir, exist_ok=True)

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if not devices:
        devices = ["0"]
    scenario_list = sorted(scenario_info.items())

    if len(devices) > 1 and len(scenario_list) > 1:
        # Distribute scenarios across GPUs round-robin
        chunks = [[] for _ in devices]
        for i, item in enumerate(scenario_list):
            chunks[i % len(devices)].append(item)

        print(f"\nRunning on {len(devices)} GPUs in parallel...")
        for gpu_id, chunk in zip(devices, chunks):
            names = [s[0] for s in chunk]
            print(f"  GPU {gpu_id}: {len(chunk)} scenarios - {names}")

        threads = []
        for gpu_id, chunk in zip(devices, chunks):
            if not chunk:
                continue
            device_str = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
            t = Thread(
                target=_run_on_device,
                args=(device_str, chunk, args, tracker_kwargs,
                      temp_images_dir, res_dir, run_outdir, queries),
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
    else:
        device_str = f"cuda:{devices[0]}" if torch.cuda.is_available() else "cpu"
        _run_on_device(device_str, scenario_list, args, tracker_kwargs,
                       temp_images_dir, res_dir, run_outdir, queries)

    # ------------------------------------------------------------------
    # Step 4: Prompt-compliance evaluation
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Prompt-Compliance Metrics")
    print(f"{'=' * 60}")

    prompt_results = run_prompt_compliance_eval(
        scenario_info, res_dir,
        iou_threshold=0.5,
        mode=args.prompt_eval_mode,
    )

    if prompt_results:
        save_prompt_results(prompt_results, run_outdir)

    # ------------------------------------------------------------------
    # Step 5: Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("SUMMARY ACROSS ALL SCENARIOS")
    print(f"{'=' * 60}")

    if prompt_results:
        avg_sp = np.mean([r["semantic_precision"] for r in prompt_results.values()])
        avg_sr = np.mean([r["semantic_recall"] for r in prompt_results.values()])
        avg_pcr = np.mean([r["prompt_coverage_ratio"] for r in prompt_results.values()])
        avg_dcr = np.mean([r["distractor_confusion_rate"] for r in prompt_results.values()])
        total_sid = sum(r["semantic_id_switches"] for r in prompt_results.values())
        print(f"  Avg Semantic Precision:        {avg_sp:.4f}")
        print(f"  Avg Semantic Recall:           {avg_sr:.4f}")
        print(f"  Avg Prompt Coverage Ratio:     {avg_pcr:.4f}")
        print(f"  Avg Distractor Confusion Rate: {avg_dcr:.4f}")
        print(f"  Total Semantic ID Switches:    {total_sid}")

    print(f"\nResults saved to: {run_outdir}")
    print(f"Log saved to: {log_path}")
    print(f"{'=' * 60}")

    # Close log file and restore streams
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    _log_fh.close()


if __name__ == "__main__":
    main()

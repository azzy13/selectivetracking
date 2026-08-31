#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for ReferKITTI RMOT evaluation.

This script optimizes tracking parameters for the CLIP tracker on ReferKITTI
using Optuna. It evaluates a combined metric of MOTA and IDF1 to find the
best parameter configuration.

Configuration:
    - Tracker: clip
    - Metric: 60% MOTA + 40% IDF1
    - Referring mode: threshold (always)
    - Debug Mode: Uses hardcoded restrictions (seq='0001', max_expr=2) for fast trials

Usage:
    conda activate dino_real
    python eval/tune_referkitti_optuna.py

Expected Runtime:
    ~3-5 hours for 100 trials with debug mode enabled
    ~50 hours (2 days) for 1000 trials with debug mode enabled

Outputs:
    - optuna_referkitti.csv: All trial results with parameters and metrics
    - Console output: Progress updates and top 10 results
"""

import optuna
import subprocess
import os
import re
import shutil
from datetime import datetime
import csv

# ============================================================================
# Configuration
# ============================================================================

EVAL_SCRIPT = "eval/eval_referkitti.py"
DATA_ROOT = "/isis/home/hasana3/vlmtest/GroundingDINO/dataset/referkitti"
WEIGHTS = "weights/swinb_light_visdrone_ft_best.pth"
CONFIG = "groundingdino/config/GroundingDINO_SwinB_cfg.py"

# Metric weighting: 60% MOTA + 40% IDF1
MOTA_WEIGHT = 0.6
IDF1_WEIGHT = 0.4

# ============================================================================
# Helper Functions
# ============================================================================

def run_eval(box_thresh, text_thresh, track_thresh, match_thresh, track_buffer,
             lambda_weight, use_clip_in_high, use_clip_in_low,
             text_gate_mode, text_gate_weight,
             referring_mode, referring_thresh,
             small_box_area_thresh, use_color_filter, use_spatial_filter,
             *, devices="0,1", jobs=2, fp16=True, show_output=False):
    """
    Run eval_referkitti.py with given parameters and parse results.

    Args:
        box_thresh: GroundingDINO box confidence threshold
        text_thresh: GroundingDINO text threshold
        track_thresh: Tracker high-confidence threshold
        match_thresh: Tracker matching threshold
        track_buffer: Tracker buffer length
        lambda_weight: CLIP fusion weight
        use_clip_in_high: Use CLIP in high-confidence stage
        use_clip_in_low: Use CLIP in low-confidence stage
        text_gate_mode: Text-grounding mode ("penalty" or "hard")
        text_gate_weight: Text-grounding weight
        referring_mode: Referring filter mode ("threshold", "none")
        referring_thresh: Threshold for referring filter
        small_box_area_thresh: Scale-aware area threshold
        use_color_filter: Enable color filtering
        use_spatial_filter: Enable spatial filtering
        devices: GPU devices to use
        jobs: Max concurrent jobs
        fp16: Use FP16
        show_output: Print subprocess output

    Returns:
        tuple: (combined_score, mota, idf1) or (None, None, None) on failure
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    outdir = f"tmp_referkitti_optuna_{timestamp}"
    os.makedirs(outdir, exist_ok=True)

    # Build command (use conda environment dino_real)
    cmd = [
        "conda", "run", "-n", "dino_real", "python", EVAL_SCRIPT,
        "--data_root", DATA_ROOT,
        "--weights", WEIGHTS,
        "--config", CONFIG,
        "--tracker", "clip",
        "--detector", "dino",
        "--devices", devices,
        "--jobs", str(jobs),
        "--outdir", outdir,

        # Detection params
        "--box_threshold", str(box_thresh),
        "--text_threshold", str(text_thresh),

        # Tracker params
        "--track_thresh", str(track_thresh),
        "--match_thresh", str(match_thresh),
        "--track_buffer", str(track_buffer),

        # CLIP fusion params (clip tracker specific)
        "--lambda_weight", str(lambda_weight),
    ]

    # Add boolean CLIP fusion flags
    if use_clip_in_high:
        cmd.append("--use_clip_in_high")
    if use_clip_in_low:
        cmd.append("--use_clip_in_low")

    # Text-gate matching
    cmd.extend(["--text_gate_mode", text_gate_mode])
    cmd.extend(["--text_gate_weight", str(text_gate_weight)])

    # Referring filter
    cmd.extend(["--referring_mode", referring_mode])
    if referring_mode == "threshold":
        cmd.extend(["--referring_thresh", str(referring_thresh)])

    # Scale-aware threshold
    cmd.extend(["--small_box_area_thresh", str(small_box_area_thresh)])

    # Color/spatial filters
    if not use_color_filter:
        cmd.append("--no_color_filter")
    if not use_spatial_filter:
        cmd.append("--no_spatial_filter")

    if fp16:
        cmd.append("--fp16")

    # Run subprocess
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr

    # Parse output: "OPTUNA_RMOT:MOTA=0.123456 IDF1=0.654321"
    m = re.search(r"OPTUNA_RMOT:MOTA=([\-0-9.]+)\s+IDF1=([\-0-9.]+)", output)

    if m:
        mota = float(m.group(1))
        idf1 = float(m.group(2))
        combined = MOTA_WEIGHT * mota + IDF1_WEIGHT * idf1

        print(f"[Trial] Combined={combined:.4f} (MOTA={mota:.4f}, IDF1={idf1:.4f})")
        print(f"   Params: box={box_thresh:.3f}, text={text_thresh:.3f}, "
              f"track={track_thresh:.3f}, match={match_thresh:.3f}, "
              f"ref_thresh={referring_thresh:.3f}, clip_high={use_clip_in_high}, "
              f"clip_low={use_clip_in_low}")
    else:
        print(f"[Trial] ⚠️  Failed to extract metrics")
        if show_output:
            print(output)
        combined, mota, idf1 = None, None, None

    # Cleanup temp directory
    try:
        shutil.rmtree(outdir)
    except Exception as e:
        print(f"Warning: Failed to delete {outdir}: {e}")

    return combined, mota, idf1


def objective(trial):
    """
    Optuna objective function for ReferKITTI optimization.

    Optimizes 15 parameters across detection, tracking, CLIP fusion,
    text-gating, referring filters, and spatial/color awareness.

    Returns:
        float: Value to minimize (1.0 - combined_score)
    """
    # Detection parameters - FULL RANGE EXPLORATION
    box_threshold = trial.suggest_float("box_threshold", 0.1, 0.9)
    text_threshold = trial.suggest_float("text_threshold", 0.1, 0.9)

    # Tracker parameters - FULL RANGE EXPLORATION
    track_thresh = trial.suggest_float("track_thresh", 0.1, 0.9)
    match_thresh = trial.suggest_float("match_thresh", 0.1, 0.99)
    track_buffer = trial.suggest_int("track_buffer", 10, 300, step=10)

    # CLIP fusion parameters - FULL RANGE EXPLORATION
    lambda_weight = trial.suggest_float("lambda_weight", 0.0, 1.0)
    use_clip_in_high = trial.suggest_categorical("use_clip_in_high", [True, False])
    use_clip_in_low = trial.suggest_categorical("use_clip_in_low", [True, False])

    # Text-gate matching - FULL RANGE EXPLORATION
    text_gate_mode = trial.suggest_categorical("text_gate_mode", ["penalty", "hard"])
    text_gate_weight = trial.suggest_float("text_gate_weight", 0.0, 1.0) if text_gate_mode == "penalty" else 0.5

    # Referring filter (always use threshold mode) - FULL RANGE EXPLORATION
    referring_mode = "threshold"
    referring_thresh = trial.suggest_float("referring_thresh", 0.0, 0.9)

    # Scale-aware detection - FULL RANGE EXPLORATION
    small_box_area_thresh = trial.suggest_int("small_box_area_thresh", 1000, 15000, step=500)

    # Spatial/color filters
    use_color_filter = trial.suggest_categorical("use_color_filter", [True, False])
    use_spatial_filter = trial.suggest_categorical("use_spatial_filter", [True, False])

    # Run evaluation
    combined, mota, idf1 = run_eval(
        box_threshold, text_threshold, track_thresh, match_thresh, track_buffer,
        lambda_weight, use_clip_in_high, use_clip_in_low,
        text_gate_mode, text_gate_weight,
        referring_mode, referring_thresh,
        small_box_area_thresh, use_color_filter, use_spatial_filter,
        devices="0,1", jobs=2, fp16=True, show_output=False
    )

    # Store metrics as trial user attributes for later retrieval
    trial.set_user_attr("mota", mota if mota is not None else -1.0)
    trial.set_user_attr("idf1", idf1 if idf1 is not None else -1.0)

    if combined is None:
        return 1.0  # Penalize failed trials

    return 1.0 - combined  # Optuna minimizes, we want to maximize


def save_results(study, output_file="optuna_referkitti.csv"):
    """
    Save all trial results to CSV file.

    Args:
        study: Optuna study object
        output_file: Output CSV filename

    Returns:
        list: All trial data as rows for further processing
    """
    param_names = [
        "box_threshold", "text_threshold", "track_thresh", "match_thresh",
        "track_buffer", "lambda_weight", "use_clip_in_high", "use_clip_in_low",
        "text_gate_mode", "text_gate_weight", "referring_thresh",
        "small_box_area_thresh", "use_color_filter", "use_spatial_filter"
    ]

    all_trials = []
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "combined_score", "MOTA", "IDF1"] + param_names)

        for trial in study.trials:
            combined = round(1.0 - trial.value, 6) if trial.value is not None else -1.0
            mota = round(trial.user_attrs.get("mota", -1.0), 6)
            idf1 = round(trial.user_attrs.get("idf1", -1.0), 6)

            row = [trial.number, combined, mota, idf1]
            for p in param_names:
                val = trial.params.get(p)
                if isinstance(val, float):
                    val = round(val, 6)
                row.append(val)

            writer.writerow(row)
            all_trials.append(row)

    print(f"\nSaved all trial results to {output_file}")
    return all_trials


def print_top_results(all_trials, top_n=10):
    """
    Print top N trials by combined score.

    Args:
        all_trials: List of trial data rows
        top_n: Number of top trials to display
    """
    all_trials.sort(key=lambda x: -x[1])  # Sort by combined_score descending

    print(f"\nTop {top_n} trials by combined score ({MOTA_WEIGHT}*MOTA + {IDF1_WEIGHT}*IDF1):")
    print(f"{'Trial':>5} | {'Combined':>8} | {'MOTA':>6} | {'IDF1':>6} | "
          f"{'box_th':>7} | {'text_th':>7} | {'ref_th':>7} | "
          f"{'clip_h':>6} | {'clip_l':>6} | {'color':>5} | {'spatial':>7}")
    print("-" * 105)

    for row in all_trials[:top_n]:
        trial_num = row[0]
        combined = row[1]
        mota = row[2]
        idf1 = row[3]
        box_th = row[4]
        text_th = row[5]
        # Parameters: box, text, track, match, buffer, lambda, clip_h, clip_l, gate_mode, gate_weight, ref_thresh, area, color, spatial
        # Indices: 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
        clip_h = row[10] if len(row) > 10 else "N/A"
        clip_l = row[11] if len(row) > 11 else "N/A"
        ref_thresh = row[14] if len(row) > 14 else "N/A"
        color = row[16] if len(row) > 16 else "N/A"
        spatial = row[17] if len(row) > 17 else "N/A"

        print(f"{trial_num:>5} | {combined:>8.4f} | {mota:>6.4f} | {idf1:>6.4f} | "
              f"{box_th:>7.3f} | {text_th:>7.3f} | {ref_thresh:>7.3f} | "
              f"{str(clip_h):>6} | {str(clip_l):>6} | {str(color):>5} | {str(spatial):>7}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ReferKITTI Optuna Parameter Optimization - FULL RANGE EXPLORATION")
    print("=" * 80)
    print(f"Tracker: clip")
    print(f"Metric: {MOTA_WEIGHT}*MOTA + {IDF1_WEIGHT}*IDF1")
    print(f"Trials: 1000")
    print(f"Referring mode: threshold (always)")
    print(f"Debug mode: ON (seq='0001', max_expr=2 - hardcoded in eval_referkitti.py)")
    print(f"Dataset: {DATA_ROOT}")
    print(f"Weights: {WEIGHTS}")
    print("=" * 80)
    print("\nOptimizing 13 parameters - FULL RANGES:")
    print("  - Detection: box_threshold [0.1-0.9], text_threshold [0.1-0.9]")
    print("  - Tracker: track_thresh [0.1-0.9], match_thresh [0.1-0.99], track_buffer [10-300]")
    print("  - CLIP: lambda_weight [0.0-1.0], use_clip_in_high, use_clip_in_low")
    print("  - Text-gate: text_gate_mode, text_gate_weight [0.0-1.0]")
    print("  - Referring: referring_thresh [0.0-0.9, threshold mode only]")
    print("  - Scale-aware: small_box_area_thresh [1000-15000]")
    print("  - Filters: use_color_filter, use_spatial_filter")
    print("=" * 80 + "\n")

    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name="referkitti_clip_tracker",
        sampler=optuna.samplers.TPESampler(seed=42),  # Reproducibility
    )

    # Run optimization
    print("Starting optimization...\n")
    study.optimize(objective, n_trials=1000)

    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best combined score: {1.0 - study.best_value:.6f}")
    print(f"  MOTA: {study.best_trial.user_attrs.get('mota', -1):.6f}")
    print(f"  IDF1: {study.best_trial.user_attrs.get('idf1', -1):.6f}")
    print("\nBest hyperparameters:")
    for key, val in sorted(study.best_params.items()):
        print(f"  {key}: {val}")

    # Save and display results
    all_trials = save_results(study, "optuna_referkitti.csv")
    print_top_results(all_trials, top_n=10)

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("  1. Review top 10 parameter sets in optuna_referkitti.csv")
    print("  2. To validate on full dataset:")
    print("     - Edit eval/eval_referkitti.py lines 614, 659")
    print("     - Remove seq_whitelist and max_expr restrictions")
    print("     - Re-run best parameters on full ReferKITTI dataset")
    print("  3. Use the best parameters for production inference")
    print("=" * 80)

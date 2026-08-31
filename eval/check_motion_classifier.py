#!/usr/bin/env python3
"""Calibration check for the motion classifier, against Refer-KITTI's labels.

Same contract as ``eval/check_color_classifier.py``: score the classifier
against a dataset's own ground truth and report how much signal is actually
there, so nothing is tuned or trusted blind.

This one matters more than the colour check, because the naive approach here is
*actively wrong* rather than merely miscalibrated. Refer-KITTI is filmed from a
moving car, so image-space displacement measures ego-motion, not object motion —
a parked car sweeps across the frame while a car matching your speed sits still
in it. The baselines this script prints make that concrete.

Run::

    python eval/check_motion_classifier.py
    python eval/check_motion_classifier.py --states moving
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import motion_classifier as mc
import referkitti_prompt_gt as rkgt
from check_color_classifier import auc

#: Refer-KITTI expressions that assert a motion state over the referred set.
MOTION_EXPRESSIONS = {
    "moving": ["moving-cars", "moving-vehicles"],
    "stationary": ["cars-which-are-parking", "vehicles-which-are-parking",
                   "left-cars-which-are-parking", "left-vehicles-which-are-parking"],
}


class _Box:
    """Minimal stand-in for an STrack: the scorer reads tlwh and track_id."""

    __slots__ = ("track_id", "tlwh")

    def __init__(self, track_id, tlwh):
        self.track_id = track_id
        self.tlwh = tlwh


def evaluate_state(data_root, state, split="training"):
    exprs = MOTION_EXPRESSIONS.get(state, [])
    scorer_pos, scorer_neg = [], []          # three-way scores
    raw_pos, raw_neg = [], []                # naive image displacement
    res_pos, res_neg = [], []                # ego-compensated residual
    n_seq = 0

    for seq in rkgt.list_sequences(data_root, split):
        available = set(rkgt.list_expressions(data_root, seq))
        use = [e for e in exprs if e in available]
        if not use:
            continue
        n_seq += 1

        W, H = rkgt.sequence_image_size(data_root, seq, split)
        boxes = rkgt.load_sequence_boxes(data_root, seq, split)
        gt = rkgt.load_prompt_gt(data_root, seq, use[0], split, boxes_by_frame=boxes)

        by_frame = collections.defaultdict(list)
        for ann in gt["annotations"]:
            by_frame[ann["image_id"]].append(ann)

        scorer = mc.MotionScorer(state)
        for fid in sorted(by_frame):
            anns = by_frame[fid]
            tracks = []
            for a in anns:
                x1, y1, x2, y2 = a["bbox_xyxy"]
                tracks.append(_Box(a["track_id"], (x1, y1, x2 - x1, y2 - y1)))

            scores = scorer.score_frame(fid, tracks, W, H)

            # The two raw features, for the comparison table.
            ids, pts, disps = [], [], []
            for t in tracks:
                got = scorer._displacement(t.track_id, fid)
                if got is not None:
                    ids.append(t.track_id)
                    pts.append(got[0])
                    disps.append(got[1])
            residuals = (mc.ego_residuals(np.array(pts), np.array(disps))
                         if len(ids) >= mc.MIN_TRACKS_FOR_FIT else None)

            valid = {a["track_id"] for a in anns if a["is_target"]}
            for k, tid in enumerate(ids):
                is_pos = tid in valid
                raw = float(np.linalg.norm(disps[k]))
                (raw_pos if is_pos else raw_neg).append(raw)
                if residuals is not None:
                    (res_pos if is_pos else res_neg).append(float(residuals[k]))
            for t in tracks:
                s = scores.get(t.track_id, 0.5)
                (scorer_pos if t.track_id in valid else scorer_neg).append(s)

    # Three-way decision quality, over the track-frames it committed on.
    tp = sum(1 for s in scorer_pos if s == 1.0)
    fn = sum(1 for s in scorer_pos if s == 0.0)
    tn = sum(1 for s in scorer_neg if s == 0.0)
    fp = sum(1 for s in scorer_neg if s == 1.0)
    committed = tp + fn + tn + fp
    total = len(scorer_pos) + len(scorer_neg)
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    with np.errstate(invalid="ignore"):
        balanced = float(np.nanmean([tpr, tnr]))

    # 'stationary' expects a LOW residual, so flip the feature for its AUC.
    flip = state == "stationary"
    f = (lambda v: [-x for x in v]) if flip else (lambda v: v)

    return {
        "state": state,
        "sequences": n_seq,
        "n_pos": len(scorer_pos),
        "n_neg": len(scorer_neg),
        "auc_raw": auc(f(raw_pos), f(raw_neg)),
        "auc_residual": auc(f(res_pos), f(res_neg)),
        "balanced_accuracy": balanced,
        "tpr": tpr,
        "tnr": tnr,
        "abstain_rate": 1.0 - (committed / total) if total else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="dataset/referkitti")
    ap.add_argument("--split", default="training")
    ap.add_argument("--states", default="moving,stationary")
    args = ap.parse_args()

    print("\nMotion classifier calibration — referkitti")
    print("=" * 78)
    print(f"{'state':12} {'seqs':>5} {'pos':>6} {'neg':>6} "
          f"{'AUC raw':>8} {'AUC ego':>8} {'bal.acc':>8} {'TPR':>6} {'TNR':>6} {'abst':>6}")
    print("-" * 78)

    for state in [s.strip() for s in args.states.split(",") if s.strip()]:
        r = evaluate_state(args.data_root, state, args.split)
        if r["n_pos"] == 0:
            print(f"{state:12} {'—':>5}   no labelled expressions in this dataset")
            continue
        print(f"{state:12} {r['sequences']:5d} {r['n_pos']:6d} {r['n_neg']:6d} "
              f"{r['auc_raw']:8.3f} {r['auc_residual']:8.3f} "
              f"{r['balanced_accuracy']:8.3f} {r['tpr']:6.3f} {r['tnr']:6.3f} "
              f"{r['abstain_rate']:6.1%}")

    print("-" * 78)
    print("AUC raw = naive image displacement, the cue a static-camera motion")
    print("  label would use. On a dashcam it is near 0.5 — no signal at all.")
    print("AUC ego = residual from the fitted radial ego-flow field, which is")
    print("  what this classifier actually scores.")
    print("Both are measured on GROUND-TRUTH boxes and so are an upper bound;")
    print("real tracks fragment and switch identity, which only costs signal.\n")


if __name__ == "__main__":
    main()

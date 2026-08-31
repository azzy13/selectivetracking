#!/usr/bin/env python3
"""Calibration check for the colour classifier, against a dataset's own labels.

The point of this script is that the colour classifier is **never tuned
blind**. A threshold that works on Refer-KITTI can be wrong on CARLA, so
before trusting any colour number on a dataset, score the classifier against
that dataset's own colour ground truth and look at the separation.

Both datasets already carry the labels needed:

* **Refer-KITTI** — the colour expressions (``cars-in-black``,
  ``cars-in-silver``, ``cars-in-light-color``, ...) name exactly which track
  ids are that colour, frame by frame. Positives are those; negatives are the
  other annotated objects in the same frames, under the same light.

* **CARLA** — ``gt.json`` annotations carry an explicit ``color`` field as an
  ``"R,G,B"`` string (see ``SEDAN_TYPE_IDS`` / the ``all_red_sedans`` mode in
  ``carla_sim/evaluate_prompt_metrics.py``). That is a far cleaner label than
  Refer-KITTI's, so a CARLA check is strictly more trustworthy.

Reported per colour:

    AUC        separability of the underlying continuous feature (L* for
               achromatic targets, chroma for chromatic). 0.5 is no signal.
    accuracy   of the three-way score, counting only the crops the classifier
               committed on (score != 0.5)
    abstained  share it declined to call — high is honest, not broken

Usage::

    python eval/check_color_classifier.py --dataset referkitti
    python eval/check_color_classifier.py --dataset referkitti --colors black,red
    python eval/check_color_classifier.py --dataset carla \\
        --carla_scenarios dataset/carla_eval/eval_scenarios
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import color_classifier as cc
import referkitti_prompt_gt as rkgt

# Refer-KITTI expressions that assert a colour over the whole referred set.
REFERKITTI_COLOR_EXPRESSIONS = {
    "black": ["cars-in-black", "vehicles-in-black"],
    "silver": ["cars-in-silver", "vehicles-in-silver"],
    "light": ["cars-in-light-color", "vehicles-in-light-color"],
    "white": ["cars-in-white", "vehicles-in-white"],
    "red": ["cars-in-red", "vehicles-in-red"],
}

MIN_CROP_PX = 16


# ----------------------------------------------------------------------
def auc(pos, neg) -> float:
    """P(random positive ranks above random negative). 0.5 = no signal."""
    if not pos or not neg:
        return float("nan")
    p = np.asarray(pos, dtype=np.float64)
    n = np.asarray(neg, dtype=np.float64)
    gt = (p[:, None] > n[None, :]).mean()
    eq = (p[:, None] == n[None, :]).mean()
    return float(gt + 0.5 * eq)


def _crop(rgb, box_xyxy):
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box_xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
        return None
    return rgb[y1:y2, x1:x2]


# ----------------------------------------------------------------------
# Frame collection — each dataset yields the same shape
# ----------------------------------------------------------------------
def frames_referkitti(data_root, target, max_frames_per_seq):
    """Yield [(is_positive, crop_rgb), ...] per frame."""
    expressions = REFERKITTI_COLOR_EXPRESSIONS.get(target, [])
    for seq in rkgt.list_sequences(data_root):
        available = set(rkgt.list_expressions(data_root, seq))
        use = [e for e in expressions if e in available]
        if not use:
            continue
        boxes = rkgt.load_sequence_boxes(data_root, seq)
        gt = rkgt.load_prompt_gt(data_root, seq, use[0], boxes_by_frame=boxes)

        by_frame = collections.defaultdict(list)
        for ann in gt["annotations"]:
            by_frame[ann["image_id"]].append(ann)

        img_dir = os.path.join(data_root, "KITTI", "training", "image_02", seq)
        ordered = sorted(by_frame)
        step = max(1, len(ordered) // max_frames_per_seq)
        for fid in ordered[::step][:max_frames_per_seq]:
            img = cv2.imread(os.path.join(img_dir, f"{fid:06d}.png"))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rec = []
            for ann in by_frame[fid]:
                crop = _crop(rgb, ann["bbox_xyxy"])
                if crop is not None:
                    rec.append((bool(ann["is_target"]), crop))
            if rec:
                yield rec


def _carla_color_label(color_str):
    """CARLA's "R,G,B" -> a classifier label, via LAB on the raw paint value.

    The GT stores the vehicle's actual paint colour, so it can be labelled
    with the same rule the classifier uses — no hand-mapping of RGB triples.
    """
    try:
        r, g, b = (int(v) for v in str(color_str).split(","))
    except (ValueError, AttributeError):
        return None
    px = np.array([[[r, g, b]]], dtype=np.uint8)
    return cc.classify_patch(*cc.lab_stats(px))


def frames_carla(scenarios_root, target, max_frames_per_seq):
    for scenario in sorted(os.listdir(scenarios_root)):
        gt_path = os.path.join(scenarios_root, scenario, "gt.json")
        if not os.path.isfile(gt_path):
            continue
        with open(gt_path) as f:
            gt = json.load(f)

        images = {im["id"]: im for im in gt.get("images", [])}
        by_frame = collections.defaultdict(list)
        for ann in gt.get("annotations", []):
            by_frame[ann["image_id"]].append(ann)

        ordered = sorted(by_frame)
        step = max(1, len(ordered) // max_frames_per_seq)
        for fid in ordered[::step][:max_frames_per_seq]:
            meta = images.get(fid)
            if not meta:
                continue
            img_path = meta.get("file_name") or meta.get("path")
            if img_path and not os.path.isabs(img_path):
                img_path = os.path.join(scenarios_root, scenario, img_path)
            img = cv2.imread(img_path) if img_path else None
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rec = []
            for ann in by_frame[fid]:
                label = _carla_color_label(ann.get("color"))
                if label is None:
                    continue
                box = ann.get("bbox_xyxy")
                if box is None and ann.get("bbox"):
                    x, y, w, h = ann["bbox"]
                    box = [x, y, x + w, y + h]
                if box is None:
                    continue
                crop = _crop(rgb, box)
                if crop is not None:
                    rec.append((label == cc.canonical_color(target), crop))
            if rec:
                yield rec


# ----------------------------------------------------------------------
def evaluate(target, frame_iter, use_peers=True):
    canon = cc.canonical_color(target) or target
    achromatic = canon in cc.ACHROMATIC_LABELS

    feat_pos, feat_neg = [], []
    # Confusion counts over the crops the classifier committed on.  Balanced
    # accuracy (mean of TPR and TNR) rather than raw accuracy, because these
    # sets run 30-40% positive: a rule that abstains into the majority class
    # scores well on raw accuracy while being useless.
    tp = fn = tn = fp = 0
    n_abstain = n_pos = n_neg = 0
    fallback_frames = total_frames = 0

    for rec in frame_iter:
        total_frames += 1
        if len(rec) < cc.MIN_PEERS:
            fallback_frames += 1
        crops = [c for _, c in rec]

        scores = (cc.peer_relative_scores(crops, canon) if use_peers
                  else [cc.score_votes(cc.patch_votes(c)[1], canon,
                                       crop_chroma=cc.lab_stats(c)[1])
                        for c in crops])

        for (is_pos, crop), score in zip(rec, scores):
            L, chroma, _ = cc.lab_stats(crop)
            # The continuous feature the decision rests on. For 'dark' a lower
            # L* should indicate a positive, so negate to keep AUC > 0.5 = good.
            if achromatic:
                feature = -L if canon == "dark" else L
            else:
                feature = chroma
            (feat_pos if is_pos else feat_neg).append(feature)
            n_pos += is_pos
            n_neg += not is_pos

            if score == 0.5:
                n_abstain += 1
                continue
            said_yes = score == 1.0
            if is_pos and said_yes:
                tp += 1
            elif is_pos:
                fn += 1
            elif said_yes:
                fp += 1
            else:
                tn += 1

    total = n_pos + n_neg
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    with np.errstate(invalid="ignore"):
        balanced = float(np.nanmean([tpr, tnr]))
    return {
        "target": target,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "base_rate": n_pos / total if total else float("nan"),
        "auc": auc(feat_pos, feat_neg),
        "balanced_accuracy": balanced,
        "tpr": tpr,
        "tnr": tnr,
        "abstain_rate": n_abstain / total if total else float("nan"),
        "fallback_frames": fallback_frames,
        "total_frames": total_frames,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["referkitti", "carla"], default="referkitti")
    ap.add_argument("--data_root", default="dataset/referkitti")
    ap.add_argument("--carla_scenarios", default=None,
                    help="Directory of CARLA scenario folders, each with gt.json")
    ap.add_argument("--colors", default="black,silver,light,white,red",
                    help="Comma-separated colour words to check")
    ap.add_argument("--max_frames_per_seq", type=int, default=12)
    ap.add_argument("--no_peers", action="store_true",
                    help="Score with absolute thresholds only, for comparison")
    args = ap.parse_args()

    if args.dataset == "carla" and not args.carla_scenarios:
        raise SystemExit("--carla_scenarios is required for --dataset carla")

    targets = [c.strip() for c in args.colors.split(",") if c.strip()]

    print(f"\nColour classifier calibration — {args.dataset}"
          f"{' (absolute thresholds)' if args.no_peers else ' (peer-relative)'}")
    print("=" * 78)
    print(f"{'colour':10} {'pos':>6} {'neg':>6} {'AUC':>7} {'bal.acc':>8} "
          f"{'TPR':>6} {'TNR':>6} {'abstain':>8} {'fallback':>9}")
    print("-" * 78)

    any_rows = False
    for target in targets:
        if args.dataset == "referkitti":
            it = frames_referkitti(args.data_root, target, args.max_frames_per_seq)
        else:
            it = frames_carla(args.carla_scenarios, target, args.max_frames_per_seq)

        r = evaluate(target, it, use_peers=not args.no_peers)
        if r["n_pos"] == 0:
            print(f"{target:10} {'—':>6} {'—':>6}   no labelled positives in this dataset")
            continue
        any_rows = True
        frac = (r["fallback_frames"] / r["total_frames"]) if r["total_frames"] else 0.0
        print(f"{target:10} {r['n_pos']:6d} {r['n_neg']:6d} {r['auc']:7.3f} "
              f"{r['balanced_accuracy']:8.3f} {r['tpr']:6.3f} {r['tnr']:6.3f} "
              f"{r['abstain_rate']:8.1%} {frac:9.0%}")

    print("-" * 78)
    if any_rows:
        print("AUC 0.5 = no signal, on the continuous feature behind the decision.")
        print("bal.acc = mean(TPR, TNR) over the crops the classifier committed on —")
        print("  balanced, not raw, because these sets run ~30-40% positive and a rule")
        print("  that abstains into the majority class flatters raw accuracy.")
        print("abstain = share it declined to call. Honest uncertainty, not error.")
        print(f"fallback = frames with < {cc.MIN_PEERS} detections, where there is no")
        print("  peer ordering to read and the absolute rule is used instead.")
    print()


if __name__ == "__main__":
    main()

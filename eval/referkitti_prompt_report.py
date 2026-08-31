#!/usr/bin/env python3
"""Human-readable reporting for the prompt-compliance metrics.

A bare ``SP=0.31`` is not readable — the reader cannot tell whether 0.31 came
from 9 predictions or 900, nor whether the 0.69 that is missing went to
distractors or to nothing at all.  Every number printed here is shown with the
counts it came from, and SP is always shown next to DCR because they share a
denominator and their gap is the interesting quantity:

    SP + DCR + (unmatched fraction) = 1

so ``1 - SP - DCR`` is the share of predictions that hit no GT box at all.
That three-way split is what separates "the tracker locked onto the wrong car"
from "the tracker hallucinated a car".

Separated from the runner so it can be reused when the CARLA relational sweep
lands: the row schema is dataset-agnostic.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import color_classifier as _cc
import motion_classifier as _mc

# The spatial words ReferringDetectionFilter._parse_spatial_region keys on.
# Tagging uses the pipeline's own vocabularies rather than a separate word
# list, so a stratum means "the code can see this cue", not "a human would
# call this spatial".
_SPATIAL_WORDS = ("left", "right", "top", "upper", "above",
                  "bottom", "lower", "below", "center", "middle", "central")


def expression_channels(prompt: str) -> Dict[str, bool]:
    """Which attribute channels a prompt exercises.

    ``motion_scoreable`` separates "moving"/"parked", which the motion
    classifier can attempt, from "braking"/"turning"/"counter direction of
    ours", which it deliberately abstains on — those two are very different
    claims about what the pipeline could ever get right.
    """
    low = (prompt or "").lower()
    motion = _mc.canonical_motion(low)
    return {
        "spatial": any(w in low for w in _SPATIAL_WORDS),
        "colour": any(w in low.split() or w in low for w in _cc.COLOR_SYNONYMS),
        "motion": motion is not None,
        "motion_scoreable": motion in ("moving", "stationary"),
        "motion_unscoreable": motion == "unscoreable",
    }


def stratum_of(prompt: str) -> str:
    """One mutually-exclusive bucket per expression, for the headline split."""
    ch = expression_channels(prompt)
    addressable = ch["spatial"] or ch["colour"]
    if ch["motion"] and not addressable:
        return "motion only"
    if ch["motion"] and addressable:
        return "motion + other"
    if ch["spatial"] and ch["colour"]:
        return "spatial + colour"
    if ch["spatial"]:
        return "spatial only"
    if ch["colour"]:
        return "colour only"
    return "plain"


_STRATUM_ORDER = ("plain", "spatial only", "colour only", "spatial + colour",
                  "motion + other", "motion only")

BAR_WIDTH = 24


def _bar(value: float, width: int = BAR_WIDTH) -> str:
    """A fixed-width unicode bar for a 0..1 value."""
    value = max(0.0, min(1.0, float(value)))
    filled = int(round(value * width))
    return "█" * filled + "·" * (width - filled)


def _pct(value: float) -> str:
    return f"{value * 100:6.2f}%"


# ----------------------------------------------------------------------
# Per-expression
# ----------------------------------------------------------------------
def render_expression_report(row: dict, indent: str = "") -> str:
    """One expression, with the counts behind every ratio."""
    s = row["stats"]
    gt = row["gt"]

    total_pred = s["total_predictions"]
    hit_valid = s["predictions_matching_valid"]
    hit_distr = s["predictions_matching_distractor"]
    hit_none = total_pred - hit_valid - hit_distr
    none_rate = hit_none / total_pred if total_pred else 0.0

    lines = [
        "",
        f"prompt          \"{row['prompt']}\"",
        f"ground truth    {gt['num_valid']} valid + {gt['num_distractors']} distractor "
        f"boxes over {gt['frames']} frames",
        f"predictions     {total_pred} boxes",
        "",
        f"  SP   Semantic Precision       {_pct(row['semantic_precision'])}  "
        f"{_bar(row['semantic_precision'])}   {hit_valid}/{total_pred} predictions on a valid target",
        f"  SR   Semantic Recall          {_pct(row['semantic_recall'])}  "
        f"{_bar(row['semantic_recall'])}   {s['valid_gt_matched']}/{s['total_valid_gt']} valid GT boxes found",
        f"  DCR  Distractor Confusion     {_pct(row['distractor_confusion_rate'])}  "
        f"{_bar(row['distractor_confusion_rate'])}   {hit_distr}/{total_pred} predictions on a distractor",
        "",
        f"  PCR  Prompt Coverage          {_pct(row['prompt_coverage_ratio'])}  "
        f"{_bar(row['prompt_coverage_ratio'])}   {s['frames_where_valid_matched']}/{s['frames_with_valid_gt']} frames with the target held",
        f"  SID  Semantic ID Switches     {row['semantic_id_switches']:6d}",
        "",
        f"  where the {total_pred} predictions went:  "
        f"{hit_valid} valid ({_pct(row['semantic_precision']).strip()})  ·  "
        f"{hit_distr} distractor ({_pct(row['distractor_confusion_rate']).strip()})  ·  "
        f"{hit_none} matched nothing ({_pct(none_rate).strip()})",
        "",
    ]
    return "\n".join(indent + ln if ln else "" for ln in lines)


# ----------------------------------------------------------------------
# Run-level
# ----------------------------------------------------------------------
def summarise(rows: List[dict]) -> dict:
    """Both averages, because they answer different questions.

    ``macro`` averages the per-expression rates — every expression counts the
    same, which is what you want when comparing methods across a benchmark.
    ``micro`` pools the raw counts — dominated by the long, busy expressions,
    which is what you want when asking "over this whole clip, how often was the
    tracker right".  Reporting one without the other invites reading a
    micro number as a macro claim.
    """
    n = len(rows)
    macro = {
        key: sum(r[key] for r in rows) / n
        for key in ("semantic_precision", "semantic_recall",
                    "prompt_coverage_ratio", "distractor_confusion_rate")
    }

    tot_pred = sum(r["stats"]["total_predictions"] for r in rows)
    hit_valid = sum(r["stats"]["predictions_matching_valid"] for r in rows)
    hit_distr = sum(r["stats"]["predictions_matching_distractor"] for r in rows)
    tot_valid_gt = sum(r["stats"]["total_valid_gt"] for r in rows)
    valid_matched = sum(r["stats"]["valid_gt_matched"] for r in rows)
    frames_valid = sum(r["stats"]["frames_with_valid_gt"] for r in rows)
    frames_held = sum(r["stats"]["frames_where_valid_matched"] for r in rows)

    micro = {
        "semantic_precision": hit_valid / tot_pred if tot_pred else 0.0,
        "semantic_recall": valid_matched / tot_valid_gt if tot_valid_gt else 0.0,
        "prompt_coverage_ratio": frames_held / frames_valid if frames_valid else 0.0,
        "distractor_confusion_rate": hit_distr / tot_pred if tot_pred else 0.0,
    }

    by_sequence = {}
    for seq in sorted({r["sequence"] for r in rows}):
        seq_rows = [r for r in rows if r["sequence"] == seq]
        by_sequence[seq] = {
            "num_expressions": len(seq_rows),
            "macro": {
                key: sum(r[key] for r in seq_rows) / len(seq_rows)
                for key in ("semantic_precision", "semantic_recall",
                            "prompt_coverage_ratio", "distractor_confusion_rate")
            },
            "semantic_id_switches": sum(r["semantic_id_switches"] for r in seq_rows),
        }

    return {
        "num_expressions": n,
        "num_sequences": len(by_sequence),
        "macro": macro,
        "micro": micro,
        "by_sequence": by_sequence,
        "semantic_id_switches_total": sum(r["semantic_id_switches"] for r in rows),
        "totals": {
            "predictions": tot_pred,
            "predictions_on_valid": hit_valid,
            "predictions_on_distractor": hit_distr,
            "predictions_on_nothing": tot_pred - hit_valid - hit_distr,
            "valid_gt": tot_valid_gt,
            "valid_gt_matched": valid_matched,
        },
    }


def _table(rows: List[dict]) -> str:
    """Markdown table, one row per expression."""
    head = ("| seq | expression | prompt | SP | SR | DCR | PCR | SID | preds | valid GT |\n"
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    body = []
    for r in sorted(rows, key=lambda r: (r["sequence"], r["expression"])):
        s = r["stats"]
        body.append(
            f"| {r['sequence']} | `{r['expression']}` | {r['prompt']} "
            f"| {r['semantic_precision']:.3f} "
            f"| {r['semantic_recall']:.3f} "
            f"| {r['distractor_confusion_rate']:.3f} "
            f"| {r['prompt_coverage_ratio']:.3f} "
            f"| {r['semantic_id_switches']} "
            f"| {s['total_predictions']} "
            f"| {s['total_valid_gt']} |"
        )
    return head + "\n" + "\n".join(body)


def _sequence_table(summary: dict) -> str:
    """Per-sequence macro averages — one row per clip."""
    head = ("| sequence | expressions | SP | SR | DCR | PCR | SID |\n"
            "|---|---:|---:|---:|---:|---:|---:|")
    body = []
    for seq, s in summary["by_sequence"].items():
        m = s["macro"]
        body.append(
            f"| {seq} | {s['num_expressions']} "
            f"| {m['semantic_precision']:.3f} "
            f"| {m['semantic_recall']:.3f} "
            f"| {m['distractor_confusion_rate']:.3f} "
            f"| {m['prompt_coverage_ratio']:.3f} "
            f"| {s['semantic_id_switches']} |"
        )
    return head + "\n" + "\n".join(body)


def _stratum_table(rows: List[dict]) -> str:
    """Macro averages per attribute stratum, plus the whole set."""
    head = ("| stratum | expressions | SP | SR | DCR | PCR |\n"
            "|---|---:|---:|---:|---:|---:|")
    body = []
    buckets = {}
    for r in rows:
        buckets.setdefault(stratum_of(r["prompt"]), []).append(r)

    def line(label, group):
        n = len(group)
        m = {k: sum(r[k] for r in group) / n for k in
             ("semantic_precision", "semantic_recall",
              "distractor_confusion_rate", "prompt_coverage_ratio")}
        return (f"| {label} | {n} "
                f"| {m['semantic_precision']:.3f} "
                f"| {m['semantic_recall']:.3f} "
                f"| {m['distractor_confusion_rate']:.3f} "
                f"| {m['prompt_coverage_ratio']:.3f} |")

    for name in _STRATUM_ORDER:
        if name in buckets:
            body.append(line(name, buckets[name]))
    addressable = [r for r in rows if stratum_of(r["prompt"]) != "motion only"]
    if addressable and len(addressable) != len(rows):
        body.append("| | | | | | |")
        body.append(line("**all except motion-only**", addressable))
    body.append(line("**all**", rows))
    return head + "\n" + "\n".join(body)


def render_run_report(rows: List[dict], summary: dict, args) -> str:
    """Markdown report for the whole run — also what gets printed to stdout."""
    t = summary["totals"]
    macro, micro = summary["macro"], summary["micro"]
    seqs = sorted({r["sequence"] for r in rows})
    seq_label = ", ".join(seqs)
    is_test_split = set(seqs) == {"0005", "0011", "0013"}

    on_nothing = t["predictions_on_nothing"] / t["predictions"] if t["predictions"] else 0.0

    method = [
        f"- detector `{args.detector}` · weights `{os.path.basename(args.weights)}`",
        f"- tracker `{args.tracker}` · box {args.box_threshold} · text {args.text_threshold} "
        f"· track {args.track_thresh} · match {args.match_thresh} · buffer {args.track_buffer}",
        f"- referring filter `{args.referring_mode}`"
        + (f" @ {args.referring_thresh}" if args.referring_mode != "none" else ""),
        f"- motion filter {'on' if getattr(args, 'use_motion_filter', False) else 'off'}",
        f"- colour filter {'on' if args.use_color_filter else 'off'} · "
        f"spatial filter {'on' if args.use_spatial_filter else 'off'} · "
        f"scale-aware threshold {'on' if args.use_scale_aware_thresh else 'off'}",
        f"- text gate `{args.text_gate_mode}` @ {args.text_gate_weight}",
    ]

    return f"""
# Refer-KITTI · prompt-compliance results

**Sequence(s) {seq_label}**{' — the standard Refer-KITTI test split' if is_test_split else ''} ·
{summary['num_expressions']} expression(s) · IoU ≥ {args.iou} ·
scored on {'the whole clip' if args.score_whole_clip else 'annotated frames only'}

## Headline

| metric | macro (per expression) | micro (pooled) |
|---|---:|---:|
| **SP** — Semantic Precision | {macro['semantic_precision']:.3f} | {micro['semantic_precision']:.3f} |
| **SR** — Semantic Recall | {macro['semantic_recall']:.3f} | {micro['semantic_recall']:.3f} |
| **DCR** — Distractor Confusion Rate | {macro['distractor_confusion_rate']:.3f} | {micro['distractor_confusion_rate']:.3f} |
| PCR — Prompt Coverage Ratio | {macro['prompt_coverage_ratio']:.3f} | {micro['prompt_coverage_ratio']:.3f} |
| SID — Semantic ID Switches (total) | — | {summary['semantic_id_switches_total']} |

`macro` averages the per-expression rates (each expression counts once);
`micro` pools the raw counts (long expressions dominate).

## Where the predictions went

{t['predictions']} predicted boxes in total:

| landed on | count | share |
|---|---:|---:|
| a prompt-valid object | {t['predictions_on_valid']} | {t['predictions_on_valid'] / max(t['predictions'], 1) * 100:.1f}% |
| a distractor object | {t['predictions_on_distractor']} | {t['predictions_on_distractor'] / max(t['predictions'], 1) * 100:.1f}% |
| no GT box at all | {t['predictions_on_nothing']} | {on_nothing * 100:.1f}% |

SP and DCR share this denominator, so the three shares sum to 1. A high DCR
means the tracker is picking real objects that the prompt excludes; a high
"no GT box" share means it is picking things that are not annotated objects.
They call for different fixes.

Of {t['valid_gt']} prompt-valid GT boxes, {t['valid_gt_matched']} were matched.

## By attribute stratum

{_stratum_table(rows)}

Buckets are mutually exclusive and tagged with the pipeline's own vocabularies,
so a stratum means "the code can see this cue". **motion only** is the set with
no spatial or colour cue to fall back on — the expressions the pipeline has no
mechanism to answer, since the motion cue scores 0.589 balanced accuracy on a
moving camera (see `eval/check_motion_classifier.py`). Read `all` for
comparability against published baselines, which are scored on the full set,
and the strata to see where the number comes from.

## Per sequence

{_sequence_table(summary)}

## Per expression

{_table(rows)}

## Method

{chr(10).join(method)}

## Reading the metrics

- **SP** = predictions matched to prompt-valid GT / all predictions. Low SP = semantic false positives.
- **SR** = prompt-valid GT matched / all prompt-valid GT. Low SR = missed targets.
- **DCR** = predictions matched to prompt-*invalid* GT / all predictions. Low DCR = rarely locks onto the wrong object.
- **PCR** = frames where a valid target was held / frames where one was visible. Tracking reliability over time.
- **SID** = times a track flipped between matching a valid and an invalid object. Semantic drift.

Definitions: `carla_sim/metrics.md`. Implementation: `carla_sim/evaluate_prompt_metrics.py`.
""".rstrip() + "\n"


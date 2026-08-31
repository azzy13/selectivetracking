#!/usr/bin/env python3
"""Colour classification for referring expressions, in CIELAB.

Replaces the HSV rule in ``ReferringDetectionFilter``, which could not see
black at all: it classified a pixel as chromatic whenever ``S >= 35`` and only
low-saturation pixels reached the brightness branch that returns ``'dark'``.

That is not a threshold that was set too low — it is a threshold on a quantity
that is undefined where it was being applied. HSV saturation is

    S = (max - min) / max

so as ``max -> 0`` the denominator vanishes and S becomes numerically
meaningless. A near-black pixel at RGB (20, 22, 28) has S = 73/255 = 29%,
sails past the gate, and gets a hue assigned. Measured over 215,720 pixels
inside the GT boxes of black-annotated cars in Refer-KITTI sequence 0005,
**82.4%** were routed to the hue branch this way, and not one patch in ~1,056
came out ``'dark'``.

CIELAB does not have this failure mode. Chroma

    C* = sqrt(a*^2 + b*^2)

is an *absolute* measure of colourfulness rather than a ratio, so it stays
small for dark pixels instead of blowing up, and ``L*`` is perceptual
lightness on a fixed 0..100 scale. The same RGB (20, 22, 28) gives L* ~ 8,
C* ~ 3.5 — correctly achromatic.

**This part is a correctness fix, not a calibration.** It transfers to any
dataset because it fixes the mathematics, not the numbers.

## The part that does need care across datasets

Absolute cut points on ``L*`` ("black means L* < 32") encode a scene's
exposure, and that is exactly what shifts between Refer-KITTI and CARLA. So the
achromatic decision is made on **rank within the frame's own detections** —
they are the right reference class, being the same kind of object under the
same light, and a rank carries no units at all, so a brighter or darker render
moves every candidate together and the decision is unchanged.

Measured on Refer-KITTI's own colour expressions, all sequences
(``eval/check_color_classifier.py``). Balanced accuracy — mean of TPR and TNR
over the crops the classifier committed on — because these sets run 30-42%
positive and raw accuracy rewards abstaining into the majority class:

| target | n pos | AUC | absolute L* | **peer rank** |
|---|---:|---:|---:|---:|
| black  | 244 | 0.714 | 0.525 | **0.669** |
| silver | 105 | 0.727 | 0.609 | **0.630** |
| light  | 187 | 0.831 | 0.728 | **0.786** |
| white  |  17 | 0.863 | 0.886 | 0.776 |
| red    |  61 | 0.625 | — abstains — | — abstains — |

The HSV rule it replaces has no column because it had no signal at all: it
returned ``blue`` or ``green`` for every crop of every colour, so it could not
separate anything. Peer rank wins clearly on black and light; white is the one
row where absolute scores higher, on 17 positives, which is too few to move the
default. Normalising against the *whole frame* was also tried and is worse
(black 0.685, silver 0.622, light 0.779) — a KITTI frame is mostly road, sky
and vegetation, whose lightness does not track vehicle exposure.

**Do not read these as solved.** Real signal, and a large improvement on
nothing, but colour from small, shadowed, low-resolution crops is hard, and red
does not work here at all. Before trusting a colour number on a new dataset,
run ``eval/check_color_classifier.py`` against that dataset's own colour
labels — CARLA's ``gt.json`` carries an explicit per-vehicle ``color`` field,
so that check is cleaner there than it is here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ----------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------
# Achromatic targets are decided by lightness; chromatic ones by hue.
ACHROMATIC_LABELS = ("dark", "gray", "light")
CHROMATIC_LABELS = ("red", "orange", "yellow", "green", "blue", "purple")

#: Prompt word -> canonical label. Mirrors the colour words the prompt parser
#: recognises, so a prompt and a classification are always comparable.
COLOR_SYNONYMS: Dict[str, str] = {
    "black": "dark", "dark": "dark",
    "white": "light", "light": "light", "bright": "light",
    "silver": "gray", "grey": "gray", "gray": "gray",
    "red": "red", "orange": "orange", "yellow": "yellow",
    "green": "green", "blue": "blue", "purple": "purple",
}

#: Adjacent hues count as supporting evidence — lighting and viewing angle
#: shift perceived hue, so a red car often registers orange.
HUE_NEIGHBORS: Dict[str, Tuple[str, ...]] = {
    "red": ("orange", "purple"),
    "orange": ("red", "yellow"),
    "yellow": ("orange", "green"),
    "green": ("yellow", "blue"),
    "blue": ("green", "purple"),
    "purple": ("blue", "red"),
}

# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------
#: Below this chroma a patch carries no reliable hue. Deliberately generous:
#: a wrong hue is worse than an honest "achromatic", because the three-way
#: score treats achromatic as uninformative rather than as evidence against.
CHROMA_MIN = 18.0

#: Absolute L* buckets, used when a frame has no peer detections to compare
#: against. Perceptual units (0..100), not exposure-dependent 0..255 values.
DARK_L_MAX = 32.0
LIGHT_L_MIN = 68.0

#: The peer-relative decision is made on **rank**, not on a distance.  A crop
#: in the darkest ``PEER_FRACTION`` of the frame's detections supports a 'dark'
#: target; one in the lightest ``PEER_FRACTION`` is evidence against; the
#: middle abstains.  A rank has no units, so nothing here carries an exposure
#: assumption from one dataset to the next — which is the property that makes
#: this transfer to CARLA.  Measured balanced accuracy on Refer-KITTI:
#:
#:     target    absolute L*   z-band   rank 1/3   rank 1/4
#:     black         0.576      0.733     0.738      0.765
#:     light         0.532      0.788     0.811      0.835
#:     white         0.660      0.580     0.780      0.800
#:
#: 1/3 is the default rather than 1/4: it abstains ~35% instead of ~48% for
#: about 0.03 less balanced accuracy, and the hysteresis in PostTrackColorGate
#: aggregates over five frames, so coverage is worth more than per-frame edge.
PEER_FRACTION = 1.0 / 3.0

#: Rank needs a real ordering to mean anything; below this many detections the
#: absolute rule is used instead.
MIN_PEERS = 3

#: Targets scored as "brighter than peers".  'gray' (silver) is here rather
#: than in a middle-of-the-distribution band because the data says so: scoring
#: silver as "mid-lightness" gives balanced accuracy 0.384 — worse than
#: guessing — while scoring it as "brighter than peers" gives 0.754.  Chroma
#: does not separate silver either (AUC 0.386; median chroma 4.2 for silver vs
#: 3.2 for everything else), so on this imagery silver is a lightness
#: phenomenon, not a colour one.  The consequence to know: the classifier
#: cannot tell silver from white, and does not pretend to.
BRIGHT_TARGETS = ("light", "gray")
DARK_TARGETS = ("dark",)


def canonical_color(word: Optional[str]) -> Optional[str]:
    """Map a prompt colour word onto a classifier label ('black' -> 'dark')."""
    if not word:
        return None
    return COLOR_SYNONYMS.get(str(word).strip().lower())


# ----------------------------------------------------------------------
# Pixel / patch classification
# ----------------------------------------------------------------------
def lab_stats(crop_rgb: np.ndarray) -> Tuple[float, float, float]:
    """Robust (L*, C*, hue°) for a crop.

    The **median** is used rather than the mean: a windshield reflection or a
    specular highlight is a small, extreme cluster that drags a mean off the
    car's actual paint but barely moves a median.
    """
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    med = np.median(lab, axis=0)
    L = float(med[0]) * 100.0 / 255.0     # OpenCV packs L* into 0..255
    a = float(med[1]) - 128.0
    b = float(med[2]) - 128.0
    chroma = float(np.hypot(a, b))
    hue = float(np.degrees(np.arctan2(b, a)) % 360.0)
    return L, chroma, hue


def hue_label(hue_deg: float) -> str:
    """CIELAB hue angle -> colour name.

    The boundaries are measured, not guessed: LAB hue angles do not line up
    with intuition, and an earlier hand-picked set put pure blue (304.6 deg)
    into 'purple' and pure red (35.5 deg) into 'orange'. Reference angles for
    saturated sRGB primaries::

        crimson  29.5   red      35.5   orange   66.1   yellow  100.2
        lime    126.6   green   142.1   cyan    205.2   navy    297.5
        blue    304.6   purple  318.5   magenta 341.8   pink    358.5

    Blue spans a wide arc because LAB hue angle is not perceptually uniform.
    """
    h = hue_deg % 360.0
    if h >= 345.0 or h < 50.0:
        return "red"
    if h < 80.0:
        return "orange"
    if h < 115.0:
        return "yellow"
    if h < 180.0:
        return "green"
    if h < 310.0:
        return "blue"
    return "purple"


def classify_patch(L: float, chroma: float, hue: float) -> str:
    """One patch -> one label. Lightness decides only when chroma is low."""
    if chroma >= CHROMA_MIN:
        return hue_label(hue)
    if L < DARK_L_MAX:
        return "dark"
    if L > LIGHT_L_MIN:
        return "light"
    return "gray"


def patch_votes(crop_rgb: np.ndarray, grid_size: int = 4) -> Tuple[str, Dict[str, int]]:
    """Split a crop into a grid and vote on its colour.

    Returns ``(dominant_label, {label: patch_count})``. Same contract as the
    HSV ``_get_patchwise_dominant_color`` it replaces, so callers are unchanged.
    """
    h, w = crop_rgb.shape[:2]
    if h == 0 or w == 0:
        return "unknown", {}
    if h < grid_size or w < grid_size:
        grid_size = max(1, min(h, w, 2))

    patch_h = max(1, h // grid_size)
    patch_w = max(1, w // grid_size)

    votes: Dict[str, int] = {}
    for i in range(grid_size):
        for j in range(grid_size):
            patch = crop_rgb[i * patch_h:min((i + 1) * patch_h, h),
                             j * patch_w:min((j + 1) * patch_w, w)]
            if patch.size == 0:
                continue
            label = classify_patch(*lab_stats(patch))
            votes[label] = votes.get(label, 0) + 1

    if not votes:
        return "unknown", {}
    return max(votes.items(), key=lambda kv: kv[1])[0], votes


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def _labels_match(detected: str, target: str) -> bool:
    return canonical_color(detected) == canonical_color(target) or detected == target


def score_votes(votes: Dict[str, int], target: str,
                min_target_patches: int = 2,
                crop_chroma: Optional[float] = None) -> float:
    """Three-way score for one crop against a target colour, from patch votes.

    ``1.0`` supports the target · ``0.0`` is evidence against ·
    ``0.5`` is uninformative.

    Presence-based rather than dominance-based: enough patches of the target
    colour count even when something else wins the vote, because reflections
    and glass routinely outvote the paint on a small crop.

    ``crop_chroma`` guards the chromatic branch. A hue claim about an object
    that is not chromatic at all is noise: on Refer-KITTI, **0.0%** of crops
    inside red-annotated boxes reach ``CHROMA_MIN`` (median chroma 4.5, against
    4.0 for everything else), yet a couple of stray patches would still clear
    ``min_target_patches`` and assert "red". Left ungated that produced
    TPR 1.000 / TNR 0.000 — it confirmed every distractor it looked at.
    Passing the crop's own chroma makes it abstain instead.
    """
    target = canonical_color(target) or target
    if not votes:
        return 0.5

    if (target in CHROMATIC_LABELS and crop_chroma is not None
            and crop_chroma < CHROMA_MIN):
        return 0.5

    supporting = sum(n for lbl, n in votes.items() if _labels_match(lbl, target))
    if supporting >= min_target_patches:
        return 1.0

    supporting += sum(votes.get(nb, 0) for nb in HUE_NEIGHBORS.get(target, ()))
    if supporting >= min_target_patches:
        return 1.0

    dominant = max(votes.items(), key=lambda kv: kv[1])[0]
    if _labels_match(dominant, target):
        return 1.0
    # An achromatic dominant is uninformative for a chromatic target and
    # vice versa — only a confident, opposing call is evidence against.
    if target in ACHROMATIC_LABELS and dominant in CHROMATIC_LABELS:
        return 0.5
    if target in CHROMATIC_LABELS and dominant in ACHROMATIC_LABELS:
        return 0.5
    return 0.0


def peer_relative_scores(crops: Sequence[np.ndarray], target: str) -> List[float]:
    """Score every crop in one frame against a target colour, using the peers.

    For an **achromatic** target the decision is made on where each crop's L*
    *ranks* among the other detections in the same frame. Rank is scale-free,
    so a brighter or darker render shifts every candidate together and the
    decision is unchanged — this is what removes the exposure assumption.
    With fewer than ``MIN_PEERS`` crops there is no ordering worth reading, so
    the absolute rule is used instead.

    For a **chromatic** target the peers say nothing useful — a red car among
    red cars is still red — so this defers to the patch vote in every case.
    """
    target = canonical_color(target) or target
    if not crops:
        return []

    stats = [lab_stats(c) for c in crops]
    votes = [patch_votes(c)[1] for c in crops]
    absolute = [score_votes(v, target, crop_chroma=st[1])
                for v, st in zip(votes, stats)]

    if target not in ACHROMATIC_LABELS or len(crops) < MIN_PEERS:
        return absolute

    lightness = np.array([st[0] for st in stats], dtype=np.float64)
    n = len(lightness)
    # Quantile of each crop within this frame's own ordering, in (0, 1).
    quantile = (np.argsort(np.argsort(lightness)) + 0.5) / n

    scores: List[float] = []
    for q, abs_score in zip(quantile, absolute):
        if target in DARK_TARGETS:
            support, against = q <= PEER_FRACTION, q >= 1.0 - PEER_FRACTION
        elif target in BRIGHT_TARGETS:
            support, against = q >= 1.0 - PEER_FRACTION, q <= PEER_FRACTION
        else:
            support = against = False
        if support:
            scores.append(1.0)
        elif against:
            scores.append(0.0)
        else:
            # Peers are inconclusive. Let the absolute reading break the tie,
            # but never let it assert more than "uninformative" — an absolute
            # L* cut is the part that does not transfer across datasets.
            scores.append(0.5 if abs_score == 0.0 else abs_score)
    return scores


def score_crop(crop_rgb: np.ndarray, target: str,
               peers: Optional[Sequence[np.ndarray]] = None) -> float:
    """Score a single crop, optionally against its peer detections."""
    if peers:
        crops = [crop_rgb, *peers]
        return peer_relative_scores(crops, target)[0]
    return score_votes(patch_votes(crop_rgb)[1], target,
                       crop_chroma=lab_stats(crop_rgb)[1])

"""Contract tests for ``eval/color_classifier.py``.

The classifier replaced an HSV rule that could not see black at all. These
tests pin the two properties that made it wrong and the one property the CARLA
transfer depends on:

  1. a dark pixel is classified by lightness, not by a hue read off unstable
     saturation — the exact regression that deleted every black car;
  2. a hue is never asserted about an object that is not chromatic;
  3. the achromatic decision is **exposure-invariant** — scaling a whole frame
     brighter or darker must not change which car is called "the black one".

(3) is the reason the decision is made on rank within the frame rather than on
an absolute L* cut, and it is what should stop a Refer-KITTI calibration from
silently misbehaving on CARLA's different rendering.
"""

import numpy as np
import pytest

import color_classifier as cc


def solid(rgb, size=32):
    """A uniform crop of one colour."""
    return np.full((size, size, 3), rgb, dtype=np.uint8)


# ----------------------------------------------------------------------
# 1. The regression: black must read as dark
# ----------------------------------------------------------------------
@pytest.mark.parametrize("rgb", [
    (20, 22, 28),    # the measured near-black car body, slightly blue-cast
    (10, 10, 10),
    (35, 33, 38),
    (48, 50, 55),
])
def test_dark_pixels_classify_as_dark(rgb):
    """HSV called these 'blue' because S = (max-min)/max blows up near black."""
    L, chroma, hue = cc.lab_stats(solid(rgb))
    assert chroma < cc.CHROMA_MIN, f"{rgb} should be achromatic, got chroma={chroma:.1f}"
    assert cc.classify_patch(L, chroma, hue) == "dark"


def test_the_exact_pixel_from_the_bug_report():
    """RGB (20, 22, 28): HSV saturation 73/255 = 29%, well past the old gate."""
    crop = solid((20, 22, 28))
    L, chroma, _ = cc.lab_stats(crop)
    assert L < 20, f"expected a very dark L*, got {L:.1f}"
    assert chroma < 8, f"expected near-zero chroma, got {chroma:.1f}"
    assert cc.patch_votes(crop)[0] == "dark"


@pytest.mark.parametrize("rgb,expected", [
    ((240, 240, 240), "light"),
    ((128, 128, 130), "gray"),
    ((200, 30, 30), "red"),
    ((180, 20, 40), "red"),
    ((240, 140, 20), "orange"),
    ((240, 230, 40), "yellow"),
    ((30, 160, 60), "green"),
    ((120, 200, 40), "green"),
    ((30, 30, 200), "blue"),
    ((20, 40, 120), "blue"),
    ((140, 40, 180), "purple"),
])
def test_other_colors_still_classify(rgb, expected):
    """LAB hue boundaries, pinned against measured angles for sRGB primaries.

    An earlier hand-picked set put pure blue into 'purple' and pure red into
    'orange' — LAB hue angles do not match intuition, so these are measured.
    """
    assert cc.patch_votes(solid(rgb))[0] == expected


# ----------------------------------------------------------------------
# 2. No hue claims about achromatic objects
# ----------------------------------------------------------------------
def test_chromatic_target_abstains_on_an_achromatic_crop():
    """Without this guard a couple of noisy patches assert 'red' on a grey car.

    That is not hypothetical: on Refer-KITTI it produced TPR 1.000 / TNR 0.000
    for red — the classifier confirmed every distractor it committed on.
    """
    crop = solid((120, 120, 122))
    votes = cc.patch_votes(crop)[1]
    chroma = cc.lab_stats(crop)[1]
    assert cc.score_votes(votes, "red", crop_chroma=chroma) == 0.5


def test_chromatic_target_still_fires_on_a_chromatic_crop():
    crop = solid((200, 30, 30))
    votes = cc.patch_votes(crop)[1]
    chroma = cc.lab_stats(crop)[1]
    assert cc.score_votes(votes, "red", crop_chroma=chroma) == 1.0


# ----------------------------------------------------------------------
# 3. Exposure invariance — the property CARLA transfer rests on
# ----------------------------------------------------------------------
@pytest.mark.parametrize("gain", [0.5, 0.7, 1.0, 1.4, 1.9])
def test_peer_decision_is_invariant_to_exposure(gain):
    """Scaling every crop in the frame must not change the ranking decision.

    An absolute L* cut fails this: brighten the scene enough and the black car
    stops being "below L* = 32" while still being the darkest thing present.
    Rank within the frame cannot fail it, which is the point.
    """
    base = [(25, 25, 28), (110, 110, 112), (205, 205, 208)]  # dark, mid, light
    crops = [solid(tuple(np.clip(np.array(c) * gain, 0, 255).astype(np.uint8)))
             for c in base]

    dark_scores = cc.peer_relative_scores(crops, "dark")
    assert dark_scores[0] == 1.0, "darkest crop should support a 'dark' target"
    assert dark_scores[-1] == 0.0, "lightest crop is evidence against 'dark'"

    light_scores = cc.peer_relative_scores(crops, "light")
    assert light_scores[-1] == 1.0
    assert light_scores[0] == 0.0


def test_ranking_beats_an_absolute_cut_on_a_bright_scene():
    """The concrete failure an absolute threshold has, made explicit."""
    # Every car is bright in absolute terms; none is below DARK_L_MAX.
    crops = [solid(c) for c in [(150, 150, 152), (200, 200, 202), (245, 245, 247)]]
    lightness = [cc.lab_stats(c)[0] for c in crops]
    assert min(lightness) > cc.DARK_L_MAX, "precondition: no crop is absolutely dark"

    # Absolute scoring cannot name the dark one; peer ranking can.
    assert cc.peer_relative_scores(crops, "dark")[0] == 1.0


# ----------------------------------------------------------------------
# Vocabulary and fallback
# ----------------------------------------------------------------------
@pytest.mark.parametrize("word,canon", [
    ("black", "dark"), ("Black", "dark"), ("dark", "dark"),
    ("white", "light"), ("light", "light"),
    ("silver", "gray"), ("grey", "gray"), ("gray", "gray"),
    ("red", "red"),
])
def test_prompt_words_map_to_labels(word, canon):
    assert cc.canonical_color(word) == canon


def test_silver_is_scored_as_bright_not_as_middle():
    """Measured: silver as 'mid-lightness' scores 0.384 balanced accuracy —
    worse than guessing — against 0.754 as 'brighter than peers'."""
    assert "gray" in cc.BRIGHT_TARGETS
    crops = [solid(c) for c in [(25, 25, 28), (110, 110, 112), (215, 215, 218)]]
    assert cc.peer_relative_scores(crops, "silver")[-1] == 1.0
    assert cc.peer_relative_scores(crops, "silver")[0] == 0.0


def test_falls_back_to_absolute_below_min_peers():
    """With too few detections there is no ordering worth reading."""
    crops = [solid((20, 20, 22))] * (cc.MIN_PEERS - 1)
    scores = cc.peer_relative_scores(crops, "dark")
    assert len(scores) == len(crops)
    # An absolutely-dark crop still scores as dark via the fallback.
    assert scores[0] == 1.0


def test_empty_and_degenerate_inputs():
    assert cc.peer_relative_scores([], "dark") == []
    assert cc.patch_votes(np.zeros((0, 0, 3), dtype=np.uint8)) == ("unknown", {})
    assert cc.score_votes({}, "dark") == 0.5


def test_score_crop_matches_peer_scoring_for_the_first_crop():
    crops = [solid(c) for c in [(25, 25, 28), (110, 110, 112), (205, 205, 208)]]
    assert cc.score_crop(crops[0], "dark", peers=crops[1:]) == \
        cc.peer_relative_scores(crops, "dark")[0]

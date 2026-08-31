"""Contract tests for ``eval/motion_classifier.py``.

The thing this module exists to avoid is using image-space displacement as a
motion cue on a moving camera, where it measures ego-motion instead: on
Refer-KITTI that scores AUC 0.498, i.e. nothing. These tests pin the geometry
that replaces it, and — just as importantly — pin the places where it refuses
to answer, since the cue is weak enough that a confident wrong answer is worse
than an abstention.
"""

import numpy as np
import pytest

import motion_classifier as mc


class Track:
    """Minimal stand-in for an STrack."""

    __slots__ = ("track_id", "tlwh")

    def __init__(self, track_id, tlwh):
        self.track_id = track_id
        self.tlwh = tlwh


# ----------------------------------------------------------------------
# Vocabulary — what it will and will not claim to score
# ----------------------------------------------------------------------
@pytest.mark.parametrize("prompt,expected", [
    ("moving cars", "moving"),
    ("cars which are parking", "stationary"),
    ("parked vehicles", "stationary"),
    ("walking people", "moving"),
    ("standing men", "stationary"),
])
def test_scoreable_motion_words(prompt, expected):
    assert mc.canonical_motion(prompt) == expected


@pytest.mark.parametrize("prompt", [
    "cars which are braking",
    "turning vehicles",
    "cars in the counter direction of ours",
    "cars in the same direction of ours",
    "females back to the camera",
])
def test_unscoreable_motion_words_are_flagged_not_guessed(prompt):
    """These need a reference heading a monocular dashcam track cannot give.

    Flagged rather than silently treated as "no constraint", so the caller can
    tell "I cannot score this" apart from "there is nothing to score".
    """
    assert mc.canonical_motion(prompt) == "unscoreable"


@pytest.mark.parametrize("prompt", ["red cars", "cars in the left", "pedestrians"])
def test_non_motion_prompts_return_none(prompt):
    assert mc.canonical_motion(prompt) is None


def test_unscoreable_beats_scoreable_when_both_appear():
    """"cars which are braking" contains no scoreable word, but a prompt that
    mixes them must not be scored off the incidental one."""
    assert mc.canonical_motion("moving cars which are braking") == "unscoreable"


# ----------------------------------------------------------------------
# The geometry
# ----------------------------------------------------------------------
def _radial_field(points, s, foe):
    """Displacements a static world produces under forward camera motion."""
    return np.array([s * (p - foe) for p in points])


def test_fit_recovers_a_synthetic_ego_flow_field():
    points = np.array([[0.2, 0.8], [0.4, 0.7], [0.6, 0.75], [0.8, 0.9], [0.5, 0.6]])
    s_true, foe_true = 0.12, np.array([0.5, 0.45])
    disps = _radial_field(points, s_true, foe_true)

    s, c = mc.fit_ego_flow(points, disps)
    assert s == pytest.approx(s_true, abs=1e-6)
    assert (c / s) == pytest.approx(foe_true, abs=1e-6)


def test_static_world_gives_near_zero_residuals():
    points = np.array([[0.2, 0.8], [0.4, 0.7], [0.6, 0.75], [0.8, 0.9]])
    disps = _radial_field(points, 0.10, np.array([0.5, 0.5]))
    residuals = mc.ego_residuals(points, disps)
    assert np.allclose(residuals, 0.0, atol=1e-9)


def test_an_independently_moving_object_has_the_largest_residual():
    """The whole method in one assertion."""
    points = np.array([[0.2, 0.8], [0.4, 0.7], [0.6, 0.75], [0.8, 0.9], [0.5, 0.62]])
    disps = _radial_field(points, 0.10, np.array([0.5, 0.5]))
    disps[2] += np.array([0.09, -0.05])          # one car drives its own way

    residuals = mc.ego_residuals(points, disps)
    assert int(np.argmax(residuals)) == 2


def test_fit_declines_below_min_tracks():
    points = np.array([[0.3, 0.8], [0.6, 0.7]])
    assert mc.fit_ego_flow(points, np.zeros_like(points)) is None
    assert mc.ego_residuals(points, np.zeros_like(points)) is None


# ----------------------------------------------------------------------
# The scorer, over frames
# ----------------------------------------------------------------------
def _run(scorer, n_frames, positions_fn, img=(1242, 375)):
    """Feed synthetic tracks frame by frame; return the last frame's scores."""
    W, H = img
    scores = {}
    for f in range(n_frames):
        tracks = [Track(tid, tlwh) for tid, tlwh in positions_fn(f)]
        scores = scorer.score_frame(f, tracks, W, H)
    return scores


def test_scorer_abstains_until_it_has_history():
    scorer = mc.MotionScorer("moving")

    def positions(f):
        return [(i, (100 + 20 * i + 3 * f, 200, 40, 30)) for i in range(4)]

    first = scorer.score_frame(0, [Track(i, (100 + 20 * i, 200, 40, 30))
                                   for i in range(4)], 1242, 375)
    assert set(first.values()) == {0.5}, "no history yet -> must abstain"


def test_scorer_separates_a_mover_from_a_static_field():
    scorer = mc.MotionScorer("moving")
    W, H = 1242.0, 375.0
    foe = np.array([0.5, 0.5])
    s = 0.02
    base = [np.array([0.2, 0.80]), np.array([0.35, 0.72]),
            np.array([0.65, 0.74]), np.array([0.82, 0.86])]

    def positions(f):
        out = []
        for i, p0 in enumerate(base):
            p = p0 + f * s * (p0 - foe)      # static: rides the ego field
            if i == 2:
                p = p + np.array([f * 0.02, 0.0])   # this one also drives
            x = p[0] * W - 20
            y = p[1] * H - 30
            out.append((i, (x, y, 40.0, 30.0)))
        return out

    scores = _run(scorer, mc.WINDOW + 1, positions)
    assert scores[2] == 1.0, "the independently moving track should support 'moving'"
    assert 0.0 in scores.values(), "the best ego-field fits are evidence against"


def test_stationary_target_inverts_the_decision():
    scorer_move = mc.MotionScorer("moving")
    scorer_still = mc.MotionScorer("stationary")
    W, H = 1242.0, 375.0
    foe = np.array([0.5, 0.5])

    def positions(f):
        out = []
        for i, p0 in enumerate([np.array([0.2, 0.8]), np.array([0.35, 0.72]),
                                np.array([0.65, 0.74]), np.array([0.82, 0.86])]):
            p = p0 + f * 0.02 * (p0 - foe)
            if i == 2:
                p = p + np.array([f * 0.02, 0.0])
            out.append((i, (p[0] * W - 20, p[1] * H - 30, 40.0, 30.0)))
        return out

    moving = _run(scorer_move, mc.WINDOW + 1, positions)
    still = _run(scorer_still, mc.WINDOW + 1, positions)
    assert moving[2] == 1.0 and still[2] == 0.0


def test_unscoreable_target_never_commits():
    """A gate on an unscoreable prompt must be a no-op, not a filter."""
    scorer = mc.MotionScorer("unscoreable")

    def positions(f):
        return [(i, (100.0 + 20 * i + 5 * f, 200.0, 40.0, 30.0)) for i in range(5)]

    scores = _run(scorer, mc.WINDOW + 2, positions)
    assert set(scores.values()) == {0.5}


def test_too_few_tracks_abstains():
    scorer = mc.MotionScorer("moving")

    def positions(f):
        return [(0, (100.0 + 5 * f, 200.0, 40.0, 30.0))]

    scores = _run(scorer, mc.WINDOW + 2, positions)
    assert set(scores.values()) == {0.5}

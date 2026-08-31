#!/usr/bin/env python3
"""Motion state for referring expressions, compensated for a moving camera.

Refer-KITTI is shot from a driving car, and that breaks the obvious approach.
``SceneGraphBuilder._motion_attrs`` labels a track "moving" when its box travels
in the image, which is right for a static or slowly-panning camera and wrong
here: a *parked* car sweeps across a dashcam frame as you drive past it, while a
car matching your speed sits still in the image. Measured on Refer-KITTI's own
``moving-cars`` labels, over 8,357 track-frames:

    raw image displacement                  AUC 0.498   <- no signal whatsoever
    minus the frame's median displacement   AUC 0.569
    residual from a fitted radial ego-flow  AUC 0.645

0.498 is the number that matters: wiring the existing image-space motion label
into a gate would have produced a second silently-dead filter, exactly like the
HSV colour rule it would sit next to.

## The model

Under forward camera translation every **static** point flows radially away from
the focus of expansion (FOE), with magnitude falling off as 1/depth::

    d_i  =  s * (p_i - foe)

Both ``s`` and ``foe`` are unknown per frame, but the relation is linear in
``s`` and ``c = s * foe``, so one least-squares solve over all the frame's
tracks recovers the ego-flow field. An **independently moving** object is then
the one whose displacement does not fit::

    residual_i = || d_i - (s * p_i - c) ||

Fitting over the tracks themselves means the field is dominated by whatever the
majority are doing, which is the right prior on a road: most vehicles in view at
any moment are static relative to the world.

Ground-contact point (bottom-centre of the box) is used rather than box centre,
because it sits on the road plane and is far less sensitive to the box growing
as an object nears.

## What it is honest about

AUC 0.645 is weak — for comparison, in this same pipeline the spatial cue scores
0.95-0.97 and colour 0.71-0.86 — and that figure is measured on **ground-truth**
boxes. On real tracks, with fragmentation and ID switches, it is worse. So:

* the score is three-way (support / against / abstain) like the colour one, and
  the abstain band is deliberately wide;
* only ``moving`` and ``stationary`` are supported. ``braking``, ``turning`` and
  the ego-relative direction predicates ("counter direction of ours") need a
  reference heading that a monocular dashcam track does not give, and this
  module abstains on them rather than guessing;
* it is **off by default** in the pipeline. Turn it on with
  ``--use_motion_filter`` and check the effect with
  ``eval/check_motion_classifier.py`` before trusting it.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------
#: Prompt word -> motion state this module can actually score.
MOTION_SYNONYMS: Dict[str, str] = {
    "moving": "moving", "move": "moving", "driving": "moving",
    "running": "moving", "walking": "moving",
    "parking": "stationary", "parked": "stationary",
    "stopped": "stationary", "standing": "stationary",
    "stationary": "stationary", "still": "stationary",
}

#: Recognised as motion words but *not* scoreable from a monocular dashcam
#: track. Listed so the parser can say "this is a motion prompt I must abstain
#: on" rather than silently treating it as no constraint at all.
UNSCOREABLE_MOTION = (
    "braking", "brake", "turning", "turn",
    "counter direction", "counter-direction",
    "same direction", "same-direction",
    "back to the camera", "faster", "slower",
)

#: Frames of history used to measure displacement. Long enough that a slow
#: object separates from noise, short enough to survive occlusions.
WINDOW = 5

#: Minimum tracks needed to fit the ego-flow field. The fit has 3 unknowns
#: (s, cx, cy) and each track gives 2 equations, so 3 tracks is the practical
#: floor; below it there is no ego estimate and the module abstains.
MIN_TRACKS_FOR_FIT = 3

#: Residual rank bands within a frame, mirroring the colour classifier: the
#: top fraction by residual supports "moving", the bottom fraction supports
#: "stationary", and the middle abstains. Rank rather than an absolute residual
#: because the ego-flow magnitude scales with vehicle speed.
RESIDUAL_FRACTION = 1.0 / 3.0


def canonical_motion(text: Optional[str]) -> Optional[str]:
    """Motion state named by a prompt, or None.

    Returns ``'moving'`` / ``'stationary'`` for the states this module can
    score, ``'unscoreable'`` for a motion word it deliberately will not guess
    at, and ``None`` when the prompt names no motion at all.
    """
    if not text:
        return None
    low = str(text).lower()
    for phrase in UNSCOREABLE_MOTION:
        if phrase in low:
            return "unscoreable"
    for word, state in MOTION_SYNONYMS.items():
        if word in low:
            return state
    return None


# ----------------------------------------------------------------------
# Ego-flow fit
# ----------------------------------------------------------------------
def fit_ego_flow(points: np.ndarray, displacements: np.ndarray
                 ) -> Optional[Tuple[float, np.ndarray]]:
    """Least-squares fit of ``d = s * (p - foe)`` over a frame's tracks.

    Returns ``(s, c)`` where ``c = s * foe``, or None if the fit is degenerate.
    Solving for ``c`` rather than ``foe`` keeps the system linear.
    """
    n = len(points)
    if n < MIN_TRACKS_FOR_FIT:
        return None

    A = np.zeros((2 * n, 3), dtype=np.float64)
    b = np.zeros(2 * n, dtype=np.float64)
    for i, (p, d) in enumerate(zip(points, displacements)):
        A[2 * i] = (p[0], -1.0, 0.0)
        b[2 * i] = d[0]
        A[2 * i + 1] = (p[1], 0.0, -1.0)
        b[2 * i + 1] = d[1]

    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(sol)):
        return None
    return float(sol[0]), np.array([sol[1], sol[2]], dtype=np.float64)


def ego_residuals(points: np.ndarray, displacements: np.ndarray
                  ) -> Optional[np.ndarray]:
    """Per-track distance between observed displacement and fitted ego-flow."""
    fit = fit_ego_flow(points, displacements)
    if fit is None:
        return None
    s, c = fit
    predicted = s * points - c
    return np.linalg.norm(displacements - predicted, axis=1)


# ----------------------------------------------------------------------
# Stateful scorer
# ----------------------------------------------------------------------
class MotionScorer:
    """Tracks ground-contact history and scores motion state per frame.

    Usage mirrors the colour path: feed it every track in the frame, get back a
    three-way score per track id (1.0 support / 0.0 against / 0.5 abstain).
    """

    def __init__(self, target_state: Optional[str], window: int = WINDOW):
        self.target_state = target_state
        self.window = int(window)
        # track_id -> deque of (frame_id, x_norm, y_bottom_norm)
        self._history: Dict[int, Deque[Tuple[int, float, float]]] = {}

    def reset(self) -> None:
        self._history.clear()

    def _observe(self, frame_id: int, track_id: int, tlwh, img_w: int, img_h: int):
        x, y, w, h = tlwh
        hist = self._history.setdefault(track_id, deque(maxlen=self.window + 1))
        # Ground-contact point: bottom-centre sits on the road plane, so it is
        # far less affected by the box growing as the object approaches.
        hist.append((int(frame_id), (x + w / 2.0) / img_w, (y + h) / img_h))

    def _displacement(self, track_id: int, frame_id: int):
        """(point_now, displacement_over_window) or None if history is short."""
        hist = self._history.get(track_id)
        if not hist or len(hist) < 2:
            return None
        newest = hist[-1]
        if newest[0] != frame_id:
            return None
        oldest = hist[0]
        if newest[0] - oldest[0] < max(2, self.window // 2):
            return None
        return (np.array([newest[1], newest[2]]),
                np.array([newest[1] - oldest[1], newest[2] - oldest[2]]))

    def score_frame(self, frame_id: int, tracks: Sequence, img_w: int,
                    img_h: int) -> Dict[int, float]:
        """Three-way motion score for every track in this frame."""
        for t in tracks:
            self._observe(frame_id, t.track_id, t.tlwh, img_w, img_h)

        if self.target_state not in ("moving", "stationary"):
            # No scoreable constraint — never delete anything on a guess.
            return {t.track_id: 0.5 for t in tracks}

        ids, pts, disps = [], [], []
        for t in tracks:
            got = self._displacement(t.track_id, frame_id)
            if got is not None:
                ids.append(t.track_id)
                pts.append(got[0])
                disps.append(got[1])

        scores = {t.track_id: 0.5 for t in tracks}
        if len(ids) < MIN_TRACKS_FOR_FIT:
            return scores

        residuals = ego_residuals(np.array(pts), np.array(disps))
        if residuals is None:
            return scores

        n = len(residuals)
        quantile = (np.argsort(np.argsort(residuals)) + 0.5) / n
        for tid, q in zip(ids, quantile):
            high = q >= 1.0 - RESIDUAL_FRACTION   # least like the ego field
            low = q <= RESIDUAL_FRACTION          # fits the ego field best
            if self.target_state == "moving":
                scores[tid] = 1.0 if high else (0.0 if low else 0.5)
            else:  # 'stationary'
                scores[tid] = 1.0 if low else (0.0 if high else 0.5)
        return scores

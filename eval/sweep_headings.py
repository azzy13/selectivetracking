#!/usr/bin/env python3
"""Ground-truth headings for the single-frame sweep clips.

Why this exists
---------------
``SceneGraphBuilder._depth_relation`` computes ``behind`` / ``in-front-of`` in
the reference object's own frame when that object has a heading, and falls back
to viewer-centric image depth when it does not.  The object-centric branch is
the one the benchmark prompts mean ("the car behind the black car *moving down
F St.*"), but it needs a heading, and ``_motion_attrs`` derives headings from
three frames of track history.

A sweep clip is **one** frame.  There is no history, so every heading is
``[0, 0]`` and only the fallback ever runs — which on the sweep picks the wrong
car, because the distractor is further from the camera than the bus while the
target is nearer.

The sweep GT already carries what is missing: every ``gt_graphs`` node has a
world-space ``yaw``.  This module projects that heading into image space so it
can be handed to ``SceneGraphBuilder.update(heading_override=...)``.

What it is and is not
---------------------
This is a **harness**, not a pipeline feature.  It measures whether the relation
logic is right given a correct heading.  It says nothing about how well headings
are recovered from tracker output, which is what a live run depends on — there,
headings come from motion and no ground truth exists to substitute.  Treat the
numbers it produces as an upper bound.

Usage:
    from sweep_headings import gt_headings, headings_for_tracks

    headings = gt_headings(clip)                      # {gt role: [dx, dy]}
    override = headings_for_tracks(tracks, clip, H, W)  # {track_id: [dx, dy]}
    graph = builder.update(0, tracks, H, W, roles=roles, heading_override=override)
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Metres to step along an object's heading before re-projecting.  Only the
#: direction of the projected step is used, but the step must be long enough
#: that pixel rounding does not dominate it and short enough that perspective
#: does not bend it noticeably.
HEADING_STEP_M = 2.0


# ---------------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------------

def rotation_matrix(rotation: Dict[str, float]) -> np.ndarray:
    """World->camera-axes rotation for a CARLA camera transform.

    CARLA inherits Unreal's convention: x forward, y right, z up, with the
    transform composed as yaw about z, then pitch about y, then roll about x.
    Columns are the camera's forward / right / up axes in world coordinates, so
    the transpose maps a world offset into the camera frame.
    """
    pitch = math.radians(rotation["pitch"])
    yaw   = math.radians(rotation["yaw"])
    roll  = math.radians(rotation["roll"])
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp,                     -cp * sr,                cp * cr],
    ])


def project(point: Sequence[float], camera: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Project a world point to pixel coordinates, or None if behind the camera.

    Pinhole with a square pixel and a centred principal point, which is what
    CARLA's RGB sensor produces: ``f = W / (2 tan(fov/2))``.
    """
    width, height = camera["image_size"]
    focal = width / (2.0 * math.tan(math.radians(camera["fov"]) / 2.0))
    offset = np.asarray(point, dtype=float) - np.asarray(camera["location"], dtype=float)
    forward, right, up = rotation_matrix(camera["rotation"]).T @ offset
    if forward <= 0.01:
        return None
    return (width / 2.0 + focal * right / forward,
            height / 2.0 - focal * up / forward)


def heading_vec(loc: Sequence[float], yaw_deg: float,
                camera: Dict[str, Any]) -> Optional[List[float]]:
    """Unit heading in normalised image coords (right=+x, down=+y), or None.

    Matches ``SceneGraphBuilder._motion_attrs``: the same units and the same
    sign convention, so an override is interchangeable with a measured heading.
    Normalising by width and height separately is what makes it comparable to
    ``cx_norm``/``cy_norm``, which the depth relation is computed from.
    """
    yaw = math.radians(yaw_deg)
    tip = [loc[0] + HEADING_STEP_M * math.cos(yaw),
           loc[1] + HEADING_STEP_M * math.sin(yaw),
           loc[2]]
    base_uv, tip_uv = project(loc, camera), project(tip, camera)
    if base_uv is None or tip_uv is None:
        return None

    width, height = camera["image_size"]
    dx = (tip_uv[0] - base_uv[0]) / width
    dy = (tip_uv[1] - base_uv[1]) / height
    norm = math.hypot(dx, dy)
    if norm == 0:
        return None
    return [dx / norm, dy / norm]


# ---------------------------------------------------------------------------
# Sweep GT
# ---------------------------------------------------------------------------

def load_gt(clip: Dict[str, Any], sweep_dir: str) -> Dict[str, Dict[str, Any]]:
    """GT nodes for one clip, keyed by sweep role: 'bus' / 'target' / 'distractor'."""
    path = os.path.join(sweep_dir, clip["config_id"], clip["condition"],
                        "gt_graphs", "000000.json")
    with open(path) as fh:
        gt = json.load(fh)
    role_of = {obj_id: role for role, obj_id in gt["ids"].items()}
    return {role_of[n["id"]]: n for n in gt["nodes"]}


def gt_headings(clip: Dict[str, Any], sweep_dir: str = "dataset/sweep") -> Dict[str, List[float]]:
    """``{sweep role: [dx, dy]}`` — the true image-space heading of each GT object."""
    camera = clip["camera"]
    out = {}
    for role, node in load_gt(clip, sweep_dir).items():
        vec = heading_vec(node["loc"], node["yaw"], camera)
        if vec is not None:
            out[role] = vec
    return out


def _iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def headings_for_tracks(
    tracks: Sequence[Any],
    clip: Dict[str, Any],
    sweep_dir: str = "dataset/sweep",
    *,
    min_iou: float = 0.4,
) -> Dict[int, List[float]]:
    """``{track_id: [dx, dy]}`` ready for ``update(heading_override=...)``.

    Tracks are matched to GT objects by best IoU.  A track that matches nothing
    above ``min_iou`` gets no entry rather than a guessed heading — a wrong
    heading flips the sign of the depth relation, which is worse than the
    viewer-centric fallback the builder uses when a heading is absent.
    """
    gt = load_gt(clip, sweep_dir)
    headings = gt_headings(clip, sweep_dir)

    override: Dict[int, List[float]] = {}
    for track in tracks:
        x, y, w, h = track.tlwh
        box = [x, y, x + w, y + h]
        best_role, best_iou = None, 0.0
        for role, node in gt.items():
            iou = _iou_xyxy(box, node["box2d"])
            if iou > best_iou:
                best_role, best_iou = role, iou
        if best_role is not None and best_iou >= min_iou and best_role in headings:
            override[int(track.track_id)] = headings[best_role]
    return override

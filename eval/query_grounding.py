#!/usr/bin/env python3
"""
Wires the parsed ``Query`` into detection and scene-graph construction.

This is the Week 3 structural layer.  It sits between ``query_parser`` (text ->
``Query``) and ``scene_graph`` (tracks -> graph), and it exists to fix one
thing: the anchor of a relational prompt was never detected, so it was never a
node, so relational grounding was impossible.

Old order — anchor can never enter the graph:

    detect("red car")  ->  hard filter  ->  graph over survivors  ->  output

New order — the graph is built over everything that was detected:

    detect("red car . bus")  ->  graph over ALL candidates  ->  score  ->  output

Three pieces, matching the three steps:

  1. ``build_detector_prompt``  — target *and* anchor classes go into the caption.
  2. ``assign_detection_roles`` / ``assign_track_roles`` — every detection and
     every track is tagged ``'target_candidate'`` or ``'anchor'``.  The role
     rides through to the graph node so the scorer knows what it is looking at.
  3. ``score_candidates`` — the soft subgraph scorer that replaces the hard
     filter.  STUBBED this week (see its docstring); the point of Week 3 is that
     the anchor is in the graph and scoring runs over all candidates.

Role contract — anchors are scaffolding, never results
------------------------------------------------------
An anchor exists ONLY so the graph has something to relate a target candidate
to.  It is never emitted: no MOT line, no track ID in the output, no presence in
results or metrics.  ``emitted_tracks`` is the single choke point for that rule —
everything that writes output goes through it.  The one exception is debug
rendering, where anchors may be drawn with a DOTTED outline in a distinct colour
(``draw_dotted_rect``), off by default.

Usage:
    from query_parser import parse
    from query_grounding import (
        build_detector_prompt, assign_detection_roles, assign_track_roles,
        score_candidates, emitted_tracks,
    )

    query  = parse("red car behind the bus")
    prompt = build_detector_prompt(query)          # "red car . bus"
    ...
    roles   = assign_track_roles(tracks, dets, det_roles)
    graph   = sg.update(frame_id, tracks, H, W, frame_bgr=img, roles=roles)
    weights = score_candidates(graph, query)       # {track_id: weight}
    for t in emitted_tracks(tracks, roles):        # anchors already dropped
        ...
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Node / detection roles
# ---------------------------------------------------------------------------

#: A detection that may be the thing the prompt asks for.  Only these are ever
#: emitted as output tracks.
ROLE_TARGET = "target_candidate"

#: A detection of the anchor class ("the bus" in "red car behind the bus").
#: Graph scaffolding only — never emitted, never given a track ID in the output.
ROLE_ANCHOR = "anchor"

ROLES = (ROLE_TARGET, ROLE_ANCHOR)

#: Debug-render colour for anchor boxes (BGR).  Distinct from the solid green
#: used for target candidates, and always drawn dotted — see ``draw_dotted_rect``.
ANCHOR_DEBUG_COLOR = (0, 165, 255)   # orange
TARGET_DEBUG_COLOR = (0, 255, 0)     # green, solid — matches existing rendering


# ---------------------------------------------------------------------------
# STEP 1 — parser -> detector prompt
# ---------------------------------------------------------------------------

def entity_phrase(entity: Optional[Dict[str, Any]]) -> str:
    """Render a parsed entity as the caption phrase for the detector.

    ``{'class': 'car', 'attrs': {'color': 'red'}}`` -> ``"red car"``.

    Only attributes the detector can actually ground are included: colour, size
    and free-form adjectives.  ``count`` is a query-level constraint, not a
    visual one, so it never reaches the caption.
    """
    if not entity:
        return ""
    attrs = entity.get("attrs") or {}
    words: List[str] = []
    for key in ("color", "size"):
        if attrs.get(key):
            words.append(str(attrs[key]))
    words.extend(str(w) for w in attrs.get("other", []))
    words.append(str(entity.get("class", "")).strip())
    return " ".join(w for w in words if w).strip()


def build_detector_prompt(query, *, dotted: bool = False) -> str:
    """STEP 1: build the detector caption from BOTH target and anchor classes.

    ``target={red car}, anchor={bus}``  ->  ``"red car . bus"``

    A plain prompt (``anchor=None`` — no relation, unary relation, or an
    ego-anchored one) stays target-only and is byte-identical to what the
    detector was prompted with before this change.

    Args:
        query:  a ``query_parser.Query``.
        dotted: append a trailing "." (GroundingDINO's caption convention; some
                call sites add it themselves, so it is off by default).

    Returns:
        The caption string.  Classes are separated by " . " because that is the
        separator GroundingDINO segments phrases on, which is what makes
        per-detection role assignment possible downstream.
    """
    target = entity_phrase(query.target)
    if not target:
        raise ValueError(f"query has no usable target class: {query.target!r}")

    anchor = entity_phrase(query.anchor)
    # Same class on both sides ("car behind the car") — one caption class is
    # enough, and duplicating it would make the phrases ambiguous.
    parts = [target] if not anchor or anchor == target else [target, anchor]

    prompt = " . ".join(parts)
    return prompt + " ." if dotted else prompt


def caption_classes(query) -> List[str]:
    """The caption's classes in prompt order, as ``build_detector_prompt`` emits them."""
    prompt = build_detector_prompt(query)
    return [p.strip() for p in prompt.split(".") if p.strip()]


# ---------------------------------------------------------------------------
# STEP 2a — detection roles
# ---------------------------------------------------------------------------

def _token_set(phrase: str) -> set:
    return {t for t in phrase.lower().replace(".", " ").split() if t}


def assign_detection_roles(phrases: Sequence[str], query) -> List[str]:
    """Tag each detection with its role from the phrase GroundingDINO grounded it to.

    GroundingDINO returns, per box, the caption span the box's logits peak on.
    With a two-class caption ("red car . bus") that span is what tells us whether
    the box is a target candidate or the anchor.

    Scoring is by token overlap against the target phrase and the anchor phrase;
    the larger overlap wins.

    The tie-break is deliberately asymmetric: anything ambiguous becomes a
    ``target_candidate``.  Mislabelling an anchor as a candidate costs a possible
    false positive that the scorer can down-weight, while mislabelling a target
    as an anchor would silently delete a real result — anchors are never emitted.

    Args:
        phrases: one grounded phrase per detection, aligned with the detection rows.
        query:   the ``Query`` the caption was built from.

    Returns:
        A list of role strings aligned with ``phrases``.
    """
    if query.anchor is None:
        # Plain / unary / ego-anchored: every detection is a target candidate.
        return [ROLE_TARGET] * len(phrases)

    target_tokens = _token_set(entity_phrase(query.target))
    anchor_tokens = _token_set(entity_phrase(query.anchor))
    # Tokens shared by both phrases ("car" in "red car" / "white car") carry no
    # role signal — only the distinguishing tokens vote.
    shared = target_tokens & anchor_tokens
    target_only = target_tokens - shared
    anchor_only = anchor_tokens - shared

    roles: List[str] = []
    for phrase in phrases:
        tokens = _token_set(phrase)
        t_hits = len(tokens & target_only)
        a_hits = len(tokens & anchor_only)
        roles.append(ROLE_ANCHOR if a_hits > t_hits else ROLE_TARGET)
    return roles


# ---------------------------------------------------------------------------
# STEP 2b — track roles
# ---------------------------------------------------------------------------

def _iou_xyxy_vs_tlwh(box_xyxy: Sequence[float], tlwh: Sequence[float]) -> float:
    x1, y1, x2, y2 = box_xyxy[:4]
    tx, ty, tw, th = tlwh
    tx2, ty2 = tx + tw, ty + th
    ix1, iy1 = max(x1, tx), max(y1, ty)
    ix2, iy2 = min(x2, tx2), min(y2, ty2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, x2 - x1) * max(0.0, y2 - y1) + tw * th - inter
    return inter / union if union > 0 else 0.0


def assign_track_roles(
    tracks: Sequence,
    dets_xyxy: np.ndarray,
    det_roles: Sequence[str],
    *,
    iou_thresh: float = 0.30,
    sticky: Optional[Dict[int, Dict[str, int]]] = None,
) -> Dict[int, str]:
    """Carry detection roles through the tracker onto track IDs.

    The tracker knows nothing about roles, so each output track is matched back
    to this frame's detections by best IoU and votes for that detection's role.

    A track's role is decided by its accumulated votes, not by this frame alone.
    Per-frame assignment flaps — the detector grounds the same object to "red
    car" in one frame and "white car" in the next — and a flapping role means an
    anchor leaks into the output on the frames it happens to be tagged a
    candidate.  A tracked object's role is a property of the object, so it is
    resolved over the track's whole life:

      * matched this frame  -> that role gets a vote; the majority role wins
      * no match this frame -> the standing majority carries over unchanged
      * tie                 -> keep the role the track already had (hysteresis)
      * no votes at all     -> ``target_candidate``, as in ``assign_detection_roles``

    Args:
        tracks:     tracker output for this frame (objects with ``.tlwh``, ``.track_id``).
        dets_xyxy:  the (N, 5) detection array the tracker was updated with.
        det_roles:  roles aligned with ``dets_xyxy`` rows.
        iou_thresh: minimum IoU for a track to inherit a detection's role.
        sticky:     role-evidence memory across frames — pass the same dict every
                    frame and it is updated in place.  ``{track_id: {role: votes}}``.

    Returns:
        ``{track_id: role}`` for the tracks in this frame.
    """
    memory = sticky if sticky is not None else {}
    roles: Dict[int, str] = {}

    have_dets = dets_xyxy is not None and len(dets_xyxy) > 0 and len(det_roles) > 0

    for track in tracks:
        tid = int(track.track_id)
        state = memory.setdefault(tid, {"votes": {}, "role": None})
        votes = state["votes"]

        best_iou, best_role = 0.0, None
        if have_dets:
            for det, role in zip(dets_xyxy, det_roles):
                iou = _iou_xyxy_vs_tlwh(det, track.tlwh)
                if iou > best_iou:
                    best_iou, best_role = iou, role
        if best_role is not None and best_iou >= iou_thresh:
            votes[best_role] = votes.get(best_role, 0) + 1

        if not votes:
            roles[tid] = ROLE_TARGET
            continue
        top = max(votes.values())
        winners = [r for r, v in votes.items() if v == top]
        previous = state["role"]
        roles[tid] = (previous if previous in winners else
                      (ROLE_TARGET if ROLE_TARGET in winners else winners[0]))
        state["role"] = roles[tid]

    return roles


def emitted_tracks(tracks: Sequence, roles: Optional[Dict[int, str]]) -> List:
    """The tracks that may leave the pipeline — target candidates only.

    Every output path (MOT lines, metrics, results) goes through this function.
    Anchors are dropped here and nowhere else, so the "anchors are never emitted"
    rule has exactly one place it can be broken.

    ``roles=None`` means no query grounding is active, so everything is a target
    candidate and the list is returned unchanged.
    """
    if not roles:
        return list(tracks)
    return [t for t in tracks if roles.get(int(t.track_id), ROLE_TARGET) == ROLE_TARGET]


def anchor_tracks(tracks: Sequence, roles: Optional[Dict[int, str]]) -> List:
    """The anchor tracks — debug rendering only, never output."""
    if not roles:
        return []
    return [t for t in tracks if roles.get(int(t.track_id), ROLE_TARGET) == ROLE_ANCHOR]


# ---------------------------------------------------------------------------
# STEP 3 — soft subgraph scorer  (STUB — Week 4 finishes this)
# ---------------------------------------------------------------------------

def score_candidates(
    frame_graph: Dict[str, Any],
    query,
    *,
    relation_weight: Optional[float] = None,
) -> Dict[int, float]:
    """Score every target candidate by how well it fits the query subgraph.

    This replaces the hard filter.  Nothing is dropped: every target candidate in
    the graph gets a weight, and the caller decides what to do with it.

    The subgraph being matched is the one the parser emits:

        (target)-[relation]->(anchor)

    against the scene graph's nodes and edges for this frame.

    STATUS: STUB.  The appearance term is the detector confidence and the
    relation term is hard-zero, so today this returns detector confidence for
    every candidate.  The structure — not the number — is the Week 3 deliverable:
    the anchor is in the graph, and scoring runs over all candidates rather than
    over post-filter survivors.

    TODO(Week 4): implement ``_relation_term``.  For each candidate node, find
    the anchor nodes in ``frame_graph['nodes']`` with ``role == ROLE_ANCHOR``,
    look up the edge between them in ``frame_graph['edges']``, and turn the
    edge's relation labels into a match score for ``query.relation['name']``.
    The scene-graph edge labels are hyphenated ('left-of') while the query
    vocabulary is snake_case ('left_of'); ``RelationSpec.scene_graph_hint``
    records the mapping where one exists.  Ego-relative relations
    (``counter_direction``, ``same_direction``) and any relation flagged in
    ``query.notes`` must stay at 0.0 — they cannot be scored without a reference
    heading the GT does not carry.

    Args:
        frame_graph:     a frame dict from ``SceneGraphBuilder.update()``.
        query:           the ``Query`` this frame was detected for.
        relation_weight: override for ``query.relation['weight']``.  The soft
                         weight is a score multiplier, never a filter.

    Returns:
        ``{track_id: weight}`` for target candidates only.  Anchors are not
        scored — they are not results.
    """
    weight = relation_weight
    if weight is None:
        weight = query.relation["weight"] if query.relation else 0.0

    scores: Dict[int, float] = {}
    for node in frame_graph.get("nodes", []):
        if node.get("role", ROLE_TARGET) != ROLE_TARGET:
            continue

        appearance_term = float(node.get("confidence", 0.0))
        relation_term = _relation_term(node, frame_graph, query)

        # This is where the parsed relation weight plugs in.  Week 4 turns
        # relation_term into a real [0, 1] match score; until then it is 0.0 and
        # the product contributes nothing.
        scores[int(node["track_id"])] = appearance_term + weight * relation_term

    return scores


def _relation_term(node: Dict[str, Any], frame_graph: Dict[str, Any], query) -> float:
    """How well this candidate's edge to the anchor matches ``query.relation``.

    STUB — always 0.0.  See the TODO in ``score_candidates`` for what Week 4
    fills in here.  Signature is fixed now so the call site does not move.
    """
    return 0.0


# ---------------------------------------------------------------------------
# Debug rendering
# ---------------------------------------------------------------------------

def draw_dotted_rect(
    img,
    pt1,
    pt2,
    color=ANCHOR_DEBUG_COLOR,
    thickness: int = 2,
    dot_len: int = 6,
    gap: int = 6,
) -> None:
    """Draw a dotted rectangle in place — the anchor convention for debug frames.

    OpenCV has no dashed-rectangle primitive, so each side is stepped manually.
    Anchors are dotted and target candidates solid so a debug frame reads as
    "scaffolding vs result" without a legend.
    """
    import cv2

    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    step = max(1, dot_len + gap)

    for x in range(x1, x2, step):
        xe = min(x + dot_len, x2)
        cv2.line(img, (x, y1), (xe, y1), color, thickness)
        cv2.line(img, (x, y2), (xe, y2), color, thickness)
    for y in range(y1, y2, step):
        ye = min(y + dot_len, y2)
        cv2.line(img, (x1, y), (x1, ye), color, thickness)
        cv2.line(img, (x2, y), (x2, ye), color, thickness)


def describe_grounding(query) -> str:
    """One-line summary of what grounding will do, for startup logging."""
    prompt = build_detector_prompt(query)
    if query.anchor is None:
        return f"prompt='{prompt}' | target-only (plain MOT path unchanged)"
    rel = query.relation["name"] if query.relation else "?"
    return (
        f"prompt='{prompt}' | target='{entity_phrase(query.target)}' "
        f"anchor='{entity_phrase(query.anchor)}' relation='{rel}' "
        f"(anchor detected for the graph, never emitted)"
    )

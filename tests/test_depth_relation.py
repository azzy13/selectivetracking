"""Depth relations: the 'behind' / 'in-front-of' edge and the scorer that reads it.

The point of the relation term is that detector confidence alone ranks these
backwards — the car behind the bus is the further, smaller, lower-confidence
box — so every test here checks a *ranking*, not just a label.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))

from query_grounding import (  # noqa: E402
    _EDGE_FOR_RELATION,
    _EDGE_INVERSE,
    ROLE_ANCHOR,
    ROLE_TARGET,
    relation_holds,
    score_candidates,
    select_answers,
    _relation_term,
)
from scene_graph import SceneGraphBuilder  # noqa: E402


# --------------------------------------------------------------------------
# helpers — build graph fragments directly, no detector or tracker involved
# --------------------------------------------------------------------------

def node(track_id, cx, cy, *, role=ROLE_TARGET, heading=(0.0, 0.0), conf=0.5, area=0.01):
    return {
        "track_id": track_id, "cx_norm": cx, "cy_norm": cy, "area_norm": area,
        "bbox_tlwh": [cx * 100, cy * 100, 10, 10], "role": role,
        "heading_vec": list(heading), "confidence": conf,
    }


class FakeTrack:
    """Minimal STrack stand-in: the builder only reads these attributes."""

    def __init__(self, track_id, cx, cy, size=10.0, score=0.5):
        self.track_id = track_id
        self.tlwh = [cx * 100 - size / 2, cy * 100 - size / 2, size, size]
        self.score = score
        self.embedding = None


def rels(n1, n2):
    b = SceneGraphBuilder()
    return b._compute_relations(n1, n2)


class FakeQuery:
    def __init__(self, name, anchor={"class": "bus"}, weight=1.0):
        self.relation = {"name": name, "arity": "binary", "temporal": False,
                         "weight": weight} if name else None
        self.anchor = anchor
        self.target = {"class": "car"}


def graph(nodes, edges):
    return {"nodes": nodes, "edges": edges}


# --------------------------------------------------------------------------
# object-centric: the anchor's own heading decides, not camera depth
# --------------------------------------------------------------------------

def test_behind_is_measured_in_the_anchors_frame():
    # Bus heading up-screen (away from camera). The car LOWER on screen is
    # nearer the camera, but it trails the bus, so it is 'behind' the bus.
    bus = node(1, 0.5, 0.5, role=ROLE_ANCHOR, heading=(0.0, -1.0))
    trailing = node(2, 0.5, 0.8)
    assert "behind" in rels(trailing, bus)


def test_leading_car_is_in_front_even_though_it_is_further_away():
    bus = node(1, 0.5, 0.5, role=ROLE_ANCHOR, heading=(0.0, -1.0))
    leading = node(2, 0.5, 0.2)          # higher on screen => further from camera
    assert "in-front-of" in rels(leading, bus)


def test_the_two_frames_genuinely_disagree():
    # Same geometry read both ways: this is the sweep's confound in miniature.
    bus_moving = node(1, 0.5, 0.5, role=ROLE_ANCHOR, heading=(0.0, -1.0))
    bus_parked = node(1, 0.5, 0.5, role=ROLE_ANCHOR, heading=(0.0, 0.0))
    car = node(2, 0.5, 0.2)
    assert "in-front-of" in rels(car, bus_moving)   # ahead along the bus's path
    assert "behind" in rels(car, bus_parked)        # further from the camera


def test_lateral_heading_uses_the_lateral_axis():
    bus = node(1, 0.5, 0.5, role=ROLE_ANCHOR, heading=(1.0, 0.0))   # driving right
    assert "in-front-of" in rels(node(2, 0.9, 0.5), bus)
    assert "behind" in rels(node(2, 0.1, 0.5), bus)


def test_too_close_to_call_emits_no_depth_label():
    bus = node(1, 0.5, 0.5, role=ROLE_ANCHOR, heading=(0.0, -1.0))
    same = node(2, 0.5, 0.505)
    assert "behind" not in rels(same, bus)
    assert "in-front-of" not in rels(same, bus)


# --------------------------------------------------------------------------
# viewer-centric fallback: no heading available (stationary / single frame)
# --------------------------------------------------------------------------

def test_fallback_orders_by_vertical_position():
    parked = node(1, 0.5, 0.5, role=ROLE_ANCHOR)
    assert "behind" in rels(node(2, 0.5, 0.2), parked)
    assert "in-front-of" in rels(node(2, 0.5, 0.8), parked)


def test_fallback_ignores_area_across_classes():
    # A car in front of a bus is still the smaller box; size must not veto it.
    bus = node(1, 0.5, 0.4, role=ROLE_ANCHOR, area=0.20)
    car = node(2, 0.5, 0.9, area=0.005)
    assert "in-front-of" in rels(car, bus)


# --------------------------------------------------------------------------
# the scorer
# --------------------------------------------------------------------------

def test_depth_labels_are_dropped_not_inverted_when_read_backwards():
    # Depth is measured in the reference node's own frame, so "the bus is in
    # front of the car (in the CAR's frame)" says nothing about where the car is
    # relative to the bus. Inverting it here is what made the sweep pick the
    # oncoming distractor, whose frame points the other way.
    bus, car = node(1, 0.5, 0.5, role=ROLE_ANCHOR), node(2, 0.5, 0.8)
    g = graph([bus, car], [{"source": 1, "target": 2, "relations": ["in-front-of"]}])
    assert _relation_term(car, g, FakeQuery("behind")) == 0.0


def test_viewer_frame_labels_are_still_inverted_when_read_backwards():
    # left-of/right-of are computed in the image frame, which is shared, so
    # swapping subject and object really is just a relabel.
    bus, car = node(1, 0.5, 0.5, role=ROLE_ANCHOR), node(2, 0.2, 0.5)
    g = graph([bus, car], [{"source": 1, "target": 2, "relations": ["right-of"]}])
    assert _relation_term(car, g, FakeQuery("left_of")) == 1.0


def test_builder_orients_anchor_edges_candidate_to_anchor():
    # The reordering is what keeps the drop above from losing real information:
    # an anchor-spanning edge is always stored with the anchor second, so its
    # depth label is already in the anchor's frame.
    b = SceneGraphBuilder()
    anchor = FakeTrack(1, 0.5, 0.5)
    cand = FakeTrack(2, 0.5, 0.8)
    g = b.update(0, [anchor, cand], 100, 100,
                 roles={1: ROLE_ANCHOR, 2: ROLE_TARGET},
                 heading_override={1: [0.0, -1.0]})
    spanning = [e for e in g["edges"] if 1 in (e["source"], e["target"])]
    assert spanning, "anchor edges are always emitted"
    for e in spanning:
        assert e["target"] == 1, "the anchor must be the reference node"
    assert "behind" in spanning[0]["relations"]


def test_relation_term_matches_edge_stored_candidate_first():
    bus, car = node(1, 0.5, 0.5, role=ROLE_ANCHOR), node(2, 0.5, 0.8)
    g = graph([bus, car], [{"source": 2, "target": 1, "relations": ["behind"]}])
    assert _relation_term(car, g, FakeQuery("behind")) == 1.0


def test_relation_term_ignores_edges_between_two_candidates():
    a, b = node(2, 0.5, 0.8), node(3, 0.5, 0.2)
    bus = node(1, 0.5, 0.5, role=ROLE_ANCHOR)
    g = graph([bus, a, b], [{"source": 2, "target": 3, "relations": ["behind"]}])
    assert _relation_term(a, g, FakeQuery("behind")) == 0.0


def test_relation_term_is_zero_without_an_anchor_node():
    car = node(2, 0.5, 0.8)
    g = graph([car], [])
    assert _relation_term(car, g, FakeQuery("behind")) == 0.0


def test_ego_anchored_relations_stay_zero():
    # The parser sets anchor=None for every ego case; that is the whole guard.
    bus, car = node(1, 0.5, 0.5, role=ROLE_ANCHOR), node(2, 0.5, 0.8)
    g = graph([bus, car], [{"source": 2, "target": 1, "relations": ["behind"]}])
    assert _relation_term(car, g, FakeQuery("behind", anchor=None)) == 0.0
    assert _relation_term(car, g, FakeQuery("counter_direction", anchor=None)) == 0.0


def test_relations_with_no_graph_equivalent_stay_zero():
    bus, car = node(1, 0.5, 0.5, role=ROLE_ANCHOR), node(2, 0.5, 0.8)
    g = graph([bus, car], [{"source": 2, "target": 1, "relations": ["behind"]}])
    for name in ("between", "following", "approaching", "overtaking", "leading"):
        assert _relation_term(car, g, FakeQuery(name)) == 0.0


def test_relation_term_reorders_a_ranking_confidence_gets_backwards():
    # The sweep's failure mode: the correct answer is the further, lower
    # confidence box, so confidence alone ranks the distractor first.
    bus = node(1, 0.5, 0.5, role=ROLE_ANCHOR, heading=(0.0, -1.0))
    target = node(2, 0.5, 0.85, conf=0.40)     # trailing the bus  -> the answer
    distractor = node(3, 0.5, 0.20, conf=0.70)  # leading the bus  -> not
    g = graph([bus, target, distractor], [
        {"source": 2, "target": 1, "relations": ["behind"]},
        {"source": 3, "target": 1, "relations": ["in-front-of"]},
    ])
    query = FakeQuery("behind")

    confidence_only = {n["track_id"]: n["confidence"] for n in (target, distractor)}
    assert max(confidence_only, key=confidence_only.get) == 3   # wrong

    scores = score_candidates(g, query)
    assert set(scores) == {2, 3}, "anchors are never scored"
    assert max(scores, key=scores.get) == 2                     # right


def test_weight_zero_falls_back_to_confidence_only():
    bus = node(1, 0.5, 0.5, role=ROLE_ANCHOR, heading=(0.0, -1.0))
    target, distractor = node(2, 0.5, 0.85, conf=0.40), node(3, 0.5, 0.20, conf=0.70)
    g = graph([bus, target, distractor], [
        {"source": 2, "target": 1, "relations": ["behind"]},
        {"source": 3, "target": 1, "relations": ["in-front-of"]},
    ])
    scores = score_candidates(g, FakeQuery("behind"), relation_weight=0.0)
    assert max(scores, key=scores.get) == 3


# --------------------------------------------------------------------------
# vocabulary contract — _EDGE_INVERSE must cover everything the builder emits
# --------------------------------------------------------------------------

def test_edge_inverse_covers_every_label_the_builder_can_emit():
    # _relation_term indexes _EDGE_INVERSE strictly; a label the builder emits
    # but the table lacks would be a KeyError mid-run.
    emitted = {
        "left-of", "right-of", "above", "below", "behind", "in-front-of",
        "near", "far", "larger-than", "smaller-than", "overlapping",
        "visually-similar",
    }
    assert emitted <= set(_EDGE_INVERSE)


def test_edge_inverse_is_an_involution():
    for label, inverse in _EDGE_INVERSE.items():
        assert _EDGE_INVERSE[inverse] == label


def test_every_mapped_relation_has_a_known_edge_label():
    assert set(_EDGE_FOR_RELATION.values()) <= set(_EDGE_INVERSE)


# --------------------------------------------------------------------------
# selection policy — a threshold, not an argmax
# --------------------------------------------------------------------------

def _scene(*, n_behind=1, n_ahead=1):
    """One anchor plus some candidates, half behind it and half in front."""
    bus = node(1, 0.5, 0.5, role=ROLE_ANCHOR, heading=(0.0, -1.0))
    nodes, edges, behind_ids = [bus], [], []
    tid = 2
    for k in range(n_behind):
        nodes.append(node(tid, 0.4 + 0.05 * k, 0.8, conf=0.3 + 0.05 * k))
        edges.append({"source": tid, "target": 1, "relations": ["behind"]})
        behind_ids.append(tid); tid += 1
    for k in range(n_ahead):
        nodes.append(node(tid, 0.4 + 0.05 * k, 0.2, conf=0.9))
        edges.append({"source": tid, "target": 1, "relations": ["in-front-of"]})
        tid += 1
    return graph(nodes, edges), behind_ids


def _tracks(g):
    return [FakeTrack(n["track_id"], n["cx_norm"], n["cy_norm"])
            for n in g["nodes"] if n["role"] == ROLE_TARGET]


def test_selection_returns_every_candidate_the_relation_holds_for():
    # The plural case: "cars behind the bus" must be able to return several.
    g, behind = _scene(n_behind=3, n_ahead=2)
    picked = select_answers(_tracks(g), g, FakeQuery("behind"), score_candidates(g, FakeQuery("behind")))
    assert sorted(t.track_id for t in picked) == sorted(behind)


def test_selection_is_not_an_argmax():
    g, behind = _scene(n_behind=3, n_ahead=1)
    picked = select_answers(_tracks(g), g, FakeQuery("behind"))
    assert len(picked) == 3, "top-1 would have collapsed a plural answer"


def test_selection_can_legitimately_return_nothing():
    # "nothing in this frame is behind the bus" is an answer, not a failure.
    g, _ = _scene(n_behind=0, n_ahead=2)
    assert select_answers(_tracks(g), g, FakeQuery("behind")) == []


def test_selection_drops_the_higher_confidence_candidate_when_it_fails_the_relation():
    g, behind = _scene(n_behind=1, n_ahead=1)
    picked = select_answers(_tracks(g), g, FakeQuery("behind"))
    assert [t.track_id for t in picked] == behind


def test_unjudgeable_relations_pass_every_candidate_through():
    # Grounding must never turn a missing feature into a confident empty answer.
    g, _ = _scene(n_behind=1, n_ahead=1)
    tracks = _tracks(g)
    for q in (FakeQuery(None), FakeQuery("behind", anchor=None), FakeQuery("following")):
        assert relation_holds(g, q) is None
        assert len(select_answers(tracks, g, q)) == len(tracks)


def test_missing_anchor_this_frame_passes_candidates_through():
    # The anchor was not detected here; that is not evidence against anyone.
    g, _ = _scene(n_behind=1, n_ahead=1)
    g["nodes"] = [n for n in g["nodes"] if n["role"] != ROLE_ANCHOR]
    assert relation_holds(g, FakeQuery("behind")) is None
    assert len(select_answers(_tracks(g), g, FakeQuery("behind"))) == 2


def test_explicit_count_keeps_the_highest_scoring():
    g, behind = _scene(n_behind=3, n_ahead=1)
    q = FakeQuery("behind")
    q.target = {"class": "car", "attrs": {"count": 2}}
    scores = score_candidates(g, q)
    picked = select_answers(_tracks(g), g, q, scores)
    assert len(picked) == 2
    best_two = sorted(behind, key=lambda t: -scores[t])[:2]
    assert sorted(t.track_id for t in picked) == sorted(best_two)


def test_count_is_left_unenforced_without_scores():
    g, _ = _scene(n_behind=3, n_ahead=1)
    q = FakeQuery("behind")
    q.target = {"class": "car", "attrs": {"count": 2}}
    assert len(select_answers(_tracks(g), g, q)) == 3


# --------------------------------------------------------------------------
# aspect ratio — the normalised space is anisotropic
# --------------------------------------------------------------------------

def test_depth_threshold_is_isotropic():
    # cx_norm divides by width and cy_norm by height, so a raw dot product in
    # that space over-weights the vertical term by (w/h)^2. On a 16:9 frame a
    # near-perpendicular offset would land the wrong side of the deadband.
    b = SceneGraphBuilder()
    b._aspect = 1080 / 1920

    ref = node(1, 0.50, 0.50, role=ROLE_ANCHOR, heading=(0.6, -0.8))
    cand = node(2, 0.55, 0.54)
    dx = cand["cx_norm"] - ref["cx_norm"]
    dy = cand["cy_norm"] - ref["cy_norm"]

    uncorrected = dx * 0.6 + dy * -0.8
    assert abs(uncorrected) < b.DEPTH_THRESH, "this case is inside the raw deadband"
    assert b._depth_relation(cand, ref, dx, dy) == "in-front-of"


def test_aspect_is_refreshed_from_the_frame():
    b = SceneGraphBuilder()
    assert b._aspect == 1.0
    b.update(0, [FakeTrack(1, 0.5, 0.5)], 1080, 1920)
    assert b._aspect == pytest.approx(1080 / 1920)


def test_square_frames_are_unaffected():
    b = SceneGraphBuilder()
    b.update(0, [FakeTrack(1, 0.5, 0.5)], 512, 512)
    assert b._aspect == 1.0

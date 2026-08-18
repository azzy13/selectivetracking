"""Contract tests for the Week 3 grounding layer (``eval/query_grounding.py``).

These pin the three things the rest of the pipeline is allowed to rely on:

  1. the detector caption carries the anchor class,
  2. every detection / track / node carries a role,
  3. anchors never leave the pipeline.

The scorer is a stub this week, so the tests assert its *shape* — which
candidates it scores and where the relation weight enters — not its values.
"""

import numpy as np
import pytest

from query_grounding import (ROLE_ANCHOR, ROLE_TARGET, anchor_tracks,
                             assign_detection_roles, assign_track_roles,
                             build_detector_prompt, caption_classes,
                             emitted_tracks, entity_phrase, score_candidates)
from query_parser import parse
from scene_graph import SceneGraphBuilder


class FakeTrack:
    """Minimal stand-in for an STrack — tlwh / track_id / score is all the graph reads."""

    def __init__(self, track_id, tlwh, score=0.9):
        self.track_id = track_id
        self.tlwh = list(tlwh)
        self.score = score
        self.tracklet_len = 3


# ---------------------------------------------------------------------------
# STEP 1 — parser -> detector prompt
# ---------------------------------------------------------------------------

class TestDetectorPrompt:

    def test_target_and_anchor_both_in_caption(self):
        assert build_detector_prompt(parse("red car behind the bus")) == "red car . bus"

    def test_anchor_attributes_survive(self):
        assert build_detector_prompt(parse("red sedan behind the white car")) == \
            "red sedan . white car"

    def test_plain_prompt_is_target_only(self):
        # The pre-Week-3 caption, unchanged: no anchor means nothing is added.
        assert build_detector_prompt(parse("red car")) == "red car"

    def test_unary_relation_has_no_anchor_class(self):
        assert build_detector_prompt(parse("moving cars")) == "car"

    def test_ego_anchored_relation_has_no_anchor_class(self):
        # "us" is not a tracked object, so there is nothing to detect for it.
        assert build_detector_prompt(parse("cars approaching us")) == "car"

    def test_dotted_form(self):
        assert build_detector_prompt(parse("red car behind the bus"), dotted=True) == \
            "red car . bus ."

    def test_identical_classes_collapse(self):
        # Two identical caption classes would make phrases unassignable.
        q = parse("red car behind the bus")
        q.anchor = dict(q.target)
        assert build_detector_prompt(q) == "red car"

    def test_caption_classes_round_trip(self):
        assert caption_classes(parse("red car behind the bus")) == ["red car", "bus"]

    def test_entity_phrase_drops_count(self):
        # count is a query-level constraint; the detector cannot ground it.
        assert entity_phrase({"class": "car", "attrs": {"color": "red", "count": 3}}) == "red car"

    def test_entity_phrase_keeps_size_and_other(self):
        assert entity_phrase({"class": "truck",
                              "attrs": {"size": "large", "other": ["damaged"]}}) == \
            "large damaged truck"


# ---------------------------------------------------------------------------
# STEP 2 — roles
# ---------------------------------------------------------------------------

class TestDetectionRoles:

    Q = parse("red car behind the bus")

    def test_anchor_phrase_tagged_anchor(self):
        assert assign_detection_roles(["bus"], self.Q) == [ROLE_ANCHOR]

    def test_target_phrase_tagged_candidate(self):
        assert assign_detection_roles(["red car"], self.Q) == [ROLE_TARGET]

    @pytest.mark.parametrize("phrase", ["", "car", "red car bus", "building"])
    def test_ambiguous_defaults_to_target_candidate(self, phrase):
        # Deliberate asymmetry: a mislabelled anchor costs a false positive the
        # scorer can down-weight; a mislabelled target is silently deleted.
        assert assign_detection_roles([phrase], self.Q) == [ROLE_TARGET]

    def test_shared_tokens_do_not_vote(self):
        # "car" is in both phrases, so only "white" / "red" decide.
        q = parse("red sedan behind the white car")
        assert assign_detection_roles(["white car", "red sedan"], q) == \
            [ROLE_ANCHOR, ROLE_TARGET]

    def test_plain_query_makes_everything_a_candidate(self):
        q = parse("red car")
        assert assign_detection_roles(["bus", "red car", ""], q) == [ROLE_TARGET] * 3


class TestTrackRoles:

    DETS = np.array([[10, 10, 50, 50, 0.9], [100, 100, 160, 160, 0.8]], dtype=np.float32)
    DET_ROLES = [ROLE_TARGET, ROLE_ANCHOR]

    def test_role_inherited_from_best_iou_detection(self):
        tracks = [FakeTrack(1, [10, 10, 40, 40]), FakeTrack(2, [100, 100, 60, 60])]
        assert assign_track_roles(tracks, self.DETS, self.DET_ROLES) == \
            {1: ROLE_TARGET, 2: ROLE_ANCHOR}

    def test_unmatched_track_defaults_to_candidate(self):
        tracks = [FakeTrack(9, [800, 800, 20, 20])]
        assert assign_track_roles(tracks, self.DETS, self.DET_ROLES) == {9: ROLE_TARGET}

    def test_sticky_memory_survives_a_missed_frame(self):
        memory = {}
        assign_track_roles([FakeTrack(2, [100, 100, 60, 60])], self.DETS,
                           self.DET_ROLES, sticky=memory)
        # Next frame the tracker coasts on prediction and matches no detection.
        roles = assign_track_roles([FakeTrack(2, [700, 700, 60, 60])],
                                   np.empty((0, 5), dtype=np.float32), [], sticky=memory)
        assert roles == {2: ROLE_ANCHOR}

    def test_role_is_a_majority_vote_not_the_last_frame(self):
        # The detector grounds the same object to the anchor phrase twice and
        # the target phrase once; a per-frame role would flap and leak the
        # anchor into the output on that third frame.
        memory = {}
        track = FakeTrack(2, [100, 100, 60, 60])
        for det_roles in ([ROLE_TARGET, ROLE_ANCHOR], [ROLE_TARGET, ROLE_ANCHOR],
                          [ROLE_TARGET, ROLE_TARGET]):
            roles = assign_track_roles([track], self.DETS, det_roles, sticky=memory)
        assert roles == {2: ROLE_ANCHOR}

    def test_majority_flips_when_the_evidence_does(self):
        memory = {}
        track = FakeTrack(2, [100, 100, 60, 60])
        assign_track_roles([track], self.DETS, [ROLE_TARGET, ROLE_ANCHOR], sticky=memory)
        for _ in range(2):
            roles = assign_track_roles([track], self.DETS,
                                       [ROLE_TARGET, ROLE_TARGET], sticky=memory)
        assert roles == {2: ROLE_TARGET}

    def test_no_detections_at_all(self):
        roles = assign_track_roles([FakeTrack(1, [0, 0, 10, 10])],
                                   np.empty((0, 5), dtype=np.float32), [])
        assert roles == {1: ROLE_TARGET}


# ---------------------------------------------------------------------------
# The rule: anchors are never emitted
# ---------------------------------------------------------------------------

class TestAnchorsNeverEmitted:

    TRACKS = [FakeTrack(1, [10, 10, 40, 40]), FakeTrack(2, [100, 100, 60, 60]),
              FakeTrack(3, [200, 200, 30, 30])]
    ROLES = {1: ROLE_TARGET, 2: ROLE_ANCHOR, 3: ROLE_TARGET}

    def test_emitted_excludes_anchors(self):
        assert [t.track_id for t in emitted_tracks(self.TRACKS, self.ROLES)] == [1, 3]

    def test_anchor_tracks_are_the_complement(self):
        assert [t.track_id for t in anchor_tracks(self.TRACKS, self.ROLES)] == [2]

    def test_no_roles_means_everything_is_emitted(self):
        # No grounding active: identical to the pre-Week-3 output path.
        assert len(emitted_tracks(self.TRACKS, None)) == 3
        assert anchor_tracks(self.TRACKS, None) == []

    def test_unknown_track_id_is_emitted_not_dropped(self):
        assert [t.track_id for t in emitted_tracks(self.TRACKS, {2: ROLE_ANCHOR})] == [1, 3]


# ---------------------------------------------------------------------------
# STEP 2 — the graph is built over everything, anchors included
# ---------------------------------------------------------------------------

class TestSceneGraphRoles:

    TRACKS = [FakeTrack(1, [100, 500, 60, 40], 0.71),
              FakeTrack(2, [300, 300, 200, 120], 0.62),
              FakeTrack(3, [700, 520, 55, 38], 0.55)]
    ROLES = {1: ROLE_TARGET, 2: ROLE_ANCHOR, 3: ROLE_TARGET}

    def _graph(self, roles=None):
        return SceneGraphBuilder("red car . bus").update(1, self.TRACKS, 1080, 1920, roles=roles)

    def test_anchor_is_a_node(self):
        graph = self._graph(self.ROLES)
        assert {n["track_id"]: n["role"] for n in graph["nodes"]} == self.ROLES
        assert graph["num_anchors"] == 1
        assert graph["num_target_candidates"] == 2

    def test_anchor_has_an_edge_to_every_target_candidate(self):
        graph = self._graph(self.ROLES)
        pairs = {(e["source"], e["target"]) for e in graph["edges"]}
        for candidate in (1, 3):
            assert (2, candidate) in pairs or (candidate, 2) in pairs

    def test_anchor_edge_survives_with_no_relation_labels(self):
        # Two boxes too close to fire any threshold: the edge must still exist,
        # because "no relation held" is information, and a missing edge is
        # indistinguishable from a missing node.
        tracks = [FakeTrack(1, [500, 500, 50, 50]), FakeTrack(2, [504, 504, 50, 50])]
        graph = SceneGraphBuilder().update(1, tracks, 1080, 1920,
                                           roles={1: ROLE_TARGET, 2: ROLE_ANCHOR})
        edges = [e for e in graph["edges"] if {e["source"], e["target"]} == {1, 2}]
        assert len(edges) == 1

    def test_default_role_is_target_candidate(self):
        # Omitting roles reproduces the pre-Week-3 graph.
        graph = self._graph(None)
        assert all(n["role"] == ROLE_TARGET for n in graph["nodes"])
        assert graph["num_anchors"] == 0


# ---------------------------------------------------------------------------
# STEP 3 — the scorer's shape (stub: values are not the deliverable)
# ---------------------------------------------------------------------------

class TestScoreCandidates:

    def _graph(self):
        tracks = [FakeTrack(1, [100, 500, 60, 40], 0.71),
                  FakeTrack(2, [300, 300, 200, 120], 0.62),
                  FakeTrack(3, [700, 520, 55, 38], 0.55)]
        return SceneGraphBuilder().update(1, tracks, 1080, 1920,
                                          roles={1: ROLE_TARGET, 2: ROLE_ANCHOR,
                                                 3: ROLE_TARGET})

    def test_scores_every_target_candidate(self):
        scores = score_candidates(self._graph(), parse("red car behind the bus"))
        assert set(scores) == {1, 3}

    def test_anchors_are_not_scored(self):
        # Anchors are not results, so they get no weight.
        assert 2 not in score_candidates(self._graph(), parse("red car behind the bus"))

    def test_nothing_is_filtered_out(self):
        # The scorer replaces a hard filter: every candidate comes back with a
        # weight, however low.
        graph = self._graph()
        scores = score_candidates(graph, parse("red car behind the bus"))
        assert len(scores) == graph["num_target_candidates"]

    def test_stub_returns_detector_confidence(self):
        # TODO(Week 4): this changes when _relation_term is implemented.
        scores = score_candidates(self._graph(), parse("red car behind the bus"))
        assert scores == {1: pytest.approx(0.71), 3: pytest.approx(0.55)}

    def test_relation_weight_is_a_multiplier_not_a_filter(self):
        # Overriding the weight cannot drop a candidate; with the stubbed
        # relation term at 0.0 it cannot change a score either.
        graph = self._graph()
        query = parse("red car behind the bus")
        base = score_candidates(graph, query)
        assert score_candidates(graph, query, relation_weight=0.0) == base
        assert score_candidates(graph, query, relation_weight=5.0) == base

    def test_plain_query_still_scores(self):
        # No relation at all: classic MOT, every candidate still gets a weight.
        assert set(score_candidates(self._graph(), parse("red car"))) == {1, 3}

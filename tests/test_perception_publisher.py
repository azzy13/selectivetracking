"""Field-mapping tests for the trinity Perception output."""

import types

import pytest

from mission_parser import EntityOfInterest
from perception_publisher import (
    GT_COLOR_NAMES, build_perception, match_entity, nodes_by_track_id,
    to_gt_color,
)


class Perception022:
    """Stand-in for the trinity_msgs 0.22 Perception (no frame_number)."""

    __slots__ = ("tracking_id", "target_entity_id", "detection_prob",
                 "location", "yaw", "entity_class", "entity_color",
                 "match_prob")

    def __init__(self):
        for slot in self.__slots__:
            setattr(self, slot, None)


class Perception058(Perception022):
    """0.58 adds frame_number, occlusion and pose."""

    __slots__ = ("frame_number", "occlusion", "pose")


def set_field(msg, field, value):
    if hasattr(msg, field):
        setattr(msg, field, value)
        return True
    return False


def make_track(track_id=7, score=0.83):
    return types.SimpleNamespace(track_id=track_id, score=score)


def entity(entity_id, color, entity_class="SEDAN.1"):
    return EntityOfInterest(entity_id=entity_id, entity_type="Car",
                            entity_class=entity_class, color=color)


# ------------------------------------------------------------ colour remap

@pytest.mark.parametrize("name", sorted(GT_COLOR_NAMES))
def test_gt_colours_pass_through(name):
    assert to_gt_color(name) == name


@pytest.mark.parametrize("name", ["gray", "unknown", "dark", "light", "", None])
def test_non_vocabulary_colours_become_empty(name):
    assert to_gt_color(name) == ""


def test_colour_is_normalised():
    assert to_gt_color("  Violet ") == "violet"


# ------------------------------------------------------------- association

def test_matching_colour_yields_entity_and_score():
    entities = [entity("Car495", "violet")]
    assert match_entity("violet", entities, {"Car495": 0.62}) == ("Car495", 0.62)


def test_mismatched_colour_yields_no_entity():
    entities = [entity("Car495", "violet")]
    assert match_entity("red", entities, {"Car495": 0.9}) == ("", 0.0)


def test_track_without_colour_stays_unmatched():
    entities = [entity("Car495", "violet")]
    assert match_entity("", entities, {"Car495": 0.9}) == ("", 0.0)


def test_no_entities_yields_no_match():
    assert match_entity("violet", [], {}) == ("", 0.0)


def test_highest_scoring_candidate_wins():
    entities = [entity("A", "red"), entity("B", "red")]
    assert match_entity("red", entities, {"A": 0.2, "B": 0.7}) == ("B", 0.7)


def test_zero_score_is_not_a_match():
    # a colour-compatible entity the filter gave no evidence for
    entities = [entity("Car495", "violet")]
    assert match_entity("violet", entities, {"Car495": 0.0}) == ("", 0.0)


def test_entity_without_declared_colour_matches_any():
    entities = [entity("Car495", "")]
    assert match_entity("red", entities, {"Car495": 0.5}) == ("Car495", 0.5)


# ----------------------------------------------------------- message build

def test_fields_populated_under_022():
    msg = build_perception(
        Perception022, make_track(track_id=7, score=0.83), frame_number=42,
        location=(120.5, -33.25, 0.0), entity_color="violet",
        target_entity_id="Car495", match_prob=0.62, set_field=set_field)

    assert msg.tracking_id == 7
    assert msg.detection_prob == pytest.approx(0.83)
    assert msg.location == pytest.approx([120.5, -33.25, 0.0])
    assert msg.target_entity_id == "Car495"
    assert msg.entity_color == "violet"
    assert msg.match_prob == pytest.approx(0.62)
    # deliberately not estimated
    assert msg.yaw == 0.0
    assert msg.entity_class == ""
    # 0.22 has no frame_number; setting it must not raise
    assert not hasattr(msg, "frame_number")


def test_frame_number_set_only_when_the_build_has_it():
    msg = build_perception(
        Perception058, make_track(), frame_number=42, location=(1.0, 2.0, 0.0),
        entity_color="", target_entity_id="", match_prob=0.0,
        set_field=set_field)

    assert msg.frame_number == 42
    assert msg.occlusion == ""
    assert msg.pose == ""


def test_uint16_wrap_on_ids_and_frames():
    msg = build_perception(
        Perception058, make_track(track_id=65540), frame_number=65538,
        location=(0.0, 0.0, 0.0), entity_color="", target_entity_id="",
        match_prob=0.0, set_field=set_field)

    assert msg.tracking_id == 4
    assert msg.frame_number == 2


def test_failed_projection_becomes_zeros():
    msg = build_perception(
        Perception022, make_track(), frame_number=1, location=None,
        entity_color="", target_entity_id="", match_prob=0.0,
        set_field=set_field)

    assert msg.location == pytest.approx([0.0, 0.0, 0.0])


# --------------------------------------------------------------- graph glue

def test_nodes_indexed_by_track_id():
    graph = {"nodes": [{"track_id": 3, "color": "red"},
                       {"track_id": 9, "color": "blue"}]}
    assert nodes_by_track_id(graph)[9]["color"] == "blue"


def test_empty_graph_is_tolerated():
    assert nodes_by_track_id({}) == {}
    assert nodes_by_track_id(None) == {}

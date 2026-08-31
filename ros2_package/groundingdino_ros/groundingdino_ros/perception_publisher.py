"""
Assembles msgs/PerceptionArray from GroundingDINO tracks.

Kept separate from the ROS2 node so the field mapping is testable without
a running ROS graph: everything here is plain Python over track objects,
scene-graph nodes and EntityOfInterest records.

Field mapping:

    tracking_id       ByteTrack track id, wrapped to uint16
    target_entity_id  episode entity_id, only on a class+colour match
    detection_prob    detector score for the track
    location          metres, AirSim NED, from geometry.GroundProjector
    yaw               left 0.0 -- see DEMO.md
    entity_class      left "" -- GroundingDINO cannot tell SEDAN.1 from SEDAN.2
    entity_color      scene-graph colour, in the benchmark's vocabulary
    match_prob        scene-graph mission score for the matched entity

Perception.frame_number is optional and is not in this repo's definitions,
so it is set only when the built message defines it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

UINT16_MODULUS = 65536

# The colour vocabulary the benchmark episodes use for attributes.color.
# target_map_node does an exact string match against it, so a name outside
# this set is worse than no name at all.
GT_COLOR_NAMES = frozenset({
    "red", "orange", "yellow", "green", "blue", "violet", "white", "black",
})


def to_gt_color(color: Optional[str]) -> str:
    """Map a scene-graph colour to the benchmark vocabulary, or "".

    The scene graph's gt_vocabulary mode already emits these names, but it
    can also return "gray" or "unknown", which no episode ever uses. Those
    become "" -- an empty colour is honest, a wrong one is matched against.
    """

    if not color:
        return ""
    color = color.strip().lower()
    return color if color in GT_COLOR_NAMES else ""


def match_entity(
    entity_color: str,
    entities: Sequence[Any],
    scores_by_entity: Dict[str, float],
) -> Tuple[str, float]:
    """Pick the entity of interest this track's appearance matches.

    Returns (target_entity_id, match_prob), or ("", 0.0) when nothing
    matches. Only colour is compared: entity_class is the AirSim class
    token ("SEDAN.1"), which an open-vocabulary detector cannot resolve,
    so requiring it would reject every track.

    A track with no colour evidence stays unmatched rather than being
    attached to an arbitrary entity.
    """

    if not entity_color or not entities:
        return "", 0.0

    best_id, best_score = "", 0.0
    for entity in entities:
        if entity.color and entity.color != entity_color:
            continue
        score = float(scores_by_entity.get(entity.entity_id, 0.0))
        if score > best_score:
            best_id, best_score = entity.entity_id, score

    return best_id, best_score


def build_perception(
    perception_cls,
    track,
    frame_number: int,
    location: Optional[Sequence[float]],
    entity_color: str,
    target_entity_id: str,
    match_prob: float,
    set_field,
):
    """Fill one msgs/Perception from a track.

    location is metres NED, or None when projection failed -- in which
    case the message carries zeros, which is what an unset float32[3] is
    anyway, and the caller decides whether to publish it.
    """

    msg = perception_cls()

    msg.tracking_id = int(track.track_id) % UINT16_MODULUS
    msg.target_entity_id = target_entity_id
    msg.detection_prob = float(track.score)
    msg.location = [float(v) for v in (location if location is not None
                                       else (0.0, 0.0, 0.0))]

    # Neither is estimated by this pipeline; see DEMO.md.
    msg.yaw = 0.0
    msg.entity_class = ""

    msg.entity_color = entity_color
    msg.match_prob = float(match_prob)

    # Optional; absent from this repo's definitions.
    set_field(msg, "frame_number", int(frame_number) % UINT16_MODULUS)
    set_field(msg, "occlusion", "")
    set_field(msg, "pose", "")

    return msg


def nodes_by_track_id(frame_graph: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Index a scene-graph frame's nodes by track id."""

    return {node["track_id"]: node
            for node in (frame_graph or {}).get("nodes", [])}

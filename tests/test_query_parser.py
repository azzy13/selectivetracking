"""Week 3 deliverable: the query schema, pinned by example.

Each prompt below is paired with its full expected parse. Together with
``query_parser.RELATIONS`` these cases *are* the contract the Week 4
association term is written against — change them only deliberately.

Run:  python -m pytest tests/test_query_parser.py -v
"""

import pytest

from query_parser import (
    DEFAULT_RELATION_WEIGHT,
    RELATION_KINDS,
    RELATIONS,
    Query,
    QueryParseError,
    UnknownRelationError,
    parse,
)


def _rel(name, arity, temporal, weight=DEFAULT_RELATION_WEIGHT):
    return {"name": name, "arity": arity, "temporal": temporal, "weight": weight}


def _spatial(name):
    return _rel(name, "binary", False)


def _motion(name):
    return _rel(name, "unary", False)


def _temporal(name):
    return _rel(name, "binary", True)


# ---------------------------------------------------------------------------
# The prompt table: (prompt, expected Query.to_dict())
# ---------------------------------------------------------------------------

PLAIN_CASES = [
    ("cars",
     {"target": {"class": "car", "attrs": {}},
      "anchor": None, "relation": None}),

    ("vehicles",
     {"target": {"class": "vehicle", "attrs": {}},
      "anchor": None, "relation": None}),

    ("the truck",
     {"target": {"class": "truck", "attrs": {}},
      "anchor": None, "relation": None}),
]

ATTRIBUTE_CASES = [
    ("red car",
     {"target": {"class": "car", "attrs": {"color": "red"}},
      "anchor": None, "relation": None}),

    ("large white truck",
     {"target": {"class": "truck", "attrs": {"size": "large", "color": "white"}},
      "anchor": None, "relation": None}),

    # "silver" tags as a compound NOUN, not an adjective — still a color.
    ("silver car",
     {"target": {"class": "car", "attrs": {"color": "silver"}},
      "anchor": None, "relation": None}),

    ("the black suv",
     {"target": {"class": "suv", "attrs": {"color": "black"}},
      "anchor": None, "relation": None}),

    ("small blue van",
     {"target": {"class": "van", "attrs": {"size": "small", "color": "blue"}},
      "anchor": None, "relation": None}),

    # Unrecognised descriptive adjectives are kept, not silently dropped.
    ("damaged red car",
     {"target": {"class": "car", "attrs": {"color": "red", "other": ["damaged"]}},
      "anchor": None, "relation": None}),
]

SPATIAL_INSTANT_CASES = [
    ("red car behind the bus",
     {"target": {"class": "car", "attrs": {"color": "red"}},
      "anchor": {"class": "bus", "attrs": {}},
      "relation": _spatial("behind")}),

    ("car in front of the van",
     {"target": {"class": "car", "attrs": {}},
      "anchor": {"class": "van", "attrs": {}},
      "relation": _spatial("in_front_of")}),

    ("the car to the left of the bus",
     {"target": {"class": "car", "attrs": {}},
      "anchor": {"class": "bus", "attrs": {}},
      "relation": _spatial("left_of")}),

    ("the white van on the left of the truck",
     {"target": {"class": "van", "attrs": {"color": "white"}},
      "anchor": {"class": "truck", "attrs": {}},
      "relation": _spatial("left_of")}),

    ("the black suv right of the bus",
     {"target": {"class": "suv", "attrs": {"color": "black"}},
      "anchor": {"class": "bus", "attrs": {}},
      "relation": _spatial("right_of")}),

    ("silver car next to the truck",
     {"target": {"class": "car", "attrs": {"color": "silver"}},
      "anchor": {"class": "truck", "attrs": {}},
      "relation": _spatial("next_to")}),

    ("car beside the truck",
     {"target": {"class": "car", "attrs": {}},
      "anchor": {"class": "truck", "attrs": {}},
      "relation": _spatial("next_to")}),

    ("the pedestrian between the two cars",
     {"target": {"class": "pedestrian", "attrs": {}},
      "anchor": {"class": "car", "attrs": {"count": 2}},
      "relation": _spatial("between")}),
]

MOTION_STATE_CASES = [
    # Unary: relation is non-None but anchor stays None.
    ("moving cars",
     {"target": {"class": "car", "attrs": {}},
      "anchor": None, "relation": _motion("moving")}),

    ("stationary car",
     {"target": {"class": "car", "attrs": {}},
      "anchor": None, "relation": _motion("stationary")}),

    ("parked cars",
     {"target": {"class": "car", "attrs": {}},
      "anchor": None, "relation": _motion("stationary")}),

    ("braking cars",
     {"target": {"class": "car", "attrs": {}},
      "anchor": None, "relation": _motion("braking")}),

    ("turning vehicles",
     {"target": {"class": "vehicle", "attrs": {}},
      "anchor": None, "relation": _motion("turning")}),

    # Refer-KITTI phrasing.
    ("cars coming from the counterdirection",
     {"target": {"class": "car", "attrs": {}},
      "anchor": None, "relation": _motion("counter_direction")}),

    ("cars in the same direction as ours",
     {"target": {"class": "car", "attrs": {}},
      "anchor": None, "relation": _motion("same_direction")}),

    ("cars driving in the same direction",
     {"target": {"class": "car", "attrs": {}},
      "anchor": None, "relation": _motion("same_direction")}),

    ("oncoming vehicles",
     {"target": {"class": "vehicle", "attrs": {}},
      "anchor": None, "relation": _motion("counter_direction")}),
]

TEMPORAL_CASES = [
    ("car following the truck",
     {"target": {"class": "car", "attrs": {}},
      "anchor": {"class": "truck", "attrs": {}},
      "relation": _temporal("following")}),

    ("the truck overtaking the car",
     {"target": {"class": "truck", "attrs": {}},
      "anchor": {"class": "car", "attrs": {}},
      "relation": _temporal("overtaking")}),

    ("car leading the convoy",
     {"target": {"class": "car", "attrs": {}},
      "anchor": {"class": "convoy", "attrs": {}},
      "relation": _temporal("leading")}),

    ("the bus approaching the intersection",
     {"target": {"class": "bus", "attrs": {}},
      "anchor": {"class": "intersection", "attrs": {}},
      "relation": _temporal("approaching")}),

    # Ego anchor: the relation stays binary, but there is no tracked anchor.
    ("cars approaching us",
     {"target": {"class": "car", "attrs": {}},
      "anchor": None,
      "relation": _temporal("approaching")}),
]

ALL_CASES = (PLAIN_CASES + ATTRIBUTE_CASES + SPATIAL_INSTANT_CASES
             + MOTION_STATE_CASES + TEMPORAL_CASES)


@pytest.mark.parametrize("prompt,expected", ALL_CASES, ids=[c[0] for c in ALL_CASES])
def test_expected_parse(prompt, expected):
    assert parse(prompt).to_dict() == expected


# ---------------------------------------------------------------------------
# Unknown relations: raise, never guess
# ---------------------------------------------------------------------------

UNKNOWN_CASES = [
    ("car under the bridge", "under"),          # unmapped preposition
    ("cars honking at the truck", "honk"),      # unmapped verb
    # "in" is only meaningful inside a known phrase; region prompts belong to
    # SceneGraphMissionFilter, not the query parser.
    ("cars in the top left", "in"),
]


@pytest.mark.parametrize("prompt,surface", UNKNOWN_CASES, ids=[c[0] for c in UNKNOWN_CASES])
def test_unknown_relation_raises(prompt, surface):
    with pytest.raises(UnknownRelationError) as exc:
        parse(prompt)
    msg = str(exc.value)
    assert surface in msg
    assert "unknown relation" in msg
    # The message must point at the editable table and the closed vocabulary.
    assert "query_parser.py" in msg


def test_empty_prompt_raises():
    with pytest.raises(QueryParseError):
        parse("   ")


def test_unknown_relation_is_a_query_parse_error():
    assert issubclass(UnknownRelationError, QueryParseError)


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", [c[0] for c in ALL_CASES])
def test_target_always_present(prompt):
    q = parse(prompt)
    assert q.target["class"]
    assert isinstance(q.target["attrs"], dict)


@pytest.mark.parametrize("prompt", [c[0] for c in ALL_CASES])
def test_relation_shape_is_exactly_the_contract(prompt):
    """The relation payload carries the four contract keys and nothing else.

    Extra per-relation metadata (kind, ego_relative) lives in RELATIONS, so the
    query itself never drifts from the schema Week 4 codes against.
    """
    q = parse(prompt)
    if q.relation is None:
        return
    assert set(q.relation) == {"name", "arity", "temporal", "weight"}
    assert q.relation["name"] in RELATIONS
    assert q.relation["arity"] in ("unary", "binary")
    assert isinstance(q.relation["temporal"], bool)
    assert isinstance(q.relation["weight"], float)


@pytest.mark.parametrize("prompt", [c[0] for c in ALL_CASES])
def test_unary_relations_never_carry_an_anchor(prompt):
    q = parse(prompt)
    if q.relation is not None and q.relation["arity"] == "unary":
        assert q.anchor is None


@pytest.mark.parametrize("prompt", [c[0] for c in ALL_CASES])
def test_anchor_implies_a_relation(prompt):
    """An anchor is only meaningful as the second argument of a relation."""
    q = parse(prompt)
    if q.anchor is not None:
        assert q.relation is not None
        assert q.relation["arity"] == "binary"


@pytest.mark.parametrize("prompt", [c[0] for c in ALL_CASES])
def test_relation_is_soft_never_a_hard_filter(prompt):
    """A weight is always present, and no key implies filtering or an eval order.

    This is what keeps the association stage from reading the query as
    "find the anchor first, then search near it".
    """
    q = parse(prompt)
    if q.relation is None:
        return
    assert q.relation["weight"] > 0.0
    forbidden = {"filter", "hard", "required", "must", "order", "stage", "first"}
    assert not (forbidden & set(q.relation))
    assert not (forbidden & set(q.to_dict()))


def test_weight_is_overridable():
    q = parse("red car behind the bus", weight=0.35)
    assert q.relation["weight"] == 0.35
    # Overriding the weight changes nothing else about the parse.
    assert q.relation["name"] == "behind"
    assert q.anchor == {"class": "bus", "attrs": {}}


def test_plain_prompts_reduce_to_classic_mot():
    for prompt, _ in PLAIN_CASES + ATTRIBUTE_CASES:
        q = parse(prompt)
        assert q.is_plain
        assert q.anchor is None and q.relation is None
    assert not parse("moving cars").is_plain


# ---------------------------------------------------------------------------
# Relation vocabulary
# ---------------------------------------------------------------------------

def test_vocabulary_is_closed_and_well_formed():
    expected = {
        "spatial_instant": {"behind", "in_front_of", "left_of", "right_of",
                            "next_to", "between"},
        "motion_state": {"moving", "stationary", "braking", "turning",
                         "counter_direction", "same_direction"},
        "temporal": {"following", "approaching", "overtaking", "leading"},
    }
    by_kind = {k: set() for k in RELATION_KINDS}
    for name, spec in RELATIONS.items():
        assert name == spec.name
        assert spec.kind in RELATION_KINDS
        by_kind[spec.kind].add(name)
    assert by_kind == expected


def test_arity_and_temporal_flags_follow_the_kind():
    for spec in RELATIONS.values():
        if spec.kind == "spatial_instant":
            assert spec.arity == "binary" and spec.temporal is False
        elif spec.kind == "motion_state":
            assert spec.arity == "unary" and spec.temporal is False
        else:
            assert spec.arity == "binary" and spec.temporal is True


def test_only_ego_relative_relations_are_flagged_ego_relative():
    ego = {n for n, s in RELATIONS.items() if s.ego_relative}
    assert ego == {"counter_direction", "same_direction"}


# ---------------------------------------------------------------------------
# Ego-relative predicates: parse fine, flagged, never computed
# ---------------------------------------------------------------------------

EGO_PROMPTS = [
    "cars coming from the counterdirection",
    "cars in the same direction as ours",
    "cars driving in the same direction",
    "oncoming vehicles",
    "cars approaching us",
]


@pytest.mark.parametrize("prompt", EGO_PROMPTS)
def test_ego_relative_prompts_are_noted_not_computed(prompt):
    q = parse(prompt)
    assert q.relation is not None
    assert q.notes, "ego-relative parse must carry a schema note"
    note = " ".join(q.notes).lower()
    assert "reference heading" in note
    assert "carla" in note
    # Flagged only — no computed heading/angle leaks into the query.
    assert "heading" not in q.target["attrs"]
    assert q.anchor is None


def test_non_ego_prompts_carry_no_notes():
    for prompt in ["red car behind the bus", "moving cars", "cars",
                   "car following the truck"]:
        assert parse(prompt).notes == []


def test_relation_kind_is_recoverable_from_the_query():
    assert parse("red car behind the bus").relation_kind == "spatial_instant"
    assert parse("moving cars").relation_kind == "motion_state"
    assert parse("car following the truck").relation_kind == "temporal"
    assert parse("cars").relation_kind is None


def test_query_is_the_returned_type():
    assert isinstance(parse("cars"), Query)

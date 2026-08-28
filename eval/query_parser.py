#!/usr/bin/env python3
"""
Natural-language query parser for referring multi-object tracking.

Decomposes a tracking prompt ("red car behind the bus") into a structured
``Query`` that the association stage consumes:

    Query
      target:   {class, attrs}          # always present
      anchor:   {class, attrs} | None   # None for plain/unary/ego-relative
      relation: {name, arity, temporal, weight} | None

Design contract with the association stage
------------------------------------------
The anchor and the relation are SOFT constraints.  ``relation.weight`` is a
score multiplier, never a filter, and nothing in this schema encodes an
evaluation order: there is no "resolve the anchor first, then search near it"
structure.  The anchor is a weighted constraint scored against *all* candidate
tracks, exactly like the target's own attributes.

A plain prompt ("cars", "red car") yields ``anchor=None, relation=None``, which
is classic MOT — the association stage should fall through to its default path.

This module classifies; it does not compute.  No geometry, no motion math, no
scene-graph lookups.  Relations that cannot be evaluated against the available
ground truth (ego-relative ones) parse normally and are flagged in
``Query.notes``.

Usage:
    from query_parser import parse, UnknownRelationError

    q = parse("red car behind the bus")
    q.target                 # {'class': 'car', 'attrs': {'color': 'red'}}
    q.anchor                 # {'class': 'bus', 'attrs': {}}
    q.relation               # {'name': 'behind', 'arity': 'binary',
                             #  'temporal': False, 'weight': 1.0}
    q.to_dict()              # plain-JSON form of the whole query

Requires spaCy and the small English model:
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Single-source the color vocabulary from the scene graph so the parser and the
# LAB color classifier can never drift apart.
from scene_graph import _COLOR_KEYWORDS


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class QueryParseError(ValueError):
    """Base class for all parse failures."""


class UnknownRelationError(QueryParseError):
    """A preposition or verb modifies the target but maps to no canonical relation.

    Raised instead of guessing.  Fix by adding the surface form to
    ``PHRASE_RELATIONS`` / ``VERB_RELATIONS`` / ``MODIFIER_RELATIONS`` below.
    """


# ---------------------------------------------------------------------------
# Relation vocabulary (closed set)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RelationSpec:
    """One entry in the closed relation vocabulary.

    kind:
        'spatial_instant' — evaluable from a single frame's geometry
        'motion_state'    — a property of one track's own trajectory
        'temporal'        — a pairwise comparison across frames

    arity:
        'binary' relations take an anchor; 'unary' relations never do.

    temporal:
        True only for the 'temporal' kind, i.e. relations that need a *pairwise*
        cross-frame comparison.  motion_state relations are False even though
        they are derived from a track's history, because the scene graph already
        maintains that per-node (see ``SceneGraphBuilder._motion_attrs``) — no
        pairwise history is needed.

    ego_relative:
        The relation is defined against the ego vehicle's heading.  Parses fine,
        but cannot be scored without a reference heading (absent from the CARLA
        GT).  Flagged in ``Query.notes``; never computed here.

    scene_graph_hint:
        Where an equivalent signal already lives in the scene graph, for Week 4.
        Documentation only — not part of the query contract.
    """

    name: str
    kind: str
    arity: str
    temporal: bool
    default_weight: float = 1.0
    ego_relative: bool = False
    scene_graph_hint: Optional[str] = None


def _spec(name, kind, arity, temporal, ego_relative=False, hint=None) -> RelationSpec:
    return RelationSpec(
        name=name, kind=kind, arity=arity, temporal=temporal,
        ego_relative=ego_relative, scene_graph_hint=hint,
    )


#: The closed relation vocabulary.  Nothing outside this table may appear in a
#: parsed query.  Keyed by canonical relation name.
RELATIONS: Dict[str, RelationSpec] = {
    # --- spatial_instant (binary, single-frame geometry) -------------------
    "behind":      _spec("behind", "spatial_instant", "binary", False,
                         hint="edge 'behind'"),
    "in_front_of": _spec("in_front_of", "spatial_instant", "binary", False,
                         hint="edge 'in-front-of'"),
    "left_of":     _spec("left_of", "spatial_instant", "binary", False,
                         hint="edge 'left-of'"),
    "right_of":    _spec("right_of", "spatial_instant", "binary", False,
                         hint="edge 'right-of'"),
    "next_to":     _spec("next_to", "spatial_instant", "binary", False,
                         hint="edge 'near'"),
    "between":     _spec("between", "spatial_instant", "binary", False),

    # --- motion_state (unary, one track's own trajectory) ------------------
    "moving":      _spec("moving", "motion_state", "unary", False,
                         hint="node motion == 'moving'"),
    "stationary":  _spec("stationary", "motion_state", "unary", False,
                         hint="node motion == 'stationary'"),
    "braking":     _spec("braking", "motion_state", "unary", False),
    "turning":     _spec("turning", "motion_state", "unary", False),
    "counter_direction": _spec("counter_direction", "motion_state", "unary", False,
                               ego_relative=True),
    "same_direction":    _spec("same_direction", "motion_state", "unary", False,
                               ego_relative=True),

    # --- temporal (binary, pairwise across frames) -------------------------
    "following":   _spec("following", "temporal", "binary", True),
    "approaching": _spec("approaching", "temporal", "binary", True),
    "overtaking":  _spec("overtaking", "temporal", "binary", True),
    "leading":     _spec("leading", "temporal", "binary", True),
}

RELATION_KINDS = ("spatial_instant", "motion_state", "temporal")

#: Default soft weight applied to a relation when the caller does not override it.
DEFAULT_RELATION_WEIGHT = 1.0


# ---------------------------------------------------------------------------
# Surface-form lookup tables — edit these to teach the parser new phrasings
# ---------------------------------------------------------------------------

#: Multi-token phrases, matched against the token sequence.  Longest match wins,
#: and a phrase always beats a verb or modifier match.  Phrase tokens are
#: "consumed": they can never be mistaken for the anchor noun chunk.
PHRASE_RELATIONS: Dict[Tuple[str, ...], str] = {
    # spatial
    ("behind",):                      "behind",
    ("in", "back", "of"):             "behind",
    ("in", "front", "of"):            "in_front_of",
    ("ahead", "of"):                  "in_front_of",
    ("to", "the", "left", "of"):      "left_of",
    ("on", "the", "left", "of"):      "left_of",
    ("left", "of"):                   "left_of",
    ("to", "the", "right", "of"):     "right_of",
    ("on", "the", "right", "of"):     "right_of",
    ("right", "of"):                  "right_of",
    ("next", "to"):                   "next_to",
    ("beside",):                      "next_to",
    ("alongside",):                   "next_to",
    ("between",):                     "between",
    # ego-relative motion
    ("in", "the", "same", "direction"):     "same_direction",
    ("in", "the", "same", "lane"):          "same_direction",
    ("same", "direction"):                  "same_direction",
    ("in", "the", "opposite", "direction"): "counter_direction",
    ("from", "the", "opposite", "direction"): "counter_direction",
    ("from", "the", "counterdirection"):    "counter_direction",
    ("in", "the", "counterdirection"):      "counter_direction",
    ("opposite", "direction"):              "counter_direction",
    ("counter", "direction"):               "counter_direction",
    ("counterdirection",):                  "counter_direction",
    ("oncoming",):                          "counter_direction",
}

#: Verb lemma -> canonical relation, for participles attached to the target
#: ("car *following* the truck", "*braking* cars").
VERB_RELATIONS: Dict[str, str] = {
    "follow":   "following",
    "trail":    "following",
    "approach": "approaching",
    "overtake": "overtaking",
    "pass":     "overtaking",
    "lead":     "leading",
    "brake":    "braking",
    "turn":     "turning",
    "move":     "moving",
    "drive":    "moving",
    "travel":   "moving",
    "come":     "moving",
    "go":       "moving",
    "park":     "stationary",
    "stop":     "stationary",
    "wait":     "stationary",
}

#: Adjective / participle modifier -> canonical relation ("*moving* cars").
MODIFIER_RELATIONS: Dict[str, str] = {
    "moving":     "moving",
    "stationary": "stationary",
    "parked":     "stationary",
    "stopped":    "stationary",
    "still":      "stationary",
    "idle":       "stationary",
    "braking":    "braking",
    "turning":    "turning",
    "oncoming":   "counter_direction",
}


# ---------------------------------------------------------------------------
# Attribute vocabulary
# ---------------------------------------------------------------------------

#: Colors recognised in prompts.  Sourced from the scene graph's LAB classifier
#: vocabulary so a parsed color is always comparable to ``node["color"]``.
COLOR_SET = frozenset(_COLOR_KEYWORDS)

#: Size words normalised onto ``SceneGraphBuilder._size_label`` outputs.
SIZE_ALIASES: Dict[str, str] = {
    "tiny":     "tiny",
    "small":    "small",
    "little":   "small",
    "compact":  "small",
    "medium":   "medium",
    "midsize":  "medium",
    "large":    "large",
    "big":      "large",
    "huge":     "large",
    "giant":    "large",
}

#: Pronouns and nouns that refer to the ego vehicle rather than a tracked object.
EGO_TOKENS = frozenset({
    "us", "we", "our", "ours", "ourselves", "me", "my", "mine", "i",
    "ego", "self",
})

#: Determiners/quantifiers that carry no discriminative signal.
_STOP_MODIFIERS = frozenset({"the", "a", "an", "this", "that", "these", "those",
                             "all", "any", "some", "other", "same"})


# ---------------------------------------------------------------------------
# Query schema
# ---------------------------------------------------------------------------

@dataclass
class Query:
    """Structured form of a tracking prompt.

    ``anchor`` and ``relation`` are soft constraints — see the module docstring.
    ``notes`` carries schema-level caveats (e.g. ego-relative relations that
    need a reference heading) for the association stage to surface or ignore.
    """

    target: Dict[str, Any]
    anchor: Optional[Dict[str, Any]] = None
    relation: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)
    text: str = ""

    @property
    def is_plain(self) -> bool:
        """True when this reduces to classic MOT (no anchor, no relation)."""
        return self.anchor is None and self.relation is None

    @property
    def relation_kind(self) -> Optional[str]:
        """'spatial_instant' | 'motion_state' | 'temporal', or None."""
        if self.relation is None:
            return None
        return RELATIONS[self.relation["name"]].kind

    def to_dict(self) -> Dict[str, Any]:
        """Plain-JSON form.  Key order is presentational, not an eval order."""
        return {
            "target":   self.target,
            "anchor":   self.anchor,
            "relation": self.relation,
        }


def _make_relation(name: str, weight: Optional[float] = None) -> Dict[str, Any]:
    """Build the relation payload.  Exactly the four contract keys.

    Anything else about the relation (kind, ego_relative, scene-graph hint) is
    looked up from ``RELATIONS[name]`` rather than duplicated into the query.
    """
    spec = RELATIONS[name]
    return {
        "name":     spec.name,
        "arity":    spec.arity,
        "temporal": spec.temporal,
        "weight":   float(spec.default_weight if weight is None else weight),
    }


# ---------------------------------------------------------------------------
# spaCy pipeline (lazy singleton)
# ---------------------------------------------------------------------------

_NLP = None
_SPACY_MODEL = "en_core_web_sm"


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - environment issue
            raise QueryParseError(
                "spaCy is required by query_parser. Install with: pip install spacy"
            ) from exc
        try:
            _NLP = spacy.load(_SPACY_MODEL)
        except OSError as exc:  # pragma: no cover - environment issue
            raise QueryParseError(
                f"spaCy model '{_SPACY_MODEL}' is not installed. "
                f"Install with: python -m spacy download {_SPACY_MODEL}"
            ) from exc
    return _NLP


# ---------------------------------------------------------------------------
# Phrase matching
# ---------------------------------------------------------------------------

_MAX_PHRASE_LEN = max(len(p) for p in PHRASE_RELATIONS)


def _match_phrase(doc) -> Optional[Tuple[str, int, int]]:
    """Find the first relation phrase in the token sequence.

    Returns (relation_name, first_token_idx, last_token_idx) or None.
    Longest match wins at each position, so "in front of" beats a bare "in"
    and "to the left of" beats "left of".

    Punctuation is skipped, which makes "counter-direction" match the
    ("counter", "direction") phrase.
    """
    idx = [t.i for t in doc if not t.is_punct]
    words = [doc[i].lower_ for i in idx]

    for start in range(len(words)):
        for length in range(min(_MAX_PHRASE_LEN, len(words) - start), 0, -1):
            name = PHRASE_RELATIONS.get(tuple(words[start:start + length]))
            if name is not None:
                return name, idx[start], idx[start + length - 1]
    return None


# ---------------------------------------------------------------------------
# Noun-chunk -> entity
# ---------------------------------------------------------------------------

def _entity_from_chunk(chunk, skip: frozenset = frozenset()) -> Dict[str, Any]:
    """Build a {class, attrs} entity from a noun chunk.

    class: the chunk root's lemma (so "cars" -> "car"), prefixed by any
           non-color compound noun ("fire truck").
    attrs: color / size / count, plus any remaining descriptive adjectives
           under "other".

    ``skip`` holds token indices already consumed as a relation ("moving" in
    "moving cars", "oncoming" in "oncoming vehicles") so a word can never be
    both the relation and an attribute.
    """
    root = chunk.root
    attrs: Dict[str, Any] = {}
    class_parts: List[str] = []
    other: List[str] = []

    for tok in root.children:
        if tok.i >= root.i:
            continue  # only pre-modifiers describe the head noun
        if tok.i in skip or tok.is_punct:
            continue
        low = tok.lower_

        if low in _STOP_MODIFIERS:
            continue

        if tok.dep_ == "nummod":
            try:
                attrs["count"] = int(tok.text)
            except ValueError:
                attrs["count"] = _word_to_int(low, default=tok.text)
            continue

        if low in COLOR_SET:
            attrs["color"] = "gray" if low == "grey" else low
            continue
        if low in SIZE_ALIASES:
            attrs["size"] = SIZE_ALIASES[low]
            continue
        if tok.dep_ == "compound":
            class_parts.append(low)
            continue
        if tok.dep_ == "amod":
            other.append(low)  # surface form: "damaged", not the lemma "damage"

    if other:
        attrs["other"] = other

    class_parts.append(root.lemma_.lower())
    return {"class": " ".join(class_parts), "attrs": attrs}


_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _word_to_int(word: str, default):
    return _WORD_NUMBERS.get(word, default)


def _is_ego(chunk) -> bool:
    """True when a chunk refers to the ego vehicle rather than a tracked object."""
    return chunk.root.lower_ in EGO_TOKENS or chunk.root.lemma_.lower() in EGO_TOKENS


# ---------------------------------------------------------------------------
# Relation detection
# ---------------------------------------------------------------------------

def _find_target_chunk(doc, chunks):
    """The chunk headed by the sentence ROOT, else the first chunk.

    The ROOT rule handles "cars coming from the counterdirection" (ROOT=cars).
    The fallback handles "braking cars", where the gerund is ROOT and the only
    noun chunk is its object.
    """
    for chunk in chunks:
        if chunk.root.dep_ == "ROOT":
            return chunk
    return chunks[0]


def _find_verb_relation(doc, target_chunk):
    """Map a participle/verb attached to the target onto a canonical relation.

    Returns (relation_name, verb_token) or None.
    """
    root = target_chunk.root
    candidates = []

    # "car [following] the truck" — acl/relcl/advcl child of the target
    for tok in root.children:
        if tok.pos_ in ("VERB", "AUX") and tok.dep_ in ("acl", "relcl", "advcl"):
            candidates.append(tok)
    # "[braking] cars" — target is the object of a gerund head
    if root.head is not root and root.head.pos_ == "VERB" and root.dep_ in ("dobj", "nsubj"):
        candidates.append(root.head)

    for tok in candidates:
        name = VERB_RELATIONS.get(tok.lemma_.lower())
        if name is not None:
            return name, tok
    return None


def _find_modifier_relation(target_chunk):
    """Map an adjectival/participial pre-modifier onto a relation.

    Returns (relation_name, token) or None.  Handles "moving cars",
    "stationary car", "parked cars".
    """
    root = target_chunk.root
    for tok in root.children:
        if tok.i >= root.i or tok.dep_ != "amod":
            continue
        name = MODIFIER_RELATIONS.get(tok.lower_) or MODIFIER_RELATIONS.get(tok.lemma_.lower())
        if name is not None:
            return name, tok
    return None


def _check_unmapped(doc, target_chunk, consumed: set) -> None:
    """Raise UnknownRelationError for relational modifiers of the target we
    could not map to the closed vocabulary.

    Deliberately narrow: only tokens that modify the *target* are validated.
    A preposition buried inside the anchor phrase ("... the bus at the corner")
    is left alone rather than rejecting an otherwise-good parse.
    """
    root = target_chunk.root

    for tok in root.children:
        if tok.i in consumed:
            continue
        if tok.dep_ == "prep" and tok.pos_ == "ADP":
            raise UnknownRelationError(
                f"unknown relation: preposition '{tok.text}' in "
                f"'{doc.text}' maps to no canonical relation. "
                f"Add it to PHRASE_RELATIONS in query_parser.py, or use one of: "
                f"{', '.join(sorted(RELATIONS))}."
            )
        if tok.pos_ == "VERB" and tok.dep_ in ("acl", "relcl", "advcl"):
            raise UnknownRelationError(
                f"unknown relation: verb '{tok.lemma_}' in "
                f"'{doc.text}' maps to no canonical relation. "
                f"Add it to VERB_RELATIONS in query_parser.py, or use one of: "
                f"{', '.join(sorted(RELATIONS))}."
            )


def _find_anchor(doc, chunks, target_chunk, after_idx, verb_token):
    """Locate the anchor chunk for a binary relation.

    Two routes, matching the two ways a binary relation surfaces:
      - prepositional: the first noun chunk headed after the relation phrase
      - verbal:        the direct object of the relation verb
    Returns the chunk, or None if nothing suitable follows.
    """
    if verb_token is not None:
        for tok in verb_token.children:
            if tok.dep_ in ("dobj", "pobj", "obj"):
                for chunk in chunks:
                    if chunk.root.i == tok.i:
                        return chunk
                return None

    for chunk in chunks:
        if chunk is target_chunk or chunk.root.i <= after_idx:
            continue
        return chunk
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(prompt: str, weight: Optional[float] = None) -> Query:
    """Parse a natural-language tracking prompt into a ``Query``.

    Args:
        prompt: e.g. "red car behind the bus", "moving cars", "cars".
        weight: override the relation's soft weight.  ``None`` uses the
                vocabulary default (``DEFAULT_RELATION_WEIGHT``).

    Returns:
        A ``Query``.  Plain prompts yield ``anchor=None, relation=None``.

    Raises:
        QueryParseError:      no noun chunk to use as the target.
        UnknownRelationError: a preposition or verb modifies the target but
                              maps to no canonical relation (never guessed).
    """
    text = (prompt or "").strip()
    if not text:
        raise QueryParseError("empty prompt")

    doc = _get_nlp()(text)
    chunks = list(doc.noun_chunks)
    if not chunks:
        raise QueryParseError(
            f"no noun chunk found in '{text}' — cannot determine a target class"
        )

    target_chunk = _find_target_chunk(doc, chunks)

    # --- relation: phrase > verb > modifier -------------------------------
    relation_name: Optional[str] = None
    verb_token = None
    consumed: set = set()
    phrase_end = -1

    phrase = _match_phrase(doc)
    if phrase is not None:
        relation_name, phrase_start, phrase_end = phrase
        consumed.update(range(phrase_start, phrase_end + 1))
        # The phrase may hang off a verb ("coming from the counterdirection");
        # that verb is subsumed by the more specific phrase reading.
        for tok in target_chunk.root.children:
            if tok.pos_ == "VERB" and any(c.i in consumed for c in tok.subtree):
                consumed.add(tok.i)

    if relation_name is None:
        found = _find_verb_relation(doc, target_chunk)
        if found is not None:
            relation_name, verb_token = found
            consumed.add(verb_token.i)

    if relation_name is None:
        found = _find_modifier_relation(target_chunk)
        if found is not None:
            relation_name, mod_token = found
            consumed.add(mod_token.i)

    # Anything relational left on the target that we could not map is an error,
    # not a guess.
    _check_unmapped(doc, target_chunk, consumed)

    target = _entity_from_chunk(target_chunk, skip=frozenset(consumed))

    if relation_name is None:
        return Query(target=target, anchor=None, relation=None, text=text)

    spec = RELATIONS[relation_name]
    notes: List[str] = []
    anchor: Optional[Dict[str, Any]] = None

    if spec.arity == "binary":
        after = phrase_end if verb_token is None else verb_token.i
        anchor_chunk = _find_anchor(doc, chunks, target_chunk, after, verb_token)
        if anchor_chunk is None:
            raise QueryParseError(
                f"relation '{relation_name}' is binary but no anchor noun phrase "
                f"was found in '{text}'"
            )
        if _is_ego(anchor_chunk):
            # The anchor is the ego vehicle, not a tracked candidate. Same
            # treatment as the ego-relative relations: flag, don't compute.
            anchor = None
            notes.append(
                f"relation '{relation_name}' is anchored on the ego vehicle "
                f"('{anchor_chunk.text}'); scoring it needs an ego reference "
                f"heading, which is not present in the CARLA GT. Flagged, not computed."
            )
        else:
            anchor = _entity_from_chunk(anchor_chunk, skip=frozenset(consumed))

    if spec.ego_relative:
        notes.append(
            f"relation '{relation_name}' is ego-relative; scoring it needs a "
            f"reference heading for the ego vehicle, which is not present in "
            f"the CARLA GT. Flagged, not computed."
        )

    if relation_name == "between":
        notes.append(
            "relation 'between' is declared binary but semantically takes two "
            "anchors; only the single parsed anchor span is emitted."
        )

    return Query(
        target=target,
        anchor=anchor,
        relation=_make_relation(relation_name, weight),
        notes=notes,
        text=text,
    )


def parse_many(prompts, weight: Optional[float] = None):
    """Parse a list of prompts, yielding (prompt, Query | QueryParseError)."""
    for p in prompts:
        try:
            yield p, parse(p, weight=weight)
        except QueryParseError as exc:
            yield p, exc


if __name__ == "__main__":
    import json
    import sys

    demo = sys.argv[1:] or [
        "cars",
        "red car",
        "red car behind the bus",
        "moving cars",
        "car following the truck",
        "cars coming from the counterdirection",
        "cars in the same direction as ours",
    ]
    for prompt, result in parse_many(demo):
        print("=" * 72)
        print(prompt)
        if isinstance(result, QueryParseError):
            print(f"  ERROR: {result}")
            continue
        print("  " + json.dumps(result.to_dict()))
        for note in result.notes:
            print(f"  note: {note}")

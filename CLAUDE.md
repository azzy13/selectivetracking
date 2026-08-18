# Repo notes

## Environment

Everything runs in the `dino_real` conda env
(`/isis/home/hasana3/miniconda3/envs/dino_real/bin/python`). It is the only
interpreter here that can import `groundingdino` — the prebuilt `_C` extension
links against that env's libtorch.

## Query parser (`eval/query_parser.py`)

Turns a natural-language tracking prompt into a structured `Query` for the
association stage. Rule-based over a spaCy dependency parse — no LLM, no
relation math, no detector/tracker involvement.

```python
from query_parser import parse
parse("red car behind the bus").to_dict()
```

### Schema

```
Query:
  target:   {class: str, attrs: {color, size, count, other}}   # always present
  anchor:   {class: str, attrs: {...}} | None
  relation: {name: str, arity: 'unary'|'binary',
             temporal: bool, weight: float} | None
```

The relation payload carries **exactly** those four keys. Per-relation metadata
(kind, ego-relative flag, scene-graph hint) lives in `query_parser.RELATIONS`
so the query itself can't drift from the contract.

- **Plain prompt** (`"cars"`, `"red car"`) → `anchor=None, relation=None`, i.e.
  classic MOT.
- **Anchor and relation are soft.** `relation.weight` is a score multiplier,
  never a filter. Nothing in the schema encodes an evaluation order — there is
  no "resolve the anchor first, then search near it" structure. The anchor is a
  weighted constraint scored against *all* candidates, like the target's own
  attributes.
- `Query.notes` carries schema-level caveats; `Query.is_plain` and
  `Query.relation_kind` are convenience accessors.

### The three relation kinds (closed vocabulary)

| Kind | Arity | `temporal` | Names |
|---|---|---|---|
| `spatial_instant` | binary | `False` | `behind`, `in_front_of`, `left_of`, `right_of`, `next_to`, `between` |
| `motion_state` | unary | `False` | `moving`, `stationary`, `braking`, `turning`, `counter_direction`, `same_direction` |
| `temporal` | binary | `True` | `following`, `approaching`, `overtaking`, `leading` |

`temporal=True` means the relation needs a **pairwise** cross-frame comparison.
`motion_state` is `False` even though it depends on a track's history, because
`SceneGraphBuilder._motion_attrs` already maintains that per node — no pairwise
history required.

Unary relations always have `anchor=None` even though `relation` is non-None:
`"cars coming from the counterdirection"` → `target={car}`, `anchor=None`,
`relation={counter_direction, unary, temporal:False}`.

A **binary** relation can also yield `anchor=None` when the anchor is the ego
vehicle (`"cars approaching us"`) — ego is not a tracked candidate. Arity stays
`binary`; a note is added.

### Ego-relative predicates — flagged, not computed

`counter_direction` and `same_direction` (and any relation anchored on
us/ours/ego) parse normally but cannot be scored without a reference heading,
which the CARLA GT does not provide. They are flagged in `Query.notes`. The
parser never computes or invents a heading.

### Unknown relations raise

An unmapped preposition or verb modifying the target raises
`UnknownRelationError` rather than guessing. Teach it new phrasings by editing
`PHRASE_RELATIONS`, `VERB_RELATIONS`, or `MODIFIER_RELATIONS` in
`eval/query_parser.py`. Phrase match beats verb match beats modifier match, and
the longest phrase wins (`"in front of"` over a bare `"in"`).

Only relational modifiers of the *target* are validated — a preposition inside
the anchor phrase is left alone. Region prompts (`"cars in the top left"`)
deliberately raise; those belong to `SceneGraphMissionFilter`, not the parser.

### Shared vocabulary

Colors are imported from `scene_graph._COLOR_KEYWORDS` so a parsed color is
always comparable to `node["color"]`. Sizes normalise onto
`SceneGraphBuilder._size_label` outputs (`tiny`/`small`/`medium`/`large`).

Note the naming convention differs by layer: the query vocabulary is snake_case
(`left_of`) while existing scene-graph **edge labels** are hyphenated
(`left-of`). `RelationSpec.scene_graph_hint` records the mapping where one
exists.

### Tests

`tests/test_query_parser.py` — ~32 prompts, each with its full expected parse,
plus schema invariants and the vocabulary table. These are part of the
deliverable: they pin the contract Week 4 codes against.

```bash
python -m pytest tests/test_query_parser.py -v
```

## Query grounding (`eval/query_grounding.py`)

Wires the parsed `Query` into detection and scene-graph construction. Week 3
structural fix — the order changed:

```
before   detect("red car")        -> hard filter -> graph over survivors -> output
after    detect("red car . bus")  -> graph over ALL candidates -> score  -> output
```

The anchor was never detected, so it was never a node, so relational grounding
was impossible. Measured anchor recall on the Week 2 sweep: **0.038 → 1.000**
(`eval/check_anchor_recall.py`, 26 clips, IoU ≥ 0.5).

### Detector prompt

`build_detector_prompt(query)` builds the caption from **both** classes:
`target={red car}, anchor={bus}` → `"red car . bus"`. Plain prompts
(`anchor=None` — no relation, unary, or ego-anchored) stay target-only and are
byte-identical to the pre-Week-3 caption. `count` never reaches the caption;
colour, size and free-form adjectives do.

### Node roles

Every detection, track and graph node carries a role:

| role | meaning |
|---|---|
| `target_candidate` | may be the thing the prompt asks for — the only role ever emitted |
| `anchor` | graph scaffolding: the thing a candidate is related *to* |

- `assign_detection_roles(phrases, query)` — from the phrase GroundingDINO
  grounded each box to (needs `remove_combined=True` so a phrase stays inside
  one caption segment).
- `assign_track_roles(tracks, dets, det_roles, sticky=…)` — carries the role
  through the tracker by best IoU; `sticky` remembers it across a frame where
  the tracker coasts on prediction.
- `SceneGraphBuilder.update(..., roles=…)` writes it to `node["role"]`.
  Omitting `roles` makes every node a `target_candidate` — the pre-Week-3
  behaviour, unchanged.

Ambiguity always resolves to `target_candidate`. A mislabelled anchor costs a
false positive the scorer can down-weight; a mislabelled target is silently
deleted, because:

### Anchors are never emitted

An anchor has no track ID in the output, appears in no result file, and enters
no metric. `emitted_tracks(tracks, roles)` is the single choke point — every
output path goes through it, so the rule has exactly one place it can break.

The one exception is debug rendering: `draw_dotted_rect` draws anchors with a
**dotted orange** outline, target candidates stay **solid green**. Off by
default; enable with `Worker(debug_draw_anchors=True)`.

### Graph over all candidates

`Worker(query=…)` switches the pipeline over: the scene graph is built
unconditionally (it is what candidates are scored over, not an optional export),
the referring filter and colour gate are **not constructed** — the hard filter is
exactly what removed the anchor — and scoring runs over every candidate.

Anchor↔candidate edges are always emitted, even with an empty relation list.
"No relation held this frame" is information; a missing edge is
indistinguishable from a missing node. Candidate↔candidate edges are still
pruned when empty.

### Soft subgraph scorer — STUBBED, Week 4 finishes it

```python
score_candidates(frame_graph, query, *, relation_weight=None) -> {track_id: float}
    # appearance_term + weight * _relation_term(node, frame_graph, query)
    # _relation_term is hard-0.0 today, so this returns detector confidence.
```

Scores **only** target candidates, drops nothing, and returns a weight per
candidate. Week 4 implements `_relation_term`: find the `role == 'anchor'` nodes,
look up the edge to the candidate, and turn the edge's relation labels into a
match score for `query.relation['name']` (note the naming split — query
`left_of` vs edge `left-of`; `RelationSpec.scene_graph_hint` records the
mapping). Ego-relative relations must stay 0.0.

### Validating it

```bash
python eval/check_anchor_recall.py            # Week 2 sweep, 26 clips
python -m pytest tests/test_query_grounding.py -v
```

`notebooks_debug/02_evaluate_grounding.ipynb` runs the same comparison with the
qualitative sheet (dotted anchors visible). Its selection metrics rank only
*emitted* detections — a bus box outranking every car is scaffolding, not an
answer.

### Benchmark prompt report (`eval/parse_benchmark_prompts.py`)

Runs the parser over the `task` string of every episode in a checkout of
`example_benchmark_episodes` — reading the JSON straight out of the release
`.zip` files — and prints a human-readable report: each distinct prompt, a
plain-English read-back of the parse, the structured query, notes, and
report-only sanity flags for parses that succeed but are clearly wrong.

```bash
python eval/parse_benchmark_prompts.py                 # all releases
python eval/parse_benchmark_prompts.py --release v3 --failures
python eval/parse_benchmark_prompts.py --prompt "red car behind the bus"
python eval/parse_benchmark_prompts.py --json          # machine-readable
```

The sanity flags are heuristics local to this script, not parser behaviour:
target class is a bare colour word, anchor class is a time noun, target class is
a non-discriminative noun.

### Dependencies

`spacy` + `en_core_web_sm` (`python -m spacy download en_core_web_sm`), and
`pytest` to run the tests.

spaCy itself is in `dino_real`, but the `en_core_web_sm` **model** resolves via
the user site-packages (`~/.local`), so `PYTHONNOUSERSITE=1` — which `conda run`
sets — makes `spacy.load()` fail. Run the parser without it.

The `python3` kernelspec that `jupyter` picks up by default is
`/usr/share/jupyter/kernels/python3` → `/usr/bin/python3`, which has neither
spaCy nor torch. Notebooks must run on the `dino_real` kernel.

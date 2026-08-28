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

### Soft subgraph scorer

```python
score_candidates(frame_graph, query, *, relation_weight=None) -> {track_id: float}
    # appearance_term + weight * _relation_term(node, frame_graph, query)
```

Scores **only** target candidates, drops nothing, and returns a weight per
candidate. `_relation_term` is 1.0 when the graph put the queried relation on an
edge between the candidate and a `role == 'anchor'` node, else 0.0 — binary,
because the graph hands back labels, not margins. `_EDGE_FOR_RELATION` maps the
query vocabulary onto edge labels (`in_front_of` → `in-front-of`), and
`_EDGE_INVERSE` reads an edge backwards when it was stored anchor-first; edges
are stored once per pair in node order, so half of them need inverting.

It returns 0.0 without consulting the graph when the relation cannot be scored:
`query.anchor is None` (which covers **every** ego case — the parser nulls the
anchor both for the unary ego-relative relations and for a binary relation
anchored on the ego vehicle), a relation with no scene-graph equivalent
(`between` and the four temporal relations), or no anchor node in this frame.

### The depth relation (`scene_graph._depth_relation`)

`behind` / `in-front-of` are the only edges computed in two different frames,
because prompts use both and they are **not** the same thing:

- **The reference node has a heading** (`heading_vec != [0,0]`) — object-centric.
  The candidate's offset is projected onto the reference's own direction of
  travel, so "behind the bus" means trailing the bus down the road. This is what
  every object-anchored benchmark prompt means, and it is not recoverable from
  camera depth: a car ahead of the bus can be *further* from the camera.
- **No heading** (stationary, or fewer than 3 frames of history) — viewer-centric
  fallback on vertical image position, since a nearer object sits lower on a
  ground plane. It deliberately does **not** compare areas: the candidate and
  the anchor are usually different classes, and a car in front of a bus is still
  the smaller box.

`DEPTH_THRESH = 0.02` is a deadband, smaller than `SPATIAL_THRESH` because
offsets along the view/travel axis are foreshortened. Both branches work in
units of image **width**: `cx_norm` divides by width and `cy_norm` by height, so
the normalised space is stretched by the aspect ratio and a raw dot product
there over-weights the vertical term by its square (3.16x on 16:9). Left
uncorrected, `DEPTH_THRESH` would mean one thing for a road running up the frame
and another for one running across it. (`SPATIAL_THRESH` has the same
anisotropy; that is pre-existing and untouched.)

Because depth is measured in the **reference** node's frame, it is the one
relation that cannot be recovered by swapping subject and object: "the bus is in
front of the car, in the car's frame" implies nothing about where the car is
relative to the bus when the two face opposite ways. So `update()` orients every
anchor-spanning edge candidate→anchor, and `_relation_term` **drops** rather
than inverts `behind`/`in-front-of` when reading an edge backwards
(`_FRAME_DEPENDENT_EDGES`). Getting this wrong is not subtle — it made the
oncoming distractor outrank the target on all 13 positive sweep clips.

### Headings for single-frame clips (`eval/sweep_headings.py`)

`_motion_attrs` needs three frames of history, and a sweep clip is **one**
frame, so every `heading_vec` there is `[0,0]` and only the viewer-centric
fallback runs — which on the sweep is actively wrong, because the distractor is
further from the camera than the bus and so reads as "behind" it.

The GT already carries what is missing: every `gt_graphs` node has a world-space
`yaw`. `sweep_headings.py` projects it into image space through the CARLA camera
transform (`rotation_matrix` / `project` / `heading_vec`) and
`headings_for_tracks` matches it onto tracks by IoU, ready for
`update(heading_override=...)`. The camera model is exact: all 78 projected GT
`loc` land in the lower half of their own `box2d`.

End-to-end through the real pipeline, scored against the **corrected** key (the
`front` clips are the positives — see the relabel note):

| | |
|---|---|
| viewer-centric fallback | 0/26 |
| GT heading injected | **26/26** |

This is a **harness, not a pipeline feature**. It measures whether the relation
logic is right given a correct heading; it says nothing about how well headings
are recovered from tracker output, which is what a live run depends on. Read it
as an upper bound, and validate on a CARLA sequence where headings come from
motion.

### Answer selection (`select_answers`) — a threshold, not an argmax

Scoring ranks; selection decides. `score_candidates` still drops nothing, and
`emitted_tracks` is still the anchor choke point — selection runs *after* both,
on candidates only.

The policy, in order:

1. `relation_holds()` returns `None` — the relation cannot be judged (plain
   prompt, ego-anchored or unary relation, no scene-graph equivalent, or the
   anchor was not detected this frame) → **every candidate passes through**.
   Grounding must never turn a missing feature into a confident empty answer.
2. Otherwise keep the candidates the relation held for. This can legitimately be
   empty: "nothing here is behind the bus" is an answer, not a failure.
3. If the prompt named a cardinality (`"the two cars behind the bus"` →
   `count=2`) and more survived, keep the highest-scoring `count`.

Not an argmax, because `"cars behind the bus"` is plural and must be able to
return several. Top-1 would collapse that case and make the singular case right
for the wrong reason. Note the parser only sets `count` when the prompt states a
number — bare singular and plural both give `None`, so nothing infers arity from
grammatical number.

On by default under `--grounded`; `--no-answer_selection` restores the
pre-selection behaviour (every candidate emitted) for comparison. End-to-end on
the sweep with GT headings, selection emits exactly the right set in **26/26**
clips — the target in each `front` clip, nothing in each `behind` clip.

### Validating it

```bash
python eval/check_anchor_recall.py            # Week 2 sweep, 26 clips
python -m pytest tests/test_query_grounding.py tests/test_depth_relation.py -v
```

`notebooks_debug/02_evaluate_grounding.ipynb` runs the same comparison with the
qualitative sheet (dotted anchors visible). Its selection metrics rank only
*emitted* detections — a bus box outranking every car is scaffolding, not an
answer.

### Running it (`eval/eval_carla.py --grounded`)

`--grounded` is the only driver that reaches the grounded path. Without it the
`Worker` is built with `query=None` and behaves exactly as it did pre-Week-3.

```bash
python eval/eval_carla.py \
    --carla_scenarios dataset/carla_eval/eval_scenarios \
    --scenarios follow_base \
    --grounded --text_prompt "red car behind the bus" \
    --fp16 --devices 0,1 --save_video --debug_draw_anchors
```

Every scenario's prompt is parsed by `validate_queries()` **before any model
loads**, and the run aborts if one fails. It never falls back to plain MOT — a
silent fallback produces output that looks grounded but is not. `--grounded`
requires `--worker clean`; `WorkerSimple` has no query path.

`--debug_draw_anchors` only affects `--save_video` frames. Anchors still reach
no results file, because every output path goes through `emitted_tracks`.

Note the prompt-compliance metrics at the end of the run read the prompt from
`gt.json`, not from `--text_prompt`, so the header prints the scenario's own
prompt even when you overrode it. Pre-existing behaviour, unrelated to grounding.

### The pipeline notebook (`notebooks_debug/03_scene_graph_pipeline.ipynb`)

Runs the same `Worker` over one `dataset/sweep` frame at a time, stopping after
every stage: parse → detect+roles → track → graph → score. Renders the frame
with role-coloured boxes and the frame graph as an actual Graphviz graph, and
matches every node back to the sweep GT so a score can be read as "the
distractor outranked the target" rather than as a number.

`CLIP_INDEX` selects the clip (index 1 = the sweep's 2nd frame, `cfg00/front`);
section 8 renders the matched `behind`/`front` pair, section 9 sweeps all 26.

Thresholds differ from the `eval_carla.py` defaults on purpose: a sweep clip is
**one** frame, so a detection that misses `track_thresh` never becomes a track
at all. `box=0.25, text=0.25, track_thresh=0.20` recovers all three GT objects;
the eval defaults (`0.40 / 0.80 / 0.45`) silently drop the distractor.

What it shows on the Week 2 sweep, 26 clips, confidence-only scoring:

| | |
|---|---|
| anchor (bus) detected | 1.000 |
| GT target present in the graph | 0.962 — the ceiling for any scorer |
| `behind`: top candidate is the GT target | **0.154** (chance is 0.500) |
| `front`: emitted something anyway | 1.000 — should be 0.000 |

0.154 is *below* chance and that is the point: in the `behind` condition the
target is the further/occluded car, so it reliably scores **lower** than the
distractor. Detector confidence is anti-correlated with the answer here, so
`_relation_term` is not a refinement — without it the ranking is actively wrong.

**Kernel.** The notebook pins kernelspec `dino_real`
(`~/.local/share/jupyter/kernels/dino_real`, absolute interpreter). The plain
`python3` kernelspec resolves to `/usr/bin/python3` on this machine even when
`jupyter kernelspec list` points at the env, and that interpreter has neither
spaCy nor CLIP — while `import groundingdino` still *succeeds* from the repo
root, so the obvious guard does not catch it. Cell 0 checks `torch`,
`groundingdino._C`, `spacy` and `clip`, in that order (`_C` cannot load before
libtorch). Re-register with:

```bash
/isis/home/hasana3/miniconda3/envs/dino_real/bin/python \
    -m ipykernel install --user --name dino_real --display-name "Python 3 (dino_real)"
```

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

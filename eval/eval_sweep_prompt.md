# The relational sweep under SP / SR / DCR

Scores `dataset/sweep` — the 26-clip Week-2 relational benchmark — with the
same metric implementation that scores Refer-KITTI and CARLA
(`carla_sim/evaluate_prompt_metrics.py`).

```bash
python eval/eval_sweep_prompt.py --grounded     # query grounding + answer selection
python eval/eval_sweep_prompt.py --plain        # pre-Week-3 path, for contrast
python eval/eval_sweep_prompt.py --grounded --prompt "red car behind the bus"
```

Read **SP, SR and DCR**. PCR and SID are not reported: every clip is one frame,
so PCR collapses to 0/1 and SID has no time axis to switch along.

## Why this dataset and not the CARLA eval scenarios

The sweep's distractor is matched to the target on every attribute the pipeline
can perceive — same class (`car`), same colour (`180,20,20`), similar size —
and differs *only* in the relation and its heading. That is what makes it a
relational benchmark. Measured elsewhere in this repo, the non-relational cues
are strong enough to solve a prompt on their own (spatial position AUC 0.97,
colour 0.71–0.86), so a benchmark whose distractor differs in colour would be
passed by a colour filter with no relational reasoning at all.

The sweep GT is also richer than `dataset/carla_eval/eval_scenarios/*/gt.json`:
it carries explicit roles (`ids: {bus, target, distractor}`), per-node `yaw` and
world `loc`, per-clip camera pose, and — the important one — **ground-truth
relation edges** (`edges: [{subj, relation, obj}]`).

## The answer key is in the edges, not the folder name

Clips are named `behind` / `front` and `manifest.json` carries one `prompt`
("red car behind the bus"). Neither is a usable key. Derived from the GT edges:

| prompt | `behind` clips | `front` clips |
|---|---|---|
| red car **behind** the bus | both cars valid ×11, target only ×1, none ×1 | distractor only ×11, none ×2 |
| red car **in front of** the bus | none valid ×12, distractor only ×1 | target only ×12, distractor only ×1 |

**The manifest's own prompt is the degenerate direction.** In the `behind`
geometry both red cars end up behind the bus, so "red car behind the bus" has
two valid answers in 11 of 13 clips and cannot discriminate; and in the `front`
clips its correct answer is the *distractor*, not "nothing". Only `in_front_of`
gives a clean single-answer key — which is what the "corrected key" in
`DOC.md` is really describing.

`sweep_prompt_gt.py` therefore takes the relation from the prompt and reads
validity out of the GT edges per clip. A folder named `front` gets no special
treatment. `answer_key_summary()` reports the 0 / 1 / many split so a degenerate
prompt direction is visible before any model runs.

## Results — "red car in front of the bus", 26 clips

Default thresholds (`box=0.25, text=0.25, track_thresh=0.20`), SwinB, bytetrack.
GT: 14 valid boxes, 64 non-valid (13 distractors + 26 buses + the rest).

| | SP | SR | DCR | exact-answer clips |
|---|---:|---:|---:|---:|
| `--plain` (pre-Week-3) | 0.222 | 1.000 | **0.778** | **0/26** |
| `--grounded` | **0.933** | 1.000 | **0.000** | **25/26** |

Where the predictions went:

| | valid | distractor | matched nothing | total |
|---|---:|---:|---:|---:|
| plain | 14 | **49** | 0 | 63 |
| grounded | 14 | **0** | 1 | 15 |

Both modes find every valid target (SR 1.000) — detection is not the
differentiator here, which is exactly the design intent. The difference is
entirely in what else gets emitted: the plain path emits 49 distractor boxes
against 14 correct ones, so **78% of its output is the wrong red car or the
bus**. Grounding emits 15 boxes total and none of them is a distractor.

"Exact-answer clips" counts clips where the emitted set was precisely the
correct set, *including the empty set*. The plain path scores 0/26 because it
has no mechanism to return nothing — it emits every red car it finds in all 26
clips, so it is wrong in every clip where the answer is "none" and wrong by
over-emission in the rest. This is the `select_answers` threshold-not-argmax
policy earning its keep.

The single grounded failure is `cfg02_behind`: the answer is the empty set and
it emitted one box, which matched no GT object at all (a spurious detection, not
a distractor confusion).

## The degenerate direction, as a control

Running the manifest's own prompt through the same path:

| prompt | SP | SR | DCR | exact |
|---|---:|---:|---:|---:|
| in front of the bus | 0.933 | 1.000 | 0.000 | 25/26 |
| behind the bus | 0.923 | **0.706** | 0.000 | **15/26** |

Precision holds but recall drops to 0.706 and exact-answer clips nearly halve.
That is the benchmark, not the method: with 11 clips having two valid answers,
the pipeline emits one of the two and is scored as missing the other. Quote the
`in_front_of` numbers; the `behind` direction measures the dataset's ambiguity
more than the model's grounding.

## What this does not show

Every clip is a **single frame**, so `_motion_attrs` has no history and every
`heading_vec` is `[0, 0]` — only the viewer-centric fallback in
`_depth_relation` ever runs. These numbers therefore say the relation logic and
the selection policy are right *given the detections*; they say nothing about
recovering headings from tracker output, which is what a live multi-frame run
depends on. `eval/sweep_headings.py` injects GT headings to separate those two
questions, and that path is not wired into this runner.

The 16-scenario CARLA sweep, at 30 s per scenario, is what tests the heading
recovery this cannot.

## Note on Worker reuse

`Worker.reset_sequence_state()` was added while building this. The tracker is
constructed once in `__init__` and carries track ids, buffers and Kalman state,
so two sequences processed by one Worker would have the second inherit the
first's tracks. Every existing caller builds a fresh Worker per sequence, so no
result changes; but this runner reuses one Worker across all 26 clips (the model
load is ~5.5 s, which would otherwise dominate), and that is only safe with the
reset. It is now called automatically at the top of `process_sequence`.

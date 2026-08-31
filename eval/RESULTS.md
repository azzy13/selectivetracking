# Results — prompt-compliance metrics (SP / SR / DCR)

Every measured SP / SR / DCR number, in one place, with the command that
produced it. Metric definitions: `carla_sim/metrics.md`. One shared
implementation (`carla_sim/evaluate_prompt_metrics.py`) scores all three
datasets, so rows here are comparable across them.

- **SP** — predictions matched to prompt-valid GT ÷ all predictions
- **SR** — prompt-valid GT matched ÷ all prompt-valid GT
- **DCR** — predictions matched to prompt-*invalid* GT ÷ all predictions
- **PCR** — frames where a valid target was held ÷ frames where one was visible
- **SID** — times a track flipped between matching a valid and an invalid object

SP, DCR and the unmatched share partition the predictions, so
`1 − SP − DCR` is the fraction that hit no GT box at all.

All runs: GroundingDINO SwinB, IoU ≥ 0.5, Hungarian matching, fp16.

---

## 1. Relational sweep — `dataset/sweep`, 26 clips

The only dataset here built for relational grounding: the distractor matches the
target on class, colour and rough size, differing only in the relation. 14
prompt-valid boxes; 12 of 26 clips have no valid answer.

Single-frame clips, so **PCR and SID are not meaningful** and are omitted.

### Pre-Week-3 vs now

| path | SP | SR | DCR | exact-answer clips |
|---|---:|---:|---:|---:|
| plain (pre-Week-3) | 0.222 | 1.000 | 0.778 | 0 / 26 |
| **grounded + selection** | **0.933** | **1.000** | **0.000** | **25 / 26** |

| predictions landed on | plain | grounded |
|---|---:|---:|
| a prompt-valid object | 14 | 14 |
| a distractor object | 49 | **0** |
| no GT box at all | 0 | 1 |
| **total emitted** | **63** | **15** |

Recall is 1.000 in both — detection is not the differentiator, by design. The
difference is entirely in what else is emitted. "Exact-answer clips" counts
clips where the emitted set was precisely correct, *including the empty set*;
the plain path scores 0 because it has no mechanism to return nothing.

### Control — the degenerate prompt direction

| prompt (grounded) | SP | SR | DCR | exact | clips with >1 valid answer |
|---|---:|---:|---:|---:|---:|
| red car **in front of** the bus | 0.933 | 1.000 | 0.000 | 25 / 26 | 0 |
| red car **behind** the bus | 0.923 | 0.706 | 0.000 | 15 / 26 | 11 |

Quote the `in_front_of` row. In the `behind` geometry both red cars end up
behind the bus, so 11 of 13 clips have two valid answers — the drop measures
the benchmark's ambiguity, not the model. See `eval/eval_sweep_prompt.md`.

```bash
python eval/eval_sweep_prompt.py --plain
python eval/eval_sweep_prompt.py --grounded
python eval/eval_sweep_prompt.py --grounded --prompt "red car behind the bus"
```

Config: `box=0.25 text=0.25 track_thresh=0.20`, bytetrack,
`weights/groundingdino_swinb_cogcoor.pth`.
Raw: `outputs/sweep_prompt_{plain,grounded,grounded_behind}/summary.json`

---

## 2. Refer-KITTI — sequence 0005, all 50 expressions

Complete. Tuned Trial-532 config (`eval/eval_referkitti_prompt.py` defaults),
`weights/swinb_light_visdrone_ft_best.pth`.

| | SP | SR | DCR | PCR | SID |
|---|---:|---:|---:|---:|---:|
| macro (per expression) | 0.307 | 0.425 | 0.213 | 0.523 | — |
| micro (pooled counts) | 0.306 | 0.502 | 0.281 | 0.624 | 32 |

24,083 predictions: 7,359 on a valid object · 6,756 on a distractor ·
9,968 on no GT box. 7,359 of 14,646 prompt-valid GT boxes matched.

### By attribute stratum

Mutually exclusive buckets, tagged with the pipeline's own vocabularies, so a
stratum means "the code can see this cue".

| stratum | expressions | SP | SR | DCR | PCR |
|---|---:|---:|---:|---:|---:|
| plain | 2 | 0.259 | 0.990 | 0.469 | 0.990 |
| spatial only | 4 | 0.291 | 0.291 | 0.186 | 0.442 |
| colour only | 6 | **0.442** | 0.485 | 0.244 | 0.635 |
| spatial + colour | 20 | 0.295 | **0.261** | 0.162 | **0.315** |
| motion + other | 10 | 0.251 | 0.324 | 0.205 | 0.457 |
| motion only | 8 | 0.322 | **0.839** | 0.273 | **0.967** |
| all except motion-only | 42 | 0.304 | 0.346 | 0.201 | 0.439 |
| **all** | **50** | **0.307** | **0.425** | **0.213** | **0.523** |

Two things this table exists to prevent being misread:

- **`motion only` is not a strength.** No filter acts on motion prompts, so the
  pipeline emits every car it finds — which is why SR (0.839) and PCR (0.967)
  are the highest in the table. Dropping these expressions *lowers* the headline
  (SR 0.425 → 0.346), because they inflate recall by doing nothing. Measured
  directly: on motion-only prompts the pipeline emits 0.81×–2.68× one prediction
  per annotated object in the clip, regardless of the motion word.
- **`spatial + colour` is the worst stratum and the largest.** Stacking two hard
  gates compounds their recall loss — worse than colour alone (SP 0.442).

```bash
python eval/eval_referkitti_prompt.py --sequence 0005 --all_expressions --fp16
```

Raw: `outputs/referkitti_0005_complete.json`

---

## 3. Refer-KITTI — the colour-filter repair

Controlled before/after on three expressions of 0005, identical config, only
the colour classifier changed. Not a benchmark result — an ablation.

| macro over 3 expressions | HSV rule | CIELAB + peer rank |
|---|---:|---:|
| SP | 0.353 | **0.587** |
| SR | 0.441 | **0.683** |
| PCR | 0.576 | **0.907** |
| DCR | 0.173 | 0.231 |

| per expression | SP | SR | DCR | PCR | preds |
|---|---:|---:|---:|---:|---:|
| `cars-in-left` — before | 0.598 | 0.460 | 0.230 | 0.729 | 570 |
| `cars-in-left` — after | 0.598 | 0.460 | 0.230 | 0.729 | 570 |
| `cars-in-black` — before | 0.000 | 0.000 | 0.000 | 0.000 | **0** |
| `cars-in-black` — after | **0.703** | **0.727** | 0.175 | **0.993** | 647 |
| `moving-cars` — before | 0.460 | 0.864 | 0.289 | 1.000 | 1104 |
| `moving-cars` — after | 0.460 | 0.864 | 0.289 | 1.000 | 1104 |

**The DCR rise is an artefact of the before column, not a regression.** An
expression emitting zero predictions contributed DCR = 0.000 to the average. The
two expressions that were actually running are unchanged in both columns.

Cause: the HSV rule gated on `s_val >= 35` before testing brightness, and HSV
saturation `(max−min)/max` is undefined as value approaches zero. Measured over
215,720 pixels inside black-car GT boxes, 82.4% were routed to the hue branch;
not one patch in ~1,056 came out `dark`. Across every colour expression in the
dataset it returned `blue` or `green` for every crop of every colour — no signal
at all. 346 of 818 expression files (42%) contain a colour word.

Raw: `outputs/referkitti_prompt_0005_x3/` (before),
`outputs/referkitti_prompt_0005_x3_labcolor/` (after)

---

## 4. Perception-cue calibration

Not SP/SR — these measure how much signal each attribute cue carries, against
each dataset's own labels. They explain the strata above.

| cue | AUC | balanced accuracy | harness |
|---|---:|---:|---|
| spatial — "left" | 0.970 | — | ad hoc, GT box centres |
| spatial — "right" | 0.948 | — | ad hoc, GT box centres |
| colour — light | 0.831 | 0.786 | `check_color_classifier.py` |
| colour — silver | 0.727 | 0.630 | `check_color_classifier.py` |
| colour — black | 0.714 | 0.669 | `check_color_classifier.py` |
| colour — red | 0.625 | abstains | `check_color_classifier.py` |
| motion — moving | 0.631 | 0.589 | `check_motion_classifier.py` |
| motion — stationary | 0.585 | 0.552 | `check_motion_classifier.py` |
| motion — raw image displacement | **0.498** | — | the naive cue; no signal |

Balanced accuracy (mean of TPR and TNR over committed crops) rather than raw
accuracy: these sets run 30–42% positive, so a rule that abstains into the
majority class flatters raw accuracy.

Motion is measured on **ground-truth** boxes and is therefore an upper bound.
0.498 for raw image displacement is the reason `_motion_attrs` is not wired in:
Refer-KITTI is filmed from a moving car, so image displacement measures
ego-motion. The motion gate is off by default.

```bash
python eval/check_color_classifier.py
python eval/check_color_classifier.py --no_peers    # absolute-threshold contrast
python eval/check_motion_classifier.py
```

---

## Not yet measured

- **Refer-KITTI full test split (0005 / 0011 / 0013, 158 expressions).** Started,
  stopped at 64/158 — 0005 complete, 0011 partial (16 of 64). Resume with
  `--test_split --all_expressions --resume` against
  `outputs/referkitti_prompt_testsplit_baseline`. This is the number comparable
  to the ByteTrack / FairMOT baselines in `dataset/referkitti/*_results`, which
  are scored on exactly these three sequences.
- **CARLA 16-scenario relational sweep.** GT schema extension pending — see
  `carla_sim/GT_SCHEMA_TASK.md`.
- **Sweep with GT headings injected.** `eval/sweep_headings.py` exists but is not
  wired into `eval_sweep_prompt.py`. Every sweep clip is one frame, so all
  headings are `[0,0]` and only the viewer-centric depth fallback runs — the
  section-1 numbers validate relation logic *given detections*, not heading
  recovery from tracker output.
- **`DontCare` handling.** 18% of unmatched predictions on `cars-in-left` land in
  KITTI `DontCare` regions and are currently scored as false positives, so every
  Refer-KITTI SP above is an underestimate.

# Refer-KITTI under the prompt-compliance metrics (SP / SR / DCR)

Evaluates the pipeline on Refer-KITTI with the metrics from `carla_sim/metrics.md`
— Semantic Precision, Semantic Recall, Prompt Coverage Ratio, Distractor
Confusion Rate, Semantic ID Switches — instead of MOTA/IDF1.

```bash
# solo test: one sequence, one expression
python eval/eval_referkitti_prompt.py --sequence 0005 --expression cars-in-left --fp16

# what expressions does a sequence have, and how much GT do they carry?
python eval/eval_referkitti_prompt.py --sequence 0005 --list

# re-score results already on disk (no model load)
python eval/eval_referkitti_prompt.py --sequence 0005 --expression cars-in-left \
    --outdir outputs/referkitti_prompt_solo_0005 --skip_run
```

Everything runs in the `dino_real` env
(`/isis/home/hasana3/miniconda3/envs/dino_real/bin/python`).

## Why not just MOTA

`eval_referkitti.py` builds its ground truth from the *referred objects only*.
Every other car in the frame is invisible to it, so a prediction that locks onto
a parked car nobody asked about and a prediction that fires on empty road are
both just false positives — indistinguishable in the score.

That is the error mode a referring tracker actually has. This script keeps the
whole frame in the GT and flags each box prompt-valid or not, which splits the
same predictions three ways:

| landed on | metric |
|---|---|
| a prompt-valid object | **SP** |
| a distractor (a real object the prompt excludes) | **DCR** |
| no GT box at all | `1 − SP − DCR` |

A high DCR and a high unmatched share call for different fixes — a better
referring filter versus a better detector — and MOTA cannot tell them apart.

## The pieces

| file | role |
|---|---|
| `eval/referkitti_prompt_gt.py` | joins Refer-KITTI into the CARLA GT schema |
| `eval/referkitti_prompt_report.py` | human-readable console + markdown report |
| `eval/eval_referkitti_prompt.py` | runner: select → track → score → report |
| `carla_sim/evaluate_prompt_metrics.py` | **the metrics themselves — not reimplemented** |
| `eval/color_classifier.py` | CIELAB colour classification, peer-relative |
| `eval/check_color_classifier.py` | calibration check against a dataset's colour labels |
| `eval/motion_classifier.py` | ego-compensated motion state (off by default) |
| `eval/check_motion_classifier.py` | calibration check for the motion cue |
| `tests/test_referkitti_prompt_gt.py` | 12 tests pinning the adapter |
| `tests/test_color_classifier.py` | 37 tests pinning the colour classifier |
| `tests/test_motion_classifier.py` | 23 tests pinning the motion classifier |

The metric code is imported from `carla_sim/`, the same hop `eval_carla.py`
uses. A Refer-KITTI number and a CARLA number therefore come out of one
implementation, which is the point: the two results have to be comparable when
the relational CARLA sweep lands.

## The join

Refer-KITTI already holds per-frame prompt validity, split across two files:

```
KITTI/training/labels_with_ids/image_02/<seq>/<frame>.txt   every object
expression/<seq>/<expr>.json  "label": {frame: [ids]}       which ones are referred
```

`build_prompt_gt` joins them into `{"meta": {...}, "annotations": [...]}` where
each annotation carries `image_id`, `gt_id`, `track_id`, `bbox_xyxy` and
`is_target`. Frame ids line up untouched: the label-map keys, the
`labels_with_ids` filenames, and the frame ids `worker_clean.parse_frame_id`
writes into the MOT output are all the same number.

Three things to know about it:

- **Validity is per frame.** `eval_referkitti.py` unions the referred ids over
  the whole clip and treats them as valid everywhere. That is looser than the
  annotation — an object referred to only in frames 10–38 would score as valid
  in frame 200. This adapter reads the `label` map frame by frame, which is
  what "prompt-valid" means in the metrics doc. `test_validity_is_per_frame_not_per_sequence`
  pins it, and asserts the distinction is actually observable in the test clip
  so it cannot pass vacuously.

- **Boxes are top-left, not centre.** Despite the YOLO-looking layout, the
  `labels_with_ids` columns are `x_left y_top w h` normalised by image size.
  `eval_referkitti.py` carries the same note. Reading them as centre
  coordinates shifts every box by half its size and quietly halves every IoU.

- **Only annotated frames are scored** by default. An expression's `label` map
  covers a contiguous span that can be shorter than the clip (284 of 297 frames
  for `0005`); frames outside it are *unannotated*, not "annotated as empty", so
  scoring predictions there charges the tracker for GT that was never written.
  `--score_whole_clip` takes the other reading.

`ignore` is empty in all 818 expression files of this release, so no
ignore-region handling exists. A non-empty one raises `NotImplementedError`
rather than being silently scored as a distractor.

## Validating the adapter

The two fixed points, run through the real metric code:

```
predict exactly the valid boxes    ->  SP = SR = PCR = 1.000,  DCR = 0.000
predict exactly the distractors    ->  SP = SR = PCR = 0.000,  DCR = 1.000
```

An oracle that scores 1.000 is the cheapest available proof that the frame
alignment and the coordinate convention are both right — if either were wrong,
no synthesised prediction would match its own GT box.

```bash
python -m pytest tests/test_referkitti_prompt_gt.py -v      # 12 tests
```

## Solo test — sequence 0005, "cars in left"

Tuned Trial-532 configuration (the script's defaults), `swinb_light_visdrone_ft_best.pth`,
IoU ≥ 0.5, 284 annotated frames, 742 prompt-valid and 457 distractor GT boxes.

| metric | value | counts |
|---|---:|---|
| **SP** Semantic Precision | 0.598 | 341 / 570 predictions on a valid target |
| **SR** Semantic Recall | 0.460 | 341 / 742 valid GT boxes found |
| **DCR** Distractor Confusion Rate | 0.230 | 131 / 570 predictions on a distractor |
| PCR Prompt Coverage Ratio | 0.729 | 207 / 284 frames with the target held |
| SID Semantic ID Switches | 1 | |

Where the 570 predictions went: **341 valid · 131 distractor · 98 matched nothing**.

Read together: recall is the weak number — over half the valid boxes are never
found, and the tracker holds the target in only 73% of frames. Of the errors it
does make, distractors (23%) outnumber hallucinations (17%), so the larger share
of the precision loss is *semantic* — real cars that the prompt excludes — not
detector noise. That is the split MOTA collapses.

Reproduce:

```bash
python eval/eval_referkitti_prompt.py --sequence 0005 --expression cars-in-left \
    --outdir outputs/referkitti_prompt_solo_0005 --device 0 --fp16
```

Outputs land in the run directory as `report.md` (the markdown above),
`metrics.json` (config + summary + per-expression rows), and
`metrics_<seq>_<expr>.json` (per-frame stats and SID events).

## Three expressions — and a bug the metrics found

Running `cars-in-left`, `cars-in-black` and `moving-cars` on sequence 0005
first produced this:

| expression | prompt | SP | SR | DCR | PCR | SID | preds |
|---|---|---:|---:|---:|---:|---:|---:|
| `cars-in-left` | cars in left | 0.598 | 0.460 | 0.230 | 0.729 | 1 | 570 |
| `cars-in-black` | cars in black | 0.000 | 0.000 | 0.000 | 0.000 | 0 | **0** |
| `moving-cars` | moving cars | 0.460 | 0.864 | 0.289 | 1.000 | 0 | 1104 |

`cars-in-black` emitted **zero** predictions over the whole clip — the
post-track colour gate confirmed 0 of 3 tracks in every frame, every colour
score exactly `0.0`.

### Root cause: HSV saturation is undefined where it was being tested

`_get_patchwise_dominant_color` classified a pixel as chromatic whenever
`s_val >= 35`, and only low-saturation pixels reached the brightness branch
that can return `'dark'`. But HSV saturation is `(max - min) / max`, so it is
numerically meaningless as value goes to zero: RGB (20, 22, 28) reads S = 29%
and gets a hue assigned. Measured over 215,720 pixels inside the GT boxes of
black-annotated cars in sequence 0005, **82.4%** were routed to the hue branch
that way; 70.3% were genuinely dark (`V < 80`) but only 11.5% could ever be
labelled so. Of ~1,056 patches sampled, **not one** came out `'dark'`.

Checked across every colour expression in the dataset, the old rule returned
`blue` or `green` for *every* crop of *every* colour — black, silver, light,
white and red alike. It was not strict, it was **non-discriminative**: it
carried no signal at all and simply deleted every track on a colour prompt.
That is 346 of 818 expression files (42%).

### The fix: CIELAB, decided on rank within the frame

`eval/color_classifier.py` replaces it. Two separate changes, and the
distinction matters:

1. **CIELAB instead of HSV** — chroma `C* = sqrt(a*² + b*²)` is an absolute
   measure of colourfulness rather than a ratio, so it stays small for dark
   pixels instead of blowing up. This is a *correctness* fix; it transfers to
   any dataset because it repairs the mathematics, not the numbers.

2. **Rank within the frame instead of an absolute `L*` cut** — this is the part
   that would otherwise be a KITTI-specific calibration. A crop in the darkest
   third of the frame's own detections supports "black"; one in the lightest
   third is evidence against. A rank has no units, so a brighter or darker
   render moves every candidate together and the decision is unchanged. Below
   three detections there is no ordering worth reading and the absolute rule is
   used instead.

Balanced accuracy on Refer-KITTI's own colour labels (mean of TPR and TNR over
committed crops — these sets run 30–42% positive, so raw accuracy would reward
abstaining into the majority class):

| target | n pos | AUC | absolute `L*` | **peer rank** |
|---|---:|---:|---:|---:|
| black | 244 | 0.714 | 0.525 | **0.669** |
| silver | 105 | 0.727 | 0.609 | **0.630** |
| light | 187 | 0.831 | 0.728 | **0.786** |
| white | 17 | 0.863 | **0.886** | 0.776 |
| red | 61 | 0.625 | — abstains — | — abstains — |

Two findings worth keeping:

- **Silver is a lightness phenomenon here, not a colour one.** Scored as
  "mid-lightness" it gets 0.384 balanced accuracy — worse than guessing. Scored
  as "brighter than peers", 0.754. Chroma does not separate silver at all
  (AUC 0.386; median chroma 4.2 against 4.0 for everything else). The
  consequence: the classifier cannot tell silver from white, and does not
  pretend to.

- **Red does not work on this imagery and now says so.** Zero percent of crops
  inside red-annotated boxes reach the chroma threshold (median 4.5 against 4.0
  for non-red). Before the chroma guard it scored TPR 1.000 / TNR 0.000 — it
  confirmed every distractor it committed on, off two or three noisy patches.
  It now abstains 100% of the time, which is the honest answer. Red is 36 of
  818 expression files.

### After the fix — same config, same three expressions

| expression | prompt | SP | SR | DCR | PCR | SID | preds |
|---|---|---:|---:|---:|---:|---:|---:|
| `cars-in-left` | cars in left | 0.598 | 0.460 | 0.230 | 0.729 | 1 | 570 |
| `cars-in-black` | cars in black | **0.703** | **0.727** | **0.175** | **0.993** | 3 | 647 |
| `moving-cars` | moving cars | 0.460 | 0.864 | 0.289 | 1.000 | 0 | 1104 |

| macro | before | after |
|---|---:|---:|
| SP | 0.353 | **0.587** |
| SR | 0.441 | **0.683** |
| PCR | 0.576 | **0.907** |
| DCR | 0.173 | 0.231 |

`cars-in-black` goes from nothing to the **best SP of the three** — the colour
gate is now the most effective filter in the run, ahead of the spatial filter
(0.598) and the motion prompt (0.460).

Note the macro DCR *rises*. That is an artefact of the before column, not a
regression: an expression that made zero predictions contributed DCR = 0.000 to
the average. On the two expressions that were actually running, DCR is
unchanged (0.230 and 0.289 in both columns).

### Checking it on CARLA before trusting it there

The thresholds that remain are checkable rather than assumed.
`eval/check_color_classifier.py` scores the classifier against a dataset's own
colour labels and reports AUC, balanced accuracy, TPR/TNR and abstention rate:

```bash
python eval/check_color_classifier.py --dataset referkitti
python eval/check_color_classifier.py --dataset referkitti --no_peers   # contrast
python eval/check_color_classifier.py --dataset carla \
    --carla_scenarios dataset/carla_eval/eval_scenarios
```

CARLA's `gt.json` carries an explicit per-vehicle `color` field as an `"R,G,B"`
string, so the CARLA check is *cleaner* than the Refer-KITTI one — the label is
the actual paint value rather than an inferred annotation. Run it before
quoting any colour result on the sweep.

`tests/test_color_classifier.py` (37 tests) pins the regression pixel, the
chroma guard, and exposure invariance — that last one asserts the decision is
unchanged as the whole frame is scaled from 0.5x to 1.9x brightness, which is
the property the CARLA transfer rests on.

## Motion prompts — measured, built, and left off

Motion is the third attribute channel in Refer-KITTI (42% of expression files
mention it; 23% of the test split have *only* a motion cue). Unlike colour it
was never wrong — it simply had no filter at all, which is why `moving-cars`
scores SP 0.460 / SR 0.864: it emits every car it finds.

The obvious fix is to reuse `SceneGraphBuilder._motion_attrs`, which already
labels a track moving/stationary from its position history. **That would have
been a second dead filter.** Refer-KITTI is shot from a driving car, so
image-space displacement measures ego-motion: a *parked* car sweeps across the
frame as you drive past, while a car matching your speed sits still in it.
Measured on Refer-KITTI's own `moving-cars` labels, 8,357 track-frames:

| cue | AUC |
|---|---:|
| raw image displacement — what `_motion_attrs` uses | **0.498** |
| minus the frame's median displacement | 0.569 |
| residual from a fitted radial ego-flow field | **0.645** |

0.498 is the number that matters: no signal whatsoever. The median-subtraction
fix only reaches 0.569 because ego-motion induces a *parallax* field, not a
uniform translation — near objects sweep fast, far ones barely move, so no
single vector compensates it.

### The model

Under forward camera translation every static point flows radially away from
the focus of expansion, with magnitude falling off as 1/depth:

```
d_i = s * (p_i - foe)
```

Both `s` and `foe` are unknown per frame, but the relation is linear in `s` and
`c = s·foe`, so one least-squares solve over the frame's tracks recovers the
field; an independently moving object is the one whose displacement does not
fit it. Fitting over the tracks themselves means the field follows whatever the
majority are doing, which is the right prior on a road. Ground-contact point
(box bottom-centre) is used rather than box centre, because it sits on the road
plane and barely moves as the box grows.

### What it is worth

`eval/check_motion_classifier.py`, against Refer-KITTI's own motion labels:

| state | AUC raw | AUC ego | balanced accuracy | abstains |
|---|---:|---:|---:|---:|
| moving | 0.480 | 0.631 | **0.589** | 46.5% |
| stationary | 0.524 | 0.585 | **0.552** | 40.6% |

Weak. For scale, in this same pipeline the spatial cue scores 0.95-0.97 and
colour 0.71-0.86. And these are measured on **ground-truth** boxes, so they are
an upper bound — real tracks fragment and switch identity.

At balanced accuracy 0.589 a gate deletes nearly as many targets as
distractors, so the motion gate is **off by default**. Enable with
`--use_motion_filter`, and read `check_motion_classifier.py` before trusting it.

### What it refuses to score

`braking`, `turning`, and the ego-relative direction predicates ("cars in the
counter direction of ours") need a reference heading that a monocular dashcam
track does not provide. `canonical_motion` returns `"unscoreable"` for these
rather than guessing, and the gate becomes a no-op instead of a filter. The
distinction is deliberate: "I cannot score this" and "there is nothing to
score" must not collapse into the same behaviour, and the unscoreable phrasings
are common in Refer-KITTI.

Tracks also start *confirmed* rather than unconfirmed — the cue needs several
frames of history before it says anything, and dropping every track for its
first frames would cost more than the gate could recover.

`tests/test_motion_classifier.py` (23 tests) pins the ego-flow fit against a
synthetic radial field, pins that an independently-moving object gets the
largest residual, and pins every abstention path.

## Macro vs micro## Macro vs micro

The run summary reports both, because they answer different questions and a
micro number read as a macro claim is wrong by a lot on this benchmark:

- **macro** averages the per-expression rates — every expression counts once.
  This is the number to quote when comparing methods.
- **micro** pools the raw counts — the long, busy expressions dominate.

They coincide when only one expression is scored.

## Notes and known gaps

- **`eval_referkitti.py` was broken** against `worker_clean.Worker`: it passed
  `referring_topk=`, which that `Worker.__init__` does not accept, so any
  invocation died before the first frame. Top-k has been removed everywhere —
  `eval_referkitti.py`, `tune_referkitti_optuna.py` and the legacy
  `eval/worker.py`, which was the only implementation of it and which nothing
  reached (no eval script passes `referring_*`, so every caller left it at
  `"none"`). `referring_mode` is now `{none, threshold}` throughout.

- **The Week-3 grounded path does not apply to most of this benchmark.** Of the
  215 distinct Refer-KITTI sentences, the query parser accepts 110 and raises
  `UnknownRelationError` on 105 — the region prompts (`"cars in the left"`),
  which belong to `SceneGraphMissionFilter`, not the parser. Of the 110 that do
  parse, almost all are plain or unary (`"moving cars"`, `"turning vehicles"`);
  only the `"... in front of the camera"` family produces a binary anchor, and
  that anchor is the ego vehicle, which is not a tracked candidate. So
  `score_candidates` would return 0.0 for the relation term throughout and
  `select_answers` would pass every candidate through. Running Refer-KITTI
  `--grounded` would measure the plain path with extra machinery attached, so
  this script does not enable it. The relational claim needs the CARLA sweep.

- `--max_expressions` defaults to 1. Refer-KITTI has 818 expression files over
  18 sequences; a full sweep is a different (much longer) job than a solo test.

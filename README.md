# Text-Guided Multi-Object Tracking

Built on top of [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), this extends the upstream detector into a full text-guided Multi-Object Tracking (MOT) pipeline. Describe what you want to track in plain English — the system detects, tracks, and filters objects to match your description.

---

## What It Does

- **Track objects by text description** — "red car", "pedestrian on the left", "vehicle in light color"
- **Three tracker backends** — ByteTrack (IoU only), CLIPTracker (IoU + CLIP embeddings), SmartCLIPTracker
- **Spatial and color filtering** — parses keywords from your prompt automatically
- **Scene graph per frame** — spatial relations, motion, color, and size for every tracked object
- **Color re-identification** — recovers track IDs after brief occlusion or camera perspective changes
- **Scale-aware detection** — adaptive thresholds so distant/small objects are not dropped
- **Six evaluation datasets** — VisDrone, UAVDT, UA-DETRAC, KITTI, MOT17, ReferKITTI, CARLA

---

## Setup

**Requirements:** Python 3.10, CUDA-capable GPU

```bash
git clone https://github.com/azzy13/selectivetracking.git
cd selectivetracking

conda create -n env_dino python=3.10
conda activate env_dino

pip install -r requirements.txt
pip install -e . --no-build-isolation
```

> `--no-build-isolation` is required so the build step can find the already-installed torch when compiling the CUDA ops.

Key dependencies installed by `requirements.txt`:

---

## Quick Start

### Run on a video

```bash
python3 demo/inference_w_worker.py --video videos/carla1.mp4 --output outputs/annotated.mp4 --text-prompt "red car." --fp16
```

This runs the full pipeline: GroundingDINO detection → ByteTrack → scene graph mission filter → color re-ID → annotated video output.

**Common options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--text-prompt` | `"red car."` | What to track (plain English, DINO-style dot-separated) |
| `--fp16` | off | Half-precision inference — recommended on any modern GPU |
| `--box-threshold` | 0.35 | Detection confidence cutoff |
| `--track-threshold` | 0.45 | Tracker confirmation threshold |
| `--match-threshold` | 0.80 | IoU matching threshold |
| `--small-box-area-thresh` | 5000 | px² below which lower detection thresholds apply |
| `--no-mission-filter` | — | Disable the scene graph filter (track everything) |

---

## Evaluation

All eval scripts share the same core flags:

| Flag | Description |
|------|-------------|
| `--devices 0,1` | GPUs to use |
| `--jobs 2` | Sequences to run in parallel |
| `--fp16` | Half-precision |
| `--save_video` | Write annotated video per sequence |
| `--tracker` | `bytetrack`, `clip`, or `smartclip` |

### VisDrone

```bash
python3 eval/eval_visdrone.py --data_root dataset/visdrone_mot_format --split val --text_prompt "car. pedestrian." --box_threshold 0.4 --text_threshold 0.8 --track_thresh 0.45 --match_thresh 0.85 --weights weights/swinb_light_visdrone_ft_best.pth --tracker clip --devices 0,1 --jobs 2 --fp16 --save_video
```

### KITTI

```bash
python3 eval/eval_new.py --images dataset/kitti/validation/image_02 --labels dataset/kitti/validation/label_02 --devices 0,1 --jobs 2 --fp16
```

### ReferKITTI (referring expression tracking)

Uses optimized parameters from Optuna Trial 532 (combined score 0.214):

```bash
python3 eval/eval_referkitti.py --data_root dataset/referkitti/ --weights weights/swinb_light_visdrone_ft_best.pth --tracker clip --devices 0,1 --jobs 2 --fp16 --box_threshold 0.455 --text_threshold 0.363 --track_thresh 0.45 --match_thresh 0.80 --track_buffer 110 --lambda_weight 0.568 --text_gate_mode penalty --text_gate_weight 0.736 --referring_mode threshold --referring_thresh 0.263 --small_box_area_thresh 4000 --save_video
```

### CARLA (prompt-compliance evaluation)

```bash
python3 eval/eval_carla.py --carla_scenarios dataset/carla_eval/eval_scenarios --tracker clip --fp16
```

Reports Semantic Precision, Semantic Recall, Prompt Coverage Ratio, and Semantic ID Switches per scenario.

---

## Hyperparameter Tuning

Optuna search scripts for each dataset:

```bash
python3 eval/tune_referkitti_optuna.py
python3 eval/tune_kitti_optuna.py
python3 eval/tune_visdrone_optuna.py
```

Results are saved to `docs/optuna_trials_log.csv` and `docs/optuna_visdrone.csv`.
Best ReferKITTI parameters are recorded in `TRIAL_532_BEST_PARAMS.txt`.

---

## Scene Graph Demo

Run detection + tracking on a single sequence and save the per-frame scene graph as `.jsonl`:

```bash
python3 eval/run_scene_graph_demo.py --seq scenario_001 --img_folder dataset/carla_eval/eval_scenarios/scenario_001/images --out /tmp/sg_demo/
```

Visualize the saved scene graph:

```bash
python3 eval/visualize_scene_graph.py --jsonl /tmp/sg_demo/scenario_001_scene_graphs.jsonl --images dataset/carla_eval/eval_scenarios/scenario_001/images --out /tmp/sg_demo/viz/
```

---

## Dataset

The CARLA Referring Target Evaluation Set is available on Hugging Face:

**[azzy13/carla_referring_target_evaluation_set](https://huggingface.co/datasets/azzy13/carla_referring_target_evaluation_set)**

A synthetic benchmark for **referring expression tracking** — the task of tracking a single described target (e.g. "red sedan") while ignoring all other vehicles in the scene. Unlike standard MOT datasets that evaluate all objects, this benchmark tests whether a tracker can isolate and follow exactly the object that matches the natural language description across challenging conditions including weather variation, lighting changes, camera motion, and the presence of visually similar distractors.

24 scenarios generated in CARLA simulator (Towns 10HD, 03, 05). The paper-reported benchmark uses the first 18 scenarios; this repository's released set expands that benchmark to 24 by adding follow-camera variants. Each scenario provides per-frame ground-truth bounding boxes with `is_target` flags distinguishing the referred object from distractors.

**Evaluation metrics** (computed by `eval_carla.py`):

| Metric | Description |
|--------|-------------|
| **Semantic Precision** | Fraction of tracked detections that are actually the target |
| **Semantic Recall** | Fraction of target frames where the target was tracked |
| **Prompt Coverage Ratio** | Combined measure of how consistently the prompt-described object is tracked |
| **Distractor Confusion Rate** | How often a non-target object was incorrectly tracked as the target |
| **Semantic ID Switches (SID)** | Number of times the tracker switched from the correct target to a distractor or lost it entirely |

| # | Scenario | Camera | Description |
|---|----------|--------|-------------|
| 1 | `clear_day_baseline` | static | High sun, clear sky — baseline lighting |
| 2 | `overcast` | static | Heavy cloud cover, flat diffuse light |
| 3 | `heavy_rain` | loose-follow | Rain with wet road reflections |
| 4 | `dusk_golden_hour` | loose-follow | Low sun angle, warm directional light |
| 5 | `night` | loose-follow | Night lighting |
| 6 | `dense_fog` | loose-follow | Low visibility fog |
| 7 | `color_confusable` | loose-follow | Other red vehicles present as distractors |
| 8 | `same_color_diff_class` | loose-follow | Same color, different vehicle class distractors |
| 9 | `high_density` | static | Dense mixed traffic |
| 10 | `high_altitude` | static | Elevated camera angle |
| 11 | `low_altitude_steep` | loose-follow | Low steep camera angle |
| 12 | `side_follow` | loose-follow | Side-on camera perspective |
| 13 | `town03_suburban` | static | Suburban map (Town03) |
| 14 | `town05_highway` | loose-follow | Highway map (Town05) |
| 15 | `multiple_red_sedans` | static | 3 target vehicles simultaneously |
| 16 | `long_sequence` | static | Extended duration sequence |
| 17 | `dense_urban_traffic` | static | Dense urban scene |
| 18–24 | `follow_base` / `follow_variant_*` | follow | Follow-camera variants with different spawn points and lighting |

Download and run:

```bash
hf download azzy13/carla_referring_target_evaluation_set eval_scenarios.zip --repo-type dataset --local-dir dataset/carla_eval/
unzip dataset/carla_eval/eval_scenarios.zip -d dataset/carla_eval/
python3 eval/eval_carla.py --carla_scenarios dataset/carla_eval/eval_scenarios --tracker clip --fp16
```

---

## Weights

| File | Description |
|------|-------------|
| `weights/groundingdino_swinb_cogcoor.pth` | Upstream SwinB pretrained (IDEA-Research) |
| `weights/groundingdino_swint_ogc.pth` | Upstream SwinT pretrained (IDEA-Research) |
| `weights/swinb_light_visdrone_ft_best.pth` | SwinB lightly fine-tuned on VisDrone — best checkpoint |

---

## Technical Details

There are two worker pipelines, each with its own color classification approach:

| Pipeline | Entry point | Color method | Filter |
|----------|-------------|--------------|--------|
| **Scene graph pipeline** | `worker_simple.py` / `demo/inference_w_worker.py` | LAB classifier (`scene_graph.py`) | `SceneGraphMissionFilter` |
| **CLIP pipeline** | `worker_clean.py` / `eval_referkitti.py` | HSV patch voting (`worker_clean.py`) | `ReferringDetectionFilter` + `TrackColorGate` |

### LAB Color Classifier (`scene_graph.py`) — Scene Graph Pipeline

Used by `worker_simple.py`. Each track's bounding box crop is classified in
CIE L\*a\*b\* color space. L\*a\*b\* is preferred over HSV because lightness is
fully decoupled from chroma and the a\*b\* plane has no hue wrap-around at red.

Per-crop classification:
- Chroma (`sqrt(a² + b²)`) ≥ 15 → chromatic: classify by hue angle in the a\*b\* plane (red / orange / yellow / green / blue / purple / pink).
- Chroma < 15 → achromatic: classify by L\* (dark / gray / light).

Crops narrower than 8 px in either dimension are upscaled to 16 px with
nearest-neighbor interpolation before classification, so tiny but real detections
still contribute color evidence from their first frame rather than returning `"unknown"`.

### HSV Patch Voting (`worker_clean.py`) — CLIP Pipeline

Used by `worker_clean.py` (ReferKITTI, CARLA clean mode). Each detection crop
is divided into a 4×4 grid; each patch votes for a color based on HSV values.
Hue-neighbor tolerance is applied (orange patches count as supporting evidence
for a "red" target). A presence-based threshold accepts the target color if it
reaches `min_target_patches` votes even if it is not the majority.

LAB approach is clearly superior. 

### Scale-Aware Detection

Initial DINO inference runs at `box_thresh × 0.5` to avoid missing distant objects.
A second per-detection pass applies:

| Box area | Threshold multiplier |
|----------|---------------------|
| < 1250 px² (tiny) | × 0.45 |
| < 5000 px² (small) | × 0.60 |
| < 15000 px² (medium) | × 0.80 |
| ≥ 15000 px² (large) | × 1.00 |

### Scene Graph (`SceneGraphBuilder`)

Built per-frame from tracker output. Each node carries: normalized position, size
label, spatial region label (top-left, bottom-right, etc.), LAB-based dominant color,
motion label (stationary / moving / approaching / receding), and heading vector from
10-frame position history. Edges carry pairwise spatial relations (`left-of`, `above`,
`near`, `overlapping`, etc.) and a `visually-similar` edge when CLIP embeddings agree.

### `SceneGraphMissionFilter`

Parses color and spatial constraints from the text prompt automatically. Accumulates
color votes over a 15-frame window per track and scores each track against the
constraints. Soft mode (default) keeps tracks above `score_thresh` (0.10); hard mode
requires all constraints to pass.

### `ColorReIDMatcher`

When a track is lost and a new track appears with the same dominant color and similar
area, the new ID is remapped to the old canonical ID. Lost tracks are held in a
graveyard for `max_lost_frames` (default 25) frames before expiry.

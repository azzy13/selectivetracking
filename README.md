# 🎯 Selective Tracking

**Natural Language Driven Multi-Object Tracking with Prompt-Consistency Association**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)](https://pytorch.org/)

---

## 📖 Overview

**Selective Tracking** is a text-guided multi-object tracking (MOT) system that combines **Grounding DINO** for natural language object detection with **CLIP-enhanced ByteTrack** for robust multi-object tracking. Our system introduces **prompt-consistency association**, leveraging spatially masked CLIP embeddings and text-gated matching to maintain accurate object identities in complex scenarios.

### Key Innovation

Traditional MOT systems rely solely on spatial and appearance features. Selective Tracking extends this paradigm by incorporating **semantic association** through text prompts, enabling:
- Natural language-driven object filtering
- Text-gated cost matrix fusion for improved association
- Prompt-consistent tracking across occlusions and appearance changes

---

## ✨ Key Features

### 🎯 Prompt-Driven Detection
- **Natural Language Queries**: Track objects using text prompts like "red sedan", "person wearing a hat"
- **Grounding DINO Integration**: Zero-shot detection with high accuracy
- **Flexible Object Categories**: No pre-defined class limitations

### 🔄 Hybrid Tracking Pipeline
- **ByteTrack Foundation**: Robust two-stage association (high/low confidence detections)
- **CLIP Appearance Features**: Spatially masked region embeddings for re-identification
- **Text-Gated Matching**: Semantic cost fusion with adaptive λ parameter

### 🛠️ Detection Refinements
- **HSV Color Voting**: Dominant color extraction for color-based prompts
- **Scale-Aware Thresholding**: Dynamic confidence adaptation based on object size
- **Duplicate Removal**: Appearance-based NMS for cleaner tracks

### 📊 CARLA Benchmark Metrics
Custom evaluation framework for autonomous driving scenarios:
- **SP** (Success Precision): Track completeness
- **SR** (Success Rate): Track recall
- **PCR** (Prompt Compliance Rate): Semantic alignment with text prompt
- **DCR** (Detection Compliance Rate): Detection-level semantic accuracy
- **SID** (Switch ID): Identity switch count

---

## 🚀 Quick Start

### Installation

#### 1. Create Conda Environment
```bash
conda env create -f environment.yaml
conda activate selective_tracking
```

#### 2. Install Package
```bash
pip install -e .
```

#### 3. Download Weights
Download Grounding DINO pre-trained weights:
```bash
mkdir weights
cd weights
# Download from official Grounding DINO repository
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

---

## 💻 Basic Usage

### Python API

```python
import cv2
import torch
from groundingdino.util.inference import load_model, predict
from selectivetrack.clip_tracker import CLIPTracker

# Load Grounding DINO detector
model = load_model(
    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "weights/groundingdino_swint_ogc.pth"
)

# Initialize CLIP-enhanced tracker
tracker = CLIPTracker(
    track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    device='cuda'
)

# Process video
cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects with text prompt
    boxes, logits, phrases = predict(
        model=model,
        image=frame,
        caption="red car",
        box_threshold=0.35,
        text_threshold=0.25
    )
    
    # Update tracker with detections
    tracks = tracker.update_with_clip(
        detections=boxes,
        frame=frame,
        text_prompt="red car"
    )
    
    # Draw tracks
    for track in tracks:
        x1, y1, w, h = track.tlwh
        track_id = track.track_id
        cv2.rectangle(frame, (int(x1), int(y1)), 
                     (int(x1+w), int(y1+h)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

### Command-Line Example

```bash
python examples/track_video.py \
    --video path/to/video.mp4 \
    --prompt "red car" \
    --output output.mp4 \
    --config groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --weights weights/groundingdino_swint_ogc.pth \
    --display
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Input Frame + Text Prompt                │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Grounding DINO       │
                    │  (Text-Driven Detection)│
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Detection Refinements  │
                    │ • HSV Color Voting     │
                    │ • Scale-Aware Thresh   │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Referring Filter      │
                    │ (Prompt Compliance)    │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ CLIP-Enhanced ByteTrack│
                    │ • IoU Cost (Spatial)   │
                    │ • CLIP Cost (Visual)   │
                    │ • Text Gate (Semantic) │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Hungarian Assignment │
                    └────────────┬───────────┘
                                 │
                                 ▼
                      Active Tracks with IDs
```

---

## 🧮 Prompt-Consistency Module

The core innovation is **text-gated cost matrix fusion**:

```
C_final = (1 - λ) × C_iou + λ × C_clip

where:
  C_iou   : Spatial IoU cost (motion consistency)
  C_clip  : CLIP appearance cost (visual similarity)
  λ       : Text-gating weight ∈ [0, 1]
```

**Text Gating Strategy:**
- λ increases when text prompt has high semantic relevance
- Balances motion prediction with appearance matching
- Adaptive fusion based on CLIP text-image similarity

---


## 🙏 Acknowledgements

This project builds upon excellent prior work:

- **[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)**: Open-set object detection with natural language
- **[ByteTrack](https://github.com/ifzhang/ByteTrack)**: Simple, fast and strong multi-object tracking
- **[CLIP](https://github.com/openai/CLIP)**: Contrastive Language-Image Pre-training

Special thanks to the open-source community for making these resources available.

---

## 📧 Contact

- **GitHub**: [@azzy13](https://github.com/azzy13)
- **Issues**: [Report bugs or request features](https://github.com/azzy13/selectivetracking/issues)

---

# Text Embedding Flow in ReferKITTI Evaluation

## Overview
This document traces how the referring expression text (e.g., "black car on the right") is converted to a CLIP embedding and passed through the system to the tracker's `update()` method.

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ eval_referkitti.py - Main Script                               │
└───────────────────────┬─────────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────┐
        │ For each sequence and expression: │
        │   expr["text"] = "black car..."   │
        │   expr["obj_ids"] = [5, 8]        │
        └───────────────┬───────────────────┘
                        ↓
        ┌───────────────────────────────────────────────────┐
        │ Worker.__init__() - eval/worker_clean.py:323     │
        │ Parameters:                                       │
        │   text_prompt = expr["text"]                     │
        │   tracker_type = "clip"                          │
        └───────────────┬───────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────────────────────┐
        │ Worker.__init__() - Lines 400-414                │
        │                                                   │
        │ 1. Load CLIP model:                              │
        │    clip_model, clip_preprocess =                 │
        │      clip.load("ViT-B/32", device)               │
        │                                                   │
        │ 2. Parse text prompt into class names:           │
        │    class_names = text_prompt.split(".")          │
        │    # ["black car on the right"]                  │
        │                                                   │
        │ 3. Encode text to CLIP embedding:                │
        │    tokens = clip.tokenize(class_names)           │
        │    text_embedding = F.normalize(                 │
        │      clip_model.encode_text(tokens), dim=-1      │
        │    )                                              │
        │    # Shape: [1, 512] (unit-normalized)           │
        │                                                   │
        │ 4. Store as instance variable:                   │
        │    self.text_embedding = text_embedding          │
        └───────────────┬───────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────────────────────┐
        │ Worker.process_sequence() - Line 549             │
        │ Per-frame loop for tracking                      │
        └───────────────┬───────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────────────────────┐
        │ For each frame in sequence:                       │
        │   1. Detect objects (GroundingDINO)              │
        │   2. Filter detections (referring filter)        │
        │   3. Track objects (CLIP tracker)                │
        └───────────────┬───────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────────────────────┐
        │ Worker.update_tracker_clip() - Line 475          │
        │                                                   │
        │ Parameters:                                       │
        │   dets_xyxy: np.ndarray [N, 5]                   │
        │   frame_bgr: current frame                        │
        │                                                   │
        │ Steps:                                            │
        │ 1. Compute detection embeddings:                 │
        │    det_embs = _compute_detection_embeddings()    │
        │                                                   │
        │ 2. Call tracker.update():                        │
        │    return self.tracker.update(                   │
        │      detections=dets,                            │
        │      detection_embeddings=det_embs,              │
        │      img_info=(orig_h, orig_w),                  │
        │      text_embedding=self.text_embedding, ◄────── │
        │      class_names=self.class_names                │
        │    )                                              │
        └───────────────┬───────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│ CLIPTracker.update() - tracker/tracker_w_clip.py:203           │
│                                                                 │
│ Receives:                                                       │
│   text_embedding: torch.FloatTensor [1, 512] (CLIP text feat)  │
│   detection_embeddings: List[Tensor] (CLIP image features)     │
│                                                                 │
│ Uses text_embedding for:                                        │
│   1. Text-similarity gating (lines 241-264)                    │
│   2. Tracker association (optional, via tracker params)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Code Trace

### Step 1: Expression Text Extraction (eval_referkitti.py)

**Location**: `eval/eval_referkitti.py:589-608`

```python
for expr in expr_list:
    expr_name = f"{seq}_expr{expr['expr_id']:04d}"
    print(f"Text: {expr['text']}")  # e.g., "black car on the right"
    print(f"Obj IDs: {expr['obj_ids']}")  # e.g., [5, 8]

    # Create Worker with the expression text
    worker = Worker(
        tracker_type=args.tracker,
        text_prompt=expr["text"],  # ◄── Expression passed here
        # ... other params ...
    )
```

**Key Variables**:
- `expr["text"]`: The natural language referring expression
- `expr["obj_ids"]`: Ground truth object IDs referenced by this expression

---

### Step 2: Worker Initialization - CLIP Text Encoding

**Location**: `eval/worker_clean.py:323-428`

```python
class Worker:
    def __init__(
        self,
        text_prompt: str = "car. truck. bus.",  # ◄── Expression text
        tracker_type: str = "clip",
        # ... other params ...
    ):
        self.text_prompt = text_prompt  # Store original text

        # Parse text into class names (line 400)
        self.class_names = [
            c.strip() for c in self.text_prompt.split(".")
            if c.strip()
        ] or ["object"]
        # Example: "black car on the right" → ["black car on the right"]

        # Initialize CLIP if needed (lines 406-414)
        need_clip = self.tracker_type in ("clip", "smartclip") or referring_mode != "none"
        if need_clip:
            # 1. Load CLIP model
            self.clip_model, self.clip_preprocess = clip.load(
                "ViT-B/32", device=self.device
            )
            self.clip_model.eval()

            # 2. Encode text to CLIP embedding
            with torch.no_grad():
                tokens = clip.tokenize(self.class_names).to(self.device)
                # tokens shape: [1, 77] (CLIP max length)

                self.text_embedding = F.normalize(
                    self.clip_model.encode_text(tokens).float(),
                    dim=-1
                ).contiguous()
                # text_embedding shape: [1, 512] (unit-normalized)
```

**Key Points**:
1. **Input**: `text_prompt = "black car on the right"`
2. **Processing**: CLIP's text encoder converts natural language to 512-dim vector
3. **Storage**: `self.text_embedding` is a **[1, 512]** tensor stored on GPU
4. **Normalization**: Unit-normalized (L2 norm = 1) for cosine similarity

---

### Step 3: Per-Frame Tracking Loop

**Location**: `eval/worker_clean.py:603-645`

```python
def process_sequence(self, seq, img_folder, ...):
    # Frame loop
    for idx, frame_name in enumerate(sorted(frame_files)):
        frame_id = int(os.path.splitext(frame_name)[0])
        img = cv2.imread(os.path.join(seq_path, frame_name))

        # 1. Detect
        dets = self.predict_detections(img, tensor, orig_h, orig_w)

        # 2. Filter (optional referring expression filter)
        if self.referring_filter is not None:
            dets = self.referring_filter.filter(img, dets)

        # 3. Track with CLIP
        if self.tracker_type in ("clip", "smartclip"):
            tracks = self.update_tracker_clip(dets, img, orig_h, orig_w)
            # ◄── This calls tracker.update() with text_embedding
        else:
            tracks = self.update_tracker(dets, orig_h, orig_w)
```

---

### Step 4: Computing Detection Embeddings

**Location**: `eval/worker_clean.py:488-542`

```python
def _compute_detection_embeddings(
    self, frame_bgr: np.ndarray, dets_xyxy: np.ndarray
) -> List[Optional[torch.Tensor]]:
    """
    Compute CLIP image embeddings for each detection.
    Combines crop embedding + full image embedding for spatial context.
    """
    # 1. Encode full image once
    full_img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    full_img_tensor = self.clip_preprocess(full_img_pil).unsqueeze(0).to(self.device)

    with torch.no_grad():
        full_img_emb = F.normalize(
            self.clip_model.encode_image(full_img_tensor), dim=-1
        ).float().cpu().squeeze(0)
        # Shape: [512]

    # 2. Encode each detection crop
    crops = []
    for (x1, y1, x2, y2, _) in dets_xyxy:
        # Crop with padding
        xi1 = max(0, int(x1) - self.clip_pad)
        yi1 = max(0, int(y1) - self.clip_pad)
        xi2 = min(W, int(x2) + self.clip_pad)
        yi2 = min(H, int(y2) + self.clip_pad)

        if valid_crop:
            crops.append(Image.fromarray(rgb[yi1:yi2, xi1:xi2]))
        else:
            crops.append(None)

    # 3. Batch encode crops
    batch = [self.clip_preprocess(c).unsqueeze(0) for c in crops if c is not None]
    batch_t = torch.cat(batch, 0).to(self.device)

    with torch.no_grad():
        crop_embs = F.normalize(
            self.clip_model.encode_image(batch_t), dim=-1
        ).float().cpu()
        # Shape: [N_valid, 512]

    # 4. Combine crop + full image embeddings
    out = []
    j = 0
    for c in crops:
        if c is None:
            out.append(None)
        else:
            # Average and re-normalize
            combined_emb = F.normalize(
                (crop_embs[j] + full_img_emb) / 2.0, dim=-1
            )
            out.append(combined_emb)  # Shape: [512]
            j += 1

    return out  # List of [512] tensors or None
```

**Why combine crop + full image?**
- **Crop embedding**: Captures object-specific appearance (e.g., "black car")
- **Full image embedding**: Provides spatial context (e.g., "on the right")
- **Combination**: Enables matching both appearance AND spatial expressions

---

### Step 5: Calling Tracker Update

**Location**: `eval/worker_clean.py:475-486`

```python
def update_tracker_clip(
    self, dets_xyxy: np.ndarray, frame_bgr: np.ndarray,
    orig_h: int, orig_w: int
):
    """Update CLIP-aware tracker."""
    dets = dets_xyxy if dets_xyxy.size else np.empty((0, 5), dtype=np.float32)

    # Compute detection embeddings
    det_embs = self._compute_detection_embeddings(frame_bgr, dets)

    # Call tracker update
    return self.tracker.update(
        detections=dets,                      # [N, 5] boxes + scores
        detection_embeddings=det_embs,        # List[Tensor[512]] or None
        img_info=(orig_h, orig_w),            # Frame size
        text_embedding=self.text_embedding,   # ◄── [1, 512] CLIP text features
        class_names=self.class_names,         # ["black car on the right"]
    )
```

---

### Step 6: Tracker Usage of Text Embedding

**Location**: `tracker/tracker_w_clip.py:203-402`

The tracker's `update()` method receives `text_embedding` and uses it for:

#### 6A. Text-Similarity Gating (Lines 241-264)

```python
def update(self, detections, detection_embeddings, img_info, text_embedding, ...):
    # Split detections by confidence
    dets_hi = high_confidence_detections
    dets_lo = low_confidence_detections

    # Normalize text embedding
    tnorm = text_embedding / (text_embedding.norm(dim=-1, keepdim=True) + 1e-6)
    sim_thr = self.args.text_sim_thresh  # e.g., 0.2

    # Filter function
    def _gate_and_build(dets_xyxy, scores_vec, embs_list):
        keep_dets, keep_scores, keep_embs = [], [], []
        for i, e in enumerate(embs_list):
            if e is None:
                # No embedding → always keep
                keep_dets.append(dets_xyxy[i])
                keep_scores.append(scores_vec[i])
                keep_embs.append(None)
                continue

            # Compute CLIP similarity with text
            v = e.detach().float()
            v = v / (v.norm() + 1e-6)
            sim = torch.max(torch.matmul(tnorm, v.to(tnorm.device)))

            # Gate by text similarity
            if sim.item() >= sim_thr:
                keep_dets.append(dets_xyxy[i])
                keep_scores.append(scores_vec[i])
                keep_embs.append(e)

        return keep_dets, keep_scores, keep_embs

    # Apply to both high and low confidence detections
    dets_hi, scores_hi, emb_hi = _gate_and_build(dets_hi, scores_hi, emb_hi)
    dets_lo, scores_lo, emb_lo = _gate_and_build(dets_lo, scores_lo, emb_lo)
```

**Purpose**: Filter out detections that don't match the referring expression semantically.

**Example**:
- Expression: "black car on the right"
- Detection 1: Black car → CLIP similarity = 0.85 → **KEEP**
- Detection 2: White truck → CLIP similarity = 0.15 → **DROP**

---

## Summary of Text Embedding Journey

| Step | Location | Action | Output |
|------|----------|--------|--------|
| 1 | `eval_referkitti.py:608` | Extract expression text | `"black car on the right"` |
| 2 | `worker_clean.py:323` | Pass to Worker `__init__` | `text_prompt` parameter |
| 3 | `worker_clean.py:411-413` | Encode with CLIP | `[1, 512]` tensor (GPU) |
| 4 | `worker_clean.py:484` | Pass to `tracker.update()` | `text_embedding` parameter |
| 5 | `tracker_w_clip.py:241-264` | Text-similarity gating | Filter detections by CLIP similarity |
| 6 | `tracker_w_clip.py:292-367` | (Optional) CLIP fusion | Blend IoU + appearance matching |

---

## Key Takeaways

### 1. **One-time Encoding**
The expression text is encoded to CLIP embedding **once** during Worker initialization, then reused for all frames in the sequence.

### 2. **Two Embedding Types**
- **Text embedding**: From referring expression (e.g., "black car")
- **Detection embeddings**: From image crops of detected objects

### 3. **Dual Purpose**
Text embedding serves two roles:
- **Gating**: Filter detections that don't match the expression semantically
- **Association** (optional): Help match tracks across frames using appearance

### 4. **Spatial Context**
Detection embeddings combine crop + full image to handle spatial expressions like "on the right" or "leftmost car".

### 5. **Expression-Specific Tracking**
Each referring expression gets its own Worker instance with its own text embedding, enabling expression-specific tracking.

---

## Example Flow for "black car on the right"

```
Expression: "black car on the right"
    ↓
CLIP Text Encoder
    ↓
text_embedding: [0.12, -0.34, 0.56, ..., 0.23]  (512-dim)
    ↓
Frame 1: Detect 5 objects
    ↓
Compute detection embeddings:
  - Det 1 (black car, right): [0.15, -0.30, 0.52, ...]  → sim=0.85 ✓
  - Det 2 (white truck, left): [0.88, 0.22, -0.11, ...] → sim=0.12 ✗
  - Det 3 (black car, left): [0.14, -0.32, 0.50, ...]  → sim=0.62 ✓
  - ...
    ↓
Text-similarity gating (thresh=0.2):
  - Keep: Det 1, Det 3 (similarity > 0.2)
  - Drop: Det 2 (similarity < 0.2)
    ↓
Track only relevant objects
```

---

## Configuration Parameters

### Text Similarity Threshold (`text_sim_thresh`)

**Default**: 0.2

**Effect**:
- **Lower** (e.g., 0.1): More permissive, keep more detections
- **Higher** (e.g., 0.3): Stricter filtering, only very relevant objects

**Recommendation for ReferKITTI**: 0.2 - 0.25

### Lambda Weight (`lambda_weight`)

**Default**: 0.25

**Effect**: Controls CLIP vs IoU fusion in tracking
- **0.0**: Pure IoU matching (ByteTrack)
- **0.25**: Balanced (recommended)
- **1.0**: Pure CLIP appearance matching

---

This text embedding flow enables **semantic-aware tracking** where only objects matching the natural language expression are tracked!

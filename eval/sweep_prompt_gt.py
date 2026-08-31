#!/usr/bin/env python3
"""Week-2 sweep → prompt-compliance ground truth.

Joins `dataset/sweep` onto the same schema `carla_sim/evaluate_prompt_metrics.py`
consumes, so the relational sweep is scored by SP / SR / DCR with the identical
code that scores Refer-KITTI and CARLA.

The sweep is a *better* relational benchmark than the CARLA eval scenarios,
because its distractor is matched to the target on every perceivable attribute
— same class (`car`), same colour (`180,20,20`), similar size — and differs
only in the relation and its heading. A benchmark whose distractor differs in
colour would be solved by a colour filter without any relational reasoning.

## Validity comes from the GT edges, never from the folder name

Each clip directory is named `behind` or `front`, and `manifest.json` carries a
single `prompt` ("red car behind the bus"). **Neither is a reliable answer key.**
Derived from the `edges` in each clip's `gt_graphs` file:

| prompt | `behind` clips | `front` clips |
|---|---|---|
| red car **behind** the bus | both cars valid ×11, target only ×1, none ×1 | distractor only ×11, none ×2 |
| red car **in front of** the bus | none valid ×12, distractor only ×1 | target only ×12, distractor only ×1 |

So the manifest's own prompt is the **degenerate** direction: in the `behind`
geometry both red cars end up behind the bus, giving two valid answers in 11 of
13 clips, and in the `front` clips the correct answer is the *distractor* rather
than "nothing". Only `in_front_of` yields a clean single-answer key, which is
why the corrected key in `DOC.md` reads "the target in each front clip,
nothing in each behind clip" — that statement is about the `in_front_of` prompt.

This module therefore takes the relation from the prompt you pass and reads
validity out of the GT edges, per clip. A folder called `front` gets no special
treatment.

## Pooling

Every clip is one frame, so PCR collapses to 0/1 per clip and SID is
meaningless. To get a meaningful SP/SR/DCR the 26 clips are pooled into one
`gt_data` with `image_id = clip index`, and predictions are re-keyed the same
way. Read SP, SR and DCR from a sweep run; ignore PCR and SID.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

#: Relation names as the query parser spells them (snake_case), which is also
#: what the sweep's GT edges use. The scene-graph edge labels are hyphenated;
#: do not mix the two.
ANCHOR_CLASS_DEFAULT = "bus"
TARGET_CLASS_DEFAULT = "car"


def load_manifest(root: str) -> dict:
    with open(os.path.join(root, "manifest.json")) as f:
        return json.load(f)


def clip_dir(root: str, clip: dict) -> str:
    return os.path.join(root, clip["config_id"], clip["condition"])


def clip_name(clip: dict) -> str:
    return f"{clip['config_id']}_{clip['condition']}"


def load_clip_graph(root: str, clip: dict, frame: int = 0) -> dict:
    path = os.path.join(clip_dir(root, clip), "gt_graphs", f"{frame:06d}.json")
    with open(path) as f:
        return json.load(f)


def clip_image_path(root: str, clip: dict, frame: int = 0) -> str:
    return os.path.join(clip_dir(root, clip), "rgb", f"{frame:06d}.png")


def valid_ids(graph: dict, relation: str, anchor_id: int,
              target_class: str = TARGET_CLASS_DEFAULT) -> set:
    """Node ids that satisfy ``<node> <relation> <anchor>`` in this clip's GT.

    Read straight off the stored edges — the sweep computes them from world
    poses, so they are the authority on what the prompt refers to.
    """
    edges = {(e["subj"], e["relation"], e["obj"]) for e in graph["edges"]}
    by_id = {n["id"]: n for n in graph["nodes"]}
    return {
        n["id"] for n in graph["nodes"]
        if n["id"] != anchor_id
        and n.get("class") == target_class
        and (n["id"], relation, anchor_id) in edges
    }


def anchor_id_of(graph: dict, anchor_class: str = ANCHOR_CLASS_DEFAULT) -> Optional[int]:
    for n in graph["nodes"]:
        if n.get("class") == anchor_class:
            return n["id"]
    return None


def build_pooled_gt(root: str, relation: str, prompt: str,
                    target_class: str = TARGET_CLASS_DEFAULT,
                    anchor_class: str = ANCHOR_CLASS_DEFAULT
                    ) -> Tuple[dict, List[dict]]:
    """One ``gt_data`` over every clip, with ``image_id`` = clip index.

    Returns ``(gt_data, clips)`` where ``clips`` is the manifest clip list in
    the same order, so a caller can line predictions up by index.
    """
    manifest = load_manifest(root)
    clips = manifest["clips"]

    annotations: List[dict] = []
    per_clip: List[dict] = []
    gt_id = 0

    for idx, clip in enumerate(clips):
        graph = load_clip_graph(root, clip)
        anchor = anchor_id_of(graph, anchor_class)
        valid = valid_ids(graph, relation, anchor, target_class) if anchor is not None else set()

        for node in graph["nodes"]:
            is_anchor = node["id"] == anchor
            annotations.append({
                "image_id": idx,
                "gt_id": gt_id,
                "node_id": node["id"],
                "clip": clip_name(clip),
                "class": node.get("class"),
                "color": node.get("color_rgb"),
                "bbox_xyxy": [float(v) for v in node["box2d"]],
                # The anchor is a real object that the prompt does not refer to,
                # so it is a distractor for scoring purposes: emitting a bus box
                # for "red car in front of the bus" is a genuine error, and the
                # grounded path is supposed to never emit it.
                "is_target": node["id"] in valid,
                "role": ("anchor" if is_anchor
                         else "target" if node["id"] in valid
                         else "distractor"),
                "yaw": node.get("yaw"),
                "world_loc": node.get("loc"),
            })
            gt_id += 1

        per_clip.append({
            "index": idx,
            "name": clip_name(clip),
            "config_id": clip["config_id"],
            "condition": clip["condition"],
            "anchor_id": anchor,
            "valid_ids": sorted(valid),
            "num_valid": len(valid),
            "image": clip_image_path(root, clip),
            "img_folder": clip_dir(root, clip),
        })

    gt_data = {
        "meta": {
            "prompt": prompt,
            "relation": relation,
            "dataset": "sweep",
            "clips": len(clips),
            "num_annotations": len(annotations),
            "num_valid": sum(a["is_target"] for a in annotations),
            "manifest_prompt": manifest.get("prompt"),
            "frames_per_clip": manifest.get("frames_per_clip"),
        },
        "annotations": annotations,
    }
    return gt_data, per_clip


def answer_key_summary(per_clip: List[dict]) -> Dict[str, int]:
    """How many clips have 0 / 1 / >1 valid answers — the benchmark's health.

    A relation direction where most clips have two valid answers cannot
    discriminate, and a direction where most have zero is only testing the
    empty-answer path.
    """
    out: Dict[str, int] = {"no valid answer": 0, "one valid answer": 0,
                           "several valid answers": 0}
    for c in per_clip:
        n = c["num_valid"]
        key = ("no valid answer" if n == 0
               else "one valid answer" if n == 1 else "several valid answers")
        out[key] += 1
    return out

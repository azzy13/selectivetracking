"""Contract tests for the Refer-KITTI → prompt-compliance GT adapter.

The adapter (``eval/referkitti_prompt_gt.py``) is the only place where a
Refer-KITTI number and a CARLA number can diverge: the metrics themselves are
one shared implementation, so if SP/SR/DCR come out wrong on Refer-KITTI, they
come out wrong *here*.  These tests pin it two ways.

**Schema and semantics** — that ``is_target`` tracks the expression's
``label`` map frame by frame, that the boxes are decoded as top-left (not
centre) and land inside the image, that ``gt_id`` is unique.

**Two end-to-end controls through the real metric code** — feed the scorer a
prediction file synthesised from the GT itself:

    predict exactly the valid boxes      ->  SP = SR = PCR = 1,  DCR = 0
    predict exactly the distractors      ->  SP = SR = PCR = 0,  DCR = 1

Those two are the fixed points of the whole join.  If the frame ids, the box
format, or the validity flag were wrong in any way, neither would come out
clean — an oracle that scores 1.0 is the cheapest proof that the coordinate
convention and frame alignment are right.

Needs the Refer-KITTI checkout under ``dataset/referkitti``; skipped without it.
"""

import os
import sys

import pytest

import referkitti_prompt_gt as rkgt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(_ROOT, "dataset", "referkitti")

# A sequence/expression pair with both plenty of valid boxes and plenty of
# distractors, so neither control is vacuous.
SEQ = "0005"
EXPR = "cars-in-left"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(DATA_ROOT, "expression", SEQ)),
    reason=f"Refer-KITTI not present at {DATA_ROOT}",
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def boxes():
    return rkgt.load_sequence_boxes(DATA_ROOT, SEQ)


@pytest.fixture(scope="module")
def expression():
    return rkgt.load_expression(DATA_ROOT, SEQ, EXPR)


@pytest.fixture(scope="module")
def gt_data(boxes, expression):
    return rkgt.build_prompt_gt(boxes, expression)


@pytest.fixture(scope="module")
def metrics():
    """The SP/SR/PCR/DCR implementations, from the CARLA generator."""
    carla_sim = os.path.abspath(os.path.join(_ROOT, "..", "..", "carla_sim"))
    if not os.path.isfile(os.path.join(carla_sim, "evaluate_prompt_metrics.py")):
        pytest.skip(f"evaluate_prompt_metrics.py not found in {carla_sim}")
    if carla_sim not in sys.path:
        sys.path.insert(0, carla_sim)
    import evaluate_prompt_metrics as epm
    return epm


def _write_mot(path, annotations, keep_valid):
    """A prediction file that reproduces some subset of the GT exactly."""
    with open(path, "w") as f:
        for ann in annotations:
            if bool(ann["is_target"]) is not keep_valid:
                continue
            x1, y1, x2, y2 = ann["bbox_xyxy"]
            f.write(f"{ann['image_id']},{ann['track_id']},"
                    f"{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1.0,-1,-1,-1\n")
    return path


# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------
def test_annotation_schema(gt_data):
    required = {"image_id", "gt_id", "track_id", "bbox_xyxy", "is_target"}
    assert gt_data["annotations"], "adapter produced no annotations"
    for ann in gt_data["annotations"]:
        assert required <= set(ann)
        assert isinstance(ann["is_target"], bool)
        assert len(ann["bbox_xyxy"]) == 4


def test_gt_id_is_unique(gt_data):
    ids = [a["gt_id"] for a in gt_data["annotations"]]
    assert len(ids) == len(set(ids))


def test_meta_counts_match_annotations(gt_data):
    meta = gt_data["meta"]
    anns = gt_data["annotations"]
    assert meta["num_annotations"] == len(anns)
    assert meta["num_valid"] == sum(a["is_target"] for a in anns)
    assert meta["num_valid"] + meta["num_distractors"] == len(anns)
    assert meta["prompt"] == "cars in left"


def test_every_referred_id_has_a_box(gt_data):
    """An expression naming an id with no box that frame would silently cost
    recall; the adapter counts those rather than hiding them."""
    assert gt_data["meta"]["referred_ids_without_box"] == 0


# ----------------------------------------------------------------------
# Semantics
# ----------------------------------------------------------------------
def test_validity_is_per_frame_not_per_sequence(gt_data, expression):
    """The flag must follow the ``label`` map frame by frame.

    ``eval_referkitti.py`` unions the referred ids over the clip, which makes
    an object valid in frames the annotation never referred to it.  This test
    fails if the adapter ever drifts to that looser reading.
    """
    valid_by_frame = expression["valid_ids_by_frame"]
    for ann in gt_data["annotations"]:
        expected = ann["track_id"] in valid_by_frame.get(ann["image_id"], set())
        assert ann["is_target"] is expected, (
            f"frame {ann['image_id']} track {ann['track_id']}: "
            f"is_target={ann['is_target']} but label map says {expected}"
        )

    # And the distinction has to be observable in this clip, or the test above
    # would pass on a per-sequence union too.
    all_referred = set().union(*valid_by_frame.values())
    sometimes_invalid = {
        a["track_id"] for a in gt_data["annotations"]
        if a["track_id"] in all_referred and not a["is_target"]
    }
    assert sometimes_invalid, (
        "no referred id is ever invalid in this clip — pick a different "
        "expression, this test cannot distinguish the two readings here"
    )


def test_boxes_are_top_left_and_inside_the_image(gt_data):
    """Refer-KITTI stores ``x_left y_top w h`` despite the YOLO-style layout.

    Decoding them as centre coordinates shifts every box by half its size,
    which pushes boxes off the top-left edge — cheap to detect, and it would
    quietly halve every IoU otherwise.
    """
    W, H = rkgt.sequence_image_size(DATA_ROOT, SEQ)
    for ann in gt_data["annotations"]:
        x1, y1, x2, y2 = ann["bbox_xyxy"]
        assert x2 > x1 and y2 > y1, f"degenerate box {ann}"
        assert -1.0 <= x1 and -1.0 <= y1, f"box off the top-left edge: {ann}"
        assert x2 <= W + 1.0 and y2 <= H + 1.0, f"box past the image: {ann}"


def test_frames_restricted_to_the_annotated_span(boxes, expression):
    """Frames the expression does not annotate are unannotated, not empty."""
    restricted = rkgt.build_prompt_gt(boxes, expression)
    whole = rkgt.build_prompt_gt(boxes, expression,
                                 restrict_to_labelled_frames=False)
    assert restricted["meta"]["frames"] <= whole["meta"]["frames"]
    labelled = set(expression["valid_ids_by_frame"])
    assert {a["image_id"] for a in restricted["annotations"]} <= labelled


def test_ignore_map_is_rejected_loudly(expression, monkeypatch, tmp_path):
    """No expression in this release uses ``ignore``; if one ever does, the
    adapter must refuse rather than score those boxes as distractors."""
    import json
    path = tmp_path / "0005"
    path.mkdir()
    (path / "fake.json").write_text(json.dumps({
        "label": {"0": [1]}, "ignore": {"0": [2]},
        "video_name": "KITTI_5", "sentence": "fake",
    }))
    monkeypatch.setattr(rkgt, "expression_dir", lambda root, seq: str(path))
    with pytest.raises(NotImplementedError):
        rkgt.load_expression(DATA_ROOT, SEQ, "fake")


# ----------------------------------------------------------------------
# End-to-end controls through the real metric code
# ----------------------------------------------------------------------
def test_oracle_predictions_score_perfectly(gt_data, metrics, tmp_path):
    pred_path = _write_mot(tmp_path / "oracle.txt", gt_data["annotations"],
                           keep_valid=True)
    preds = metrics.load_predictions_mot(str(pred_path))
    sp, sr, pcr, dcr, stats = metrics.compute_metrics(gt_data, preds, 0.5,
                                                      "single_target")
    assert sp == pytest.approx(1.0)
    assert sr == pytest.approx(1.0)
    assert pcr == pytest.approx(1.0)
    assert dcr == pytest.approx(0.0)
    assert stats["total_predictions"] == gt_data["meta"]["num_valid"]

    sid, _ = metrics.compute_semantic_id_switches(gt_data, preds, 0.5,
                                                  "single_target")
    assert sid == 0


def test_distractor_only_predictions_score_as_pure_confusion(gt_data, metrics,
                                                             tmp_path):
    pred_path = _write_mot(tmp_path / "distractors.txt", gt_data["annotations"],
                           keep_valid=False)
    preds = metrics.load_predictions_mot(str(pred_path))
    sp, sr, pcr, dcr, stats = metrics.compute_metrics(gt_data, preds, 0.5,
                                                      "single_target")
    assert sp == pytest.approx(0.0)
    assert sr == pytest.approx(0.0)
    assert pcr == pytest.approx(0.0)
    assert dcr == pytest.approx(1.0)
    assert stats["total_predictions"] == gt_data["meta"]["num_distractors"]


def test_empty_predictions_do_not_divide_by_zero(gt_data, metrics, tmp_path):
    pred_path = tmp_path / "empty.txt"
    pred_path.write_text("")
    preds = metrics.load_predictions_mot(str(pred_path))
    sp, sr, pcr, dcr, _ = metrics.compute_metrics(gt_data, preds, 0.5,
                                                  "single_target")
    assert (sp, sr, pcr, dcr) == (0.0, 0.0, 0.0, 0.0)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def test_prediction_shares_sum_to_one(gt_data, metrics, tmp_path):
    """SP, DCR and the unmatched share partition the predictions.

    The report leans on this identity to split "locked onto the wrong object"
    from "hallucinated an object"; it holds only because every prediction is
    matched against *all* GT, valid and distractor alike.
    """
    lines = []
    for ann in gt_data["annotations"][:400]:
        x1, y1, x2, y2 = ann["bbox_xyxy"]
        lines.append(f"{ann['image_id']},{ann['track_id']},"
                     f"{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1.0,-1,-1,-1")
    # plus some boxes that match nothing
    lines += [f"{f},999,1.0,1.0,8.0,8.0,1.0,-1,-1,-1" for f in range(0, 40)]
    pred_path = tmp_path / "mixed.txt"
    pred_path.write_text("\n".join(lines) + "\n")

    preds = metrics.load_predictions_mot(str(pred_path))
    sp, _, _, dcr, stats = metrics.compute_metrics(gt_data, preds, 0.5,
                                                   "single_target")
    unmatched = (stats["total_predictions"]
                 - stats["predictions_matching_valid"]
                 - stats["predictions_matching_distractor"])
    assert sp + dcr + unmatched / stats["total_predictions"] == pytest.approx(1.0)
    assert unmatched > 0, "control boxes should have matched nothing"

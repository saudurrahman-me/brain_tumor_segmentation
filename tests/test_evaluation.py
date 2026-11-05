import numpy as np
import pytest

from utils.evaluation import dice_score, iou_score, patient_segmentation_metrics


def test_perfect_overlap_scores_one():
    truth = np.array([0, 1, 2, 3], dtype=np.uint8)
    assert dice_score(truth == 1, truth == 1) == pytest.approx(1.0)
    assert iou_score(truth == 3, truth == 3) == pytest.approx(1.0)


def test_empty_region_scores_one_when_both_empty():
    empty = np.zeros((3, 3, 3), dtype=np.uint8)
    assert dice_score(empty, empty) == pytest.approx(1.0)
    assert iou_score(empty, empty) == pytest.approx(1.0)


def test_brats_regions_and_class_metrics():
    truth = np.array([0, 1, 2, 3], dtype=np.uint8)
    pred = np.array([0, 1, 2, 0], dtype=np.uint8)
    metrics = patient_segmentation_metrics(truth, pred)
    assert metrics["WT_dice"] == pytest.approx(0.8)
    assert metrics["TC_dice"] == pytest.approx(2 / 3)
    assert metrics["ET_dice"] == pytest.approx(0.0)
    assert "edema_dice" in metrics


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        patient_segmentation_metrics(np.zeros((2, 2)), np.zeros((3, 3)))

"""Patient-wise Dice and IoU evaluation for BraTS-style segmentation labels."""
import json
import os
import numpy as np


REGIONS = {
    "WT": lambda mask: mask > 0,
    "TC": lambda mask: np.isin(mask, (1, 3)),
    "ET": lambda mask: mask == 3,
}
CLASS_NAMES = {1: "necrotic_core", 2: "edema", 3: "enhancing_tumor"}


def dice_score(y_true, y_pred, epsilon=1e-7):
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    denominator = int(y_true.sum()) + int(y_pred.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(y_true, y_pred).sum() / (denominator + epsilon))


def iou_score(y_true, y_pred, epsilon=1e-7):
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    union = int(np.logical_or(y_true, y_pred).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(y_true, y_pred).sum() / (union + epsilon))


def patient_segmentation_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: truth {y_true.shape}, prediction {y_pred.shape}")
    result = {}
    for name, region in REGIONS.items():
        truth_region = region(y_true)
        pred_region = region(y_pred)
        result[f"{name}_dice"] = dice_score(truth_region, pred_region)
        result[f"{name}_iou"] = iou_score(truth_region, pred_region)
    for label, name in CLASS_NAMES.items():
        result[f"{name}_dice"] = dice_score(y_true == label, y_pred == label)
    return result


def evaluate_model(model, generator, output_path=None):
    """Evaluate every patient in a generator and return mean/std per metric."""
    patient_results = []
    for batch_index in range(len(generator)):
        x_batch, y_batch = generator[batch_index]
        predictions = model.predict(x_batch, verbose=0)
        y_pred = np.argmax(predictions, axis=-1)
        y_true = np.argmax(y_batch, axis=-1)
        start = batch_index * generator.batch_size
        batch_ids = generator.patient_ids[
            generator.indexes[start:start + len(x_batch)]
        ]
        for patient_id, truth, prediction in zip(batch_ids, y_true, y_pred):
            metrics = patient_segmentation_metrics(truth, prediction)
            patient_results.append({"patient_id": patient_id, **metrics})
    if not patient_results:
        raise ValueError("The evaluation generator contains no patients")
    metric_names = [key for key in patient_results[0] if key != "patient_id"]
    summary = {
        "patients_evaluated": len(patient_results),
        "per_metric": {
            name: {
                "mean": float(np.mean([item[name] for item in patient_results])),
                "std": float(np.std([item[name] for item in patient_results])),
            }
            for name in metric_names
        },
        "per_patient": patient_results,
    }
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    return summary

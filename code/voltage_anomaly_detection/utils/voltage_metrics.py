"""
Voltage Anomaly Detection Metrics Module

This module provides specialized evaluation metrics for rural power grid
voltage anomaly detection, including:
- Point adjustment strategy for event-level evaluation
- Segment-level F1 score
- Comprehensive voltage anomaly metrics

Reference: Xu et al., "Anomaly Transformer: Time Series Anomaly Detection
with Association Discrepancy" (ICLR 2022)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, average_precision_score
)


def point_adjustment(pred, gt, threshold=0.5):
    """
    Point Adjustment Strategy for anomaly detection evaluation.

    This strategy addresses the issue where detecting any point within an
    anomaly segment should be considered a successful detection. If a model
    detects at least one point within an anomaly segment, all points in that
    segment are counted as correctly detected.

    Args:
        pred: np.ndarray, predicted anomaly scores or binary predictions
        gt: np.ndarray, ground truth binary labels (0: normal, 1: anomaly)
        threshold: float, threshold for converting scores to binary predictions

    Returns:
        adjusted_pred: np.ndarray, adjusted binary predictions
        adjusted_gt: np.ndarray, ground truth (unchanged)
    """
    pred = np.array(pred).flatten()
    gt = np.array(gt).flatten()

    # Convert scores to binary predictions if needed
    if pred.max() > 1 or pred.min() < 0:
        # Normalize to [0, 1] range
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    pred_binary = (pred > threshold).astype(int)

    # Find anomaly segments in ground truth
    anomaly_state = False
    adjusted_pred = pred_binary.copy()

    for i in range(len(gt)):
        if gt[i] == 1 and not anomaly_state:
            # Start of anomaly segment
            anomaly_state = True
            segment_start = i
        elif gt[i] == 0 and anomaly_state:
            # End of anomaly segment
            anomaly_state = False
            segment_end = i
            # Check if any prediction in segment is positive
            if np.any(pred_binary[segment_start:segment_end] == 1):
                adjusted_pred[segment_start:segment_end] = 1

    # Handle case where anomaly extends to end
    if anomaly_state:
        if np.any(pred_binary[segment_start:] == 1):
            adjusted_pred[segment_start:] = 1

    return adjusted_pred, gt


def find_segments(labels):
    """
    Find contiguous segments of anomalies in labels.

    Args:
        labels: np.ndarray, binary labels

    Returns:
        List of tuples (start_idx, end_idx) for each segment
    """
    segments = []
    labels = np.array(labels).flatten()

    in_segment = False
    for i in range(len(labels)):
        if labels[i] == 1 and not in_segment:
            in_segment = True
            start = i
        elif labels[i] == 0 and in_segment:
            in_segment = False
            segments.append((start, i))

    if in_segment:
        segments.append((start, len(labels)))

    return segments


def segment_f1(pred, gt, overlap_threshold=0.5):
    """
    Segment-level F1 score for anomaly detection.

    A predicted segment is considered a true positive if it overlaps with
    a ground truth segment by at least `overlap_threshold`.

    Args:
        pred: np.ndarray, binary predictions
        gt: np.ndarray, ground truth binary labels
        overlap_threshold: float, minimum IoU for matching segments

    Returns:
        dict with segment-level precision, recall, and F1 score
    """
    pred = np.array(pred).flatten()
    gt = np.array(gt).flatten()

    pred_segments = find_segments(pred)
    gt_segments = find_segments(gt)

    if len(gt_segments) == 0:
        if len(pred_segments) == 0:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        else:
            return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0}

    if len(pred_segments) == 0:
        return {'precision': 1.0, 'recall': 0.0, 'f1': 0.0}

    # Match predicted segments to ground truth segments
    matched_gt = set()
    tp = 0

    for pred_start, pred_end in pred_segments:
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, (gt_start, gt_end) in enumerate(gt_segments):
            # Calculate IoU
            inter_start = max(pred_start, gt_start)
            inter_end = min(pred_end, gt_end)
            intersection = max(0, inter_end - inter_start)

            union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
            iou = intersection / union if union > 0 else 0

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= overlap_threshold and best_gt_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_gt_idx)

    precision = tp / len(pred_segments) if len(pred_segments) > 0 else 0
    recall = len(matched_gt) / len(gt_segments) if len(gt_segments) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1}


def voltage_anomaly_metrics(pred_scores, gt_labels, threshold=None, anomaly_ratio=0.01):
    """
    Comprehensive evaluation metrics for voltage anomaly detection.

    Args:
        pred_scores: np.ndarray, predicted anomaly scores (reconstruction error)
        gt_labels: np.ndarray, ground truth binary labels
        threshold: float, optional fixed threshold for binary classification
                   If None, uses percentile based on anomaly_ratio
        anomaly_ratio: float, expected anomaly ratio for threshold calculation

    Returns:
        dict with comprehensive evaluation metrics including:
        - Point-wise metrics (accuracy, precision, recall, F1)
        - Adjusted metrics (with point adjustment)
        - Segment-level metrics
        - ROC-AUC and PR-AUC (when applicable)
    """
    pred_scores = np.array(pred_scores).flatten()
    gt_labels = np.array(gt_labels).flatten()

    # Ensure same length
    min_len = min(len(pred_scores), len(gt_labels))
    pred_scores = pred_scores[:min_len]
    gt_labels = gt_labels[:min_len]

    # Calculate threshold if not provided
    if threshold is None:
        threshold = np.percentile(pred_scores, 100 * (1 - anomaly_ratio))

    # Binary predictions
    pred_binary = (pred_scores > threshold).astype(int)

    # Point-wise metrics
    accuracy = accuracy_score(gt_labels, pred_binary)
    precision = precision_score(gt_labels, pred_binary, zero_division=0)
    recall = recall_score(gt_labels, pred_binary, zero_division=0)
    f1 = f1_score(gt_labels, pred_binary, zero_division=0)

    # Point adjustment metrics
    adjusted_pred, _ = point_adjustment(pred_binary, gt_labels, threshold=0.5)
    adj_accuracy = accuracy_score(gt_labels, adjusted_pred)
    adj_precision = precision_score(gt_labels, adjusted_pred, zero_division=0)
    adj_recall = recall_score(gt_labels, adjusted_pred, zero_division=0)
    adj_f1 = f1_score(gt_labels, adjusted_pred, zero_division=0)

    # Segment-level metrics
    seg_metrics = segment_f1(pred_binary, gt_labels)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(gt_labels, pred_binary, labels=[0, 1]).ravel()

    # Build result dictionary
    results = {
        # Point-wise metrics
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,

        # Point-adjusted metrics
        'adj_accuracy': adj_accuracy,
        'adj_precision': adj_precision,
        'adj_recall': adj_recall,
        'adj_f1': adj_f1,

        # Segment-level metrics
        'seg_precision': seg_metrics['precision'],
        'seg_recall': seg_metrics['recall'],
        'seg_f1': seg_metrics['f1'],

        # Detection statistics
        'threshold': threshold,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'total_anomalies': int(np.sum(gt_labels)),
        'detected_anomalies': int(np.sum(pred_binary)),
    }

    # ROC-AUC (requires positive and negative samples)
    if len(np.unique(gt_labels)) > 1:
        try:
            results['roc_auc'] = roc_auc_score(gt_labels, pred_scores)
            results['pr_auc'] = average_precision_score(gt_labels, pred_scores)
        except ValueError:
            results['roc_auc'] = 0.0
            results['pr_auc'] = 0.0
    else:
        results['roc_auc'] = 0.0
        results['pr_auc'] = 0.0

    return results


def voltage_quality_analysis(voltage_data, thresholds=None):
    """
    Analyze voltage quality based on power grid standards.

    Args:
        voltage_data: np.ndarray, shape (n_samples, 3) for Va, Vb, Vc
        thresholds: dict, optional custom thresholds

    Returns:
        dict with voltage quality analysis results
    """
    if thresholds is None:
        thresholds = {
            'nominal': 220.0,
            'lower_limit': 198.0,  # -10%
            'upper_limit': 242.0,  # +10%
            'severe_lower': 176.0,  # -20%
            'severe_upper': 264.0,  # +20%
        }

    voltage_data = np.array(voltage_data)
    if voltage_data.ndim == 1:
        voltage_data = voltage_data.reshape(-1, 1)

    # Calculate statistics
    mean_voltage = np.mean(voltage_data, axis=0)
    std_voltage = np.std(voltage_data, axis=0)
    min_voltage = np.min(voltage_data, axis=0)
    max_voltage = np.max(voltage_data, axis=0)

    # Calculate deviation from nominal
    deviation_percent = (mean_voltage - thresholds['nominal']) / thresholds['nominal'] * 100

    # Count violations
    undervoltage_count = np.sum(voltage_data < thresholds['lower_limit'], axis=0)
    overvoltage_count = np.sum(voltage_data > thresholds['upper_limit'], axis=0)
    severe_under_count = np.sum(voltage_data < thresholds['severe_lower'], axis=0)
    severe_over_count = np.sum(voltage_data > thresholds['severe_upper'], axis=0)

    # Calculate unbalance factor (if 3-phase)
    if voltage_data.shape[1] >= 3:
        # Simplified unbalance calculation: max deviation from mean
        mean_3phase = np.mean(voltage_data[:, :3], axis=1, keepdims=True)
        max_deviation = np.max(np.abs(voltage_data[:, :3] - mean_3phase), axis=1)
        unbalance_factor = max_deviation / (mean_3phase.flatten() + 1e-8) * 100
        mean_unbalance = np.mean(unbalance_factor)
    else:
        mean_unbalance = 0.0

    return {
        'mean_voltage': mean_voltage.tolist(),
        'std_voltage': std_voltage.tolist(),
        'min_voltage': min_voltage.tolist(),
        'max_voltage': max_voltage.tolist(),
        'deviation_percent': deviation_percent.tolist(),
        'undervoltage_count': undervoltage_count.tolist(),
        'overvoltage_count': overvoltage_count.tolist(),
        'severe_undervoltage_count': severe_under_count.tolist(),
        'severe_overvoltage_count': severe_over_count.tolist(),
        'mean_unbalance_factor': mean_unbalance,
    }


def print_voltage_report(metrics, name="Voltage Anomaly Detection"):
    """
    Print a formatted report of voltage anomaly detection results.

    Args:
        metrics: dict, output from voltage_anomaly_metrics()
        name: str, name of the experiment
    """
    print("\n" + "=" * 60)
    print(f" {name} Results")
    print("=" * 60)

    print("\n[Point-wise Metrics]")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")

    print("\n[Point-Adjusted Metrics]")
    print(f"  Accuracy:  {metrics['adj_accuracy']:.4f}")
    print(f"  Precision: {metrics['adj_precision']:.4f}")
    print(f"  Recall:    {metrics['adj_recall']:.4f}")
    print(f"  F1 Score:  {metrics['adj_f1']:.4f}")

    print("\n[Segment-Level Metrics]")
    print(f"  Precision: {metrics['seg_precision']:.4f}")
    print(f"  Recall:    {metrics['seg_recall']:.4f}")
    print(f"  F1 Score:  {metrics['seg_f1']:.4f}")

    if 'roc_auc' in metrics:
        print("\n[AUC Metrics]")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")

    print("\n[Detection Statistics]")
    print(f"  Threshold: {metrics['threshold']:.4f}")
    print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}")
    print(f"  Total Anomalies: {metrics['total_anomalies']}")
    print(f"  Detected Anomalies: {metrics['detected_anomalies']}")

    print("=" * 60 + "\n")

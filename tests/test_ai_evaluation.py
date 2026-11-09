"""Tests for the AI-detection evaluation metrics.

Verifies that perfect predictions yield perfect metrics, random predictions
yield sensible ranges, and confusion-matrix layout is correct.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.app.ai_detection.evaluation import evaluate_from_arrays
from sentinel.app.ai_detection.dataset import LABEL_AUTHENTIC, LABEL_SYNTHETIC


def test_perfect_predictions():
    # Model is perfectly confident and correct.
    probs = np.array([0.1, 0.2, 0.9, 0.95])  # authentic, authentic, ai, ai
    labels = np.array([LABEL_AUTHENTIC, LABEL_AUTHENTIC, LABEL_SYNTHETIC, LABEL_SYNTHETIC])

    report = evaluate_from_arrays(probs, labels, threshold=0.5)
    assert report.accuracy == 1.0
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0
    assert report.confusion_matrix == [[2, 0], [0, 2]]
    assert report.false_positive_rate == 0.0
    assert report.false_negative_rate == 0.0


def test_all_wrong_predictions():
    probs = np.array([0.9, 0.95, 0.1, 0.2])  # wrong for every item
    labels = np.array([LABEL_AUTHENTIC, LABEL_AUTHENTIC, LABEL_SYNTHETIC, LABEL_SYNTHETIC])

    report = evaluate_from_arrays(probs, labels, threshold=0.5)
    assert report.accuracy == 0.0
    assert report.confusion_matrix == [[0, 2], [2, 0]]
    assert report.false_positive_rate == 1.0
    assert report.false_negative_rate == 1.0


def test_single_class_roc_auc_is_none():
    probs = np.array([0.1, 0.2])
    labels = np.array([LABEL_AUTHENTIC, LABEL_AUTHENTIC])

    report = evaluate_from_arrays(probs, labels, threshold=0.5)
    assert report.roc_auc is None


def test_threshold_changes_predictions():
    probs = np.array([0.4, 0.6])
    labels = np.array([LABEL_AUTHENTIC, LABEL_AUTHENTIC])

    # At threshold 0.5: 0.4 -> authentic (correct), 0.6 -> ai (wrong) => acc 0.5
    report_low = evaluate_from_arrays(probs, labels, threshold=0.5)
    assert report_low.accuracy == 0.5

    # At threshold 0.7: both 0.4 and 0.6 predicted authentic => acc 1.0
    report_high = evaluate_from_arrays(probs, labels, threshold=0.7)
    assert report_high.accuracy == 1.0


def test_as_dict_keys():
    probs = np.array([0.1, 0.9])
    labels = np.array([LABEL_AUTHENTIC, LABEL_SYNTHETIC])
    report = evaluate_from_arrays(probs, labels, threshold=0.5)
    d = report.as_dict()
    expected_keys = {
        "accuracy", "precision", "recall", "f1", "roc_auc",
        "confusion_matrix", "false_positive_rate", "false_negative_rate",
        "n_samples", "n_authentic", "n_ai_generated", "threshold",
    }
    assert set(d.keys()) == expected_keys

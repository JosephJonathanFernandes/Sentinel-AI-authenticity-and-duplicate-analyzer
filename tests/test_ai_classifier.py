"""Tests for the lightweight CNN classifier and inference.

Uses the synthetic dataset generator to train a tiny model quickly on CPU
and verifies:
- forward pass produces a single logit per image
- training runs and saves a checkpoint
- inference produces plausible labels / confidences
- uncertain band works when confidence is low
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sentinel.app.ai_detection.classifier import LightweightCNN, save_model, load_model
from sentinel.app.ai_detection.dataset import generate_synthetic_images, LABEL_AUTHENTIC
from sentinel.app.ai_detection.inference import AIClassifier, LABEL_UNCERTAIN
from sentinel.app.ai_detection.preprocessing import create_data_loaders
from sentinel.app.ai_detection.training import train_model


def test_forward_pass_shape():
    model = LightweightCNN(input_size=128)
    x = torch.randn(2, 3, 128, 128)
    logits = model(x)
    assert logits.shape == (2, 1)


def test_train_and_save_model(tmp_path):
    data_dir = tmp_path / "data"
    split = generate_synthetic_images(data_dir, n_per_class=60, image_size=64)

    model_path = tmp_path / "model.pt"
    history = train_model(
        split,
        model_path=model_path,
        image_size=64,
        batch_size=16,
        epochs=3,
        lr=1e-3,
        patience=2,
        log_every=1,
    )

    assert model_path.exists()
    assert history["epochs_run"] <= 3
    assert history["best_val_loss"] > 0


def test_load_model_and_predict(tmp_path):
    data_dir = tmp_path / "data2"
    split = generate_synthetic_images(data_dir, n_per_class=40, image_size=64)

    model_path = tmp_path / "model2.pt"
    train_model(
        split,
        model_path=model_path,
        image_size=64,
        batch_size=16,
        epochs=2,
        lr=1e-3,
        patience=2,
        log_every=0,
    )

    classifier = AIClassifier(model_path=model_path, image_size=64, uncertainty_threshold=0.60)
    classifier.load()

    # Predict on a test image
    test_path = split.test_images[0]
    result = classifier.predict_path(test_path)

    assert result.label in ("authentic", "ai_generated", "uncertain")
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.ai_probability <= 1.0
    assert result.error is None


def test_predict_nonexistent_file(tmp_path):
    data_dir = tmp_path / "data3"
    split = generate_synthetic_images(data_dir, n_per_class=20, image_size=64)

    model_path = tmp_path / "model3.pt"
    train_model(
        split,
        model_path=model_path,
        image_size=64,
        batch_size=16,
        epochs=1,
        lr=1e-3,
        patience=1,
        log_every=0,
    )

    classifier = AIClassifier(model_path=model_path, image_size=64)
    classifier.load()

    missing = tmp_path / "does_not_exist.png"
    result = classifier.predict_path(missing)
    assert result.label == LABEL_UNCERTAIN
    assert result.error is not None


def test_inference_on_corrupt_file(tmp_path):
    data_dir = tmp_path / "data4"
    split = generate_synthetic_images(data_dir, n_per_class=20, image_size=64)

    model_path = tmp_path / "model4.pt"
    train_model(
        split,
        model_path=model_path,
        image_size=64,
        batch_size=16,
        epochs=1,
        lr=1e-3,
        patience=1,
        log_every=0,
    )

    classifier = AIClassifier(model_path=model_path, image_size=64)
    classifier.load()

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not a real image at all")
    result = classifier.predict_path(corrupt)
    assert result.label == LABEL_UNCERTAIN
    assert result.error is not None

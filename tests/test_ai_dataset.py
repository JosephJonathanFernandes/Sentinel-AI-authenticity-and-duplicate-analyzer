"""Tests for the AI-detection dataset generator.

Verifies that synthetic image generation produces the expected directory
layout, file counts, and that real vs fake images are distinguishable by
simple statistics (a sanity check that the two classes differ).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from sentinel.app.ai_detection.dataset import (
    generate_synthetic_images,
    load_directory_dataset,
    LABEL_AUTHENTIC,
    LABEL_SYNTHETIC,
)


def test_generate_synthetic_images_layout(tmp_path):
    out = tmp_path / "syn_data"
    split = generate_synthetic_images(out, n_per_class=40, image_size=64)

    assert len(split.train_images) > 0
    assert len(split.val_images) > 0
    assert len(split.test_images) > 0

    # Check labels are balanced within each split
    for name, imgs, labels in (
        ("train", split.train_images, split.train_labels),
        ("val", split.val_images, split.val_labels),
        ("test", split.test_images, split.test_labels),
    ):
        n_auth = sum(1 for l in labels if l == LABEL_AUTHENTIC)
        n_syn = sum(1 for l in labels if l == LABEL_SYNTHETIC)
        assert n_auth > 0, f"{name} split has no authentic images"
        assert n_syn > 0, f"{name} split has no synthetic images"

    # All files exist on disk
    for p in split.train_images + split.val_images + split.test_images:
        assert p.exists(), f"Missing generated image: {p}"


def test_real_vs_fake_statistically_different(tmp_path):
    out = tmp_path / "syn_data2"
    split = generate_synthetic_images(out, n_per_class=60, image_size=64)

    # Gather a few images from each class in the test split
    real_vals = []
    fake_vals = []
    for p, l in zip(split.test_images, split.test_labels):
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
        mean_val = arr.mean()
        if l == LABEL_AUTHENTIC:
            real_vals.append(mean_val)
        else:
            fake_vals.append(mean_val)

    real_vals = np.array(real_vals)
    fake_vals = np.array(fake_vals)
    # The two classes should have measurably different mean statistics.
    assert abs(real_vals.mean() - fake_vals.mean()) > 2.0, (
        "Real and fake images are not statistically distinguishable "
        f"(real={real_vals.mean():.1f}, fake={fake_vals.mean():.1f})"
    )


def test_load_directory_dataset_missing_split_is_empty(tmp_path):
    root = tmp_path / "root"
    # Only create train/real with a couple of images; other splits missing.
    (root / "train" / "real").mkdir(parents=True)
    from PIL import Image
    Image.new("RGB", (32, 32), (100, 100, 100)).save(root / "train" / "real" / "a.png")
    Image.new("RGB", (32, 32), (150, 150, 150)).save(root / "train" / "real" / "b.png")

    split = load_directory_dataset(root)
    assert len(split.train_images) == 2
    assert len(split.val_images) == 0
    assert len(split.test_images) == 0
    assert all(l == LABEL_AUTHENTIC for l in split.train_labels)

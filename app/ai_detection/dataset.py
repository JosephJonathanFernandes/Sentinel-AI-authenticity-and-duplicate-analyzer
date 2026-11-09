"""Dataset utilities for AI-generated image detection.

Two dataset sources are supported:

1. ``SyntheticImageDataset`` — procedurally generated images used to
   verify the training / inference pipeline end-to-end without downloading
   anything.  Two styles are produced:

   * ``authentic`` — smooth gradients + camera-like noise + compression
     artefacts.
   * ``synthetic`` — high-frequency patterned noise + sharp synthetic
     textures (a stand-in for model-generated images).

   These are deliberately distinguishable so the pipeline can be unit-tested
   quickly.  They are NOT a substitute for a real research dataset.

2. ``DirectoryImageDataset`` — points at a folder with ``real/`` and
   ``fake/`` sub-folders.  This is how you connect CIFAKE or any other
   real-versus-synthetic dataset.

Dataset licence notes
---------------------
CIFAKE (https://github.com/sany182/CIFAKE) is released for academic /
research use.  Before using it in anything beyond personal study, check its
licence and the terms of any derived models.  Other real-vs-synthetic
datasets (e.g. Diffusion-Image-Detection, GAN-generated sets) carry their
own restrictions — always verify before training or redistributing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

# Label convention: 0 = authentic (camera-originated), 1 = AI-generated.
LABEL_AUTHENTIC = 0
LABEL_SYNTHETIC = 1

LABEL_NAMES = {LABEL_AUTHENTIC: "authentic", LABEL_SYNTHETIC: "ai_generated"}


@dataclass
class DatasetSplit:
    train_images: list[Path]
    train_labels: list[int]
    val_images: list[Path]
    val_labels: list[int]
    test_images: list[Path]
    test_labels: list[int]


def _save_rgb(path: Path, arr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(arr, Image.Image):
        img = arr.convert("RGB")
    else:
        img = Image.fromarray(np.asarray(arr).astype(np.uint8), mode="RGB")
    img.save(path, format="PNG")


def _gaussian_blur_channel(channel: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Simple separable Gaussian blur using numpy (avoids PIL mode issues)."""
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(channel.astype(np.float32), sigma=sigma)


def _camera_like_noise(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add subtle, spatially-correlated noise reminiscent of a camera sensor."""
    noise = rng.normal(0, 6, arr.shape).astype(np.float32)
    # Smooth the noise per channel so it looks like sensor grain.
    noise = np.stack(
        [_gaussian_blur_channel(noise[:, :, c], sigma=2.0) for c in range(3)],
        axis=2,
    )
    out = arr.astype(np.float32) / 255.0 + noise * 0.05
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def _synthetic_pattern(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add high-frequency, regular patterns typical of generative artefacts."""
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    pattern = np.stack(
        [
            0.5 + 0.3 * np.sin(xx * 0.08 + rng.uniform(0, 6)) * 255,
            0.5 + 0.3 * np.cos(yy * 0.08 + rng.uniform(0, 6)) * 255,
            0.5 + 0.2 * np.sin((xx + yy) * 0.05 + rng.uniform(0, 6)) * 255,
        ],
        axis=2,
    ).astype(np.float32)
    out = arr.astype(np.float32) * 0.7 + pattern * 0.3
    return np.clip(out, 0, 255).astype(np.uint8)


def generate_synthetic_images(
    output_dir: Path | str,
    n_per_class: int = 200,
    image_size: int = 128,
    seed: int = 42,
) -> DatasetSplit:
    """Generate a small synthetic train/val/test split.

    Returns a :class:`DatasetSplit` whose image paths live under
    ``output_dir``.  Suitable for smoke-testing the full pipeline without
    downloading any external dataset.
    """
    output_dir = Path(output_dir)
    rng = np.random.default_rng(seed)

    # Split indices
    n = n_per_class
    train_n = int(n * 0.6)
    val_n = int(n * 0.2)
    test_n = n - train_n - val_n

    splits = {
        "train": (0, train_n),
        "val": (train_n, train_n + val_n),
        "test": (train_n + val_n, n),
    }

    result = DatasetSplit([], [], [], [], [], [])

    for split_name, (start, end) in splits.items():
        count = end - start
        if count <= 0:
            continue
        split_dir = output_dir / split_name
        for cls_name, cls_label in (("real", LABEL_AUTHENTIC), ("fake", LABEL_SYNTHETIC)):
            cls_dir = split_dir / cls_name
            for i in range(start, end):
                fname = f"{cls_name}_{i:05d}.png"
                # Base: random smooth gradient blob
                base = np.zeros((image_size, image_size, 3), dtype=np.float32)
                cx, cy = rng.integers(30, image_size - 30, size=2)
                for c in range(3):
                    amp = rng.uniform(80, 200)
                    base[:, :, c] = amp * np.exp(
                        -((np.arange(image_size)[:, None] - cy) ** 2
                        + (np.arange(image_size)[None, :] - cx) ** 2)
                        / (2 * (image_size / 4) ** 2)
                    )
                arr = np.clip(base, 0, 255).astype(np.uint8)

                if cls_label == LABEL_AUTHENTIC:
                    arr = _camera_like_noise(arr, rng)
                    # Mild JPEG-like re-encode simulation via resize blur
                    if rng.random() < 0.5:
                        arr = np.array(
                            Image.fromarray(arr)
                            .resize((image_size // 2, image_size // 2))
                            .resize((image_size, image_size), Image.BILINEAR)
                        )
                else:
                    arr = _synthetic_pattern(arr, rng)
                    if rng.random() < 0.5:
                        arr = ImageOps.autocontrast(Image.fromarray(arr))

                _save_rgb(cls_dir / fname, arr)

                entry = (cls_dir / fname, cls_label)
                if split_name == "train":
                    result.train_images.append(entry[0])
                    result.train_labels.append(entry[1])
                elif split_name == "val":
                    result.val_images.append(entry[0])
                    result.val_labels.append(entry[1])
                else:
                    result.test_images.append(entry[0])
                    result.test_labels.append(entry[1])

    return result


def load_directory_dataset(
    root: Path | str,
    valid_extensions: Optional[Tuple[str, ...]] = None,
) -> DatasetSplit:
    """Load images from a ``root/train/{real,fake}`` (and val/test) layout.

    Expected structure::

        root/
          train/real/*.png
          train/fake/*.png
          val/real/*.png
          val/fake/*.png
          test/real/*.png
          test/fake/*.png

    Points at CIFAKE-extracted folders or any similarly organised real-vs-
    synthetic collection.
    """
    root = Path(root)
    if valid_extensions is None:
        valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

    splits: dict[str, list[Tuple[Path, int]]] = {"train": [], "val": [], "test": []}
    for split in splits:
        for cls_name, cls_label in (("real", LABEL_AUTHENTIC), ("fake", LABEL_SYNTHETIC)):
            cls_dir = root / split / cls_name
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in valid_extensions:
                    splits[split].append((p, cls_label))

    result = DatasetSplit([], [], [], [], [], [])
    for split, items in splits.items():
        for p, label in items:
            if split == "train":
                result.train_images.append(p)
                result.train_labels.append(label)
            elif split == "val":
                result.val_images.append(p)
                result.val_labels.append(label)
            else:
                result.test_images.append(p)
                result.test_labels.append(label)
    return result

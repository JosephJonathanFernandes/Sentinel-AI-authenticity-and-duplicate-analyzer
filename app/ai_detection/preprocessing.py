"""Image preprocessing and data-loading for the AI classifier.

Transforms are intentionally lightweight: resize to 128x128, normalise with
ImageNet-style statistics, optional horizontal flipping for training.  The
full collection is never loaded into RAM at once — PyTorch ``DataLoader``
streams from disk in bounded batches.
"""

from __future__ import annotations

from typing import Optional, Tuple

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .dataset import DatasetSplit


class ImagePathDataset(Dataset):
    """Map-style dataset over (path, label) pairs.

    Images are opened on demand in ``__getitem__`` so memory usage stays
    bounded regardless of collection size.
    """

    def __init__(
        self,
        images: list,
        labels: list,
        transform: Optional[transforms.Compose] = None,
    ):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        from PIL import Image, UnidentifiedImageError

        path = self.images[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            # Return a blank image so the batch is not dropped; the dataset
            # record still carries the path so callers can log the failure.
            img = Image.new("RGB", (128, 128), (0, 0, 0))
            img.load()
            # Attach a marker the training loop can inspect.
            return img, float(label), str(path), True  # corrupted flag
        if self.transform is not None:
            img = self.transform(img)
        return img, float(label), str(path), False


def get_train_transform(image_size: int = 128) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_eval_transform(image_size: int = 128) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_data_loaders(
    split: DatasetSplit,
    image_size: int = 128,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Build PyTorch DataLoaders for train / val / test.

    ``num_workers=0`` keeps things simple and predictable on Windows /
    single-core-limited machines.
    """
    loaders: dict[str, DataLoader] = {}
    if split.train_images:
        train_ds = ImagePathDataset(
            split.train_images, split.train_labels, transform=get_train_transform(image_size)
        )
        loaders["train"] = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    if split.val_images:
        val_ds = ImagePathDataset(
            split.val_images, split.val_labels, transform=get_eval_transform(image_size)
        )
        loaders["val"] = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    if split.test_images:
        test_ds = ImagePathDataset(
            split.test_images, split.test_labels, transform=get_eval_transform(image_size)
        )
        loaders["test"] = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    return loaders

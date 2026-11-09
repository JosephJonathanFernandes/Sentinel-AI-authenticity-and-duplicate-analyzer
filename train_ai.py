"""Train the lightweight AI-detection classifier on synthetic data.

Usage:

    python train_ai.py --data-dir ./sentinel_data --epochs 10

This is a smoke-test / demonstration entrypoint.  For real research, swap
``generate_synthetic_images`` for ``load_directory_dataset`` pointed at a
CIFAKE-extracted folder (see ``sentinel/app/ai_detection/dataset.py``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sentinel.app.ai_detection.dataset import generate_synthetic_images
from sentinel.app.ai_detection.training import train_model
from sentinel.app.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Sentinel AI-detection classifier")
    parser.add_argument("--data-dir", default="./sentinel_ai_data",
                        help="Directory to store generated synthetic dataset")
    parser.add_argument("--model-out", default=str(settings.ai_model_path),
                        help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=10, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=settings.ai_batch_size,
                        help="Batch size")
    parser.add_argument("--image-size", type=int, default=settings.ai_image_size,
                        help="Input image size")
    parser.add_argument("--n-per-class", type=int, default=200,
                        help="Synthetic images per class (total across split)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early-stopping patience")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"[1/3] Generating synthetic dataset at {data_dir} ...")
    split = generate_synthetic_images(
        data_dir,
        n_per_class=args.n_per_class,
        image_size=args.image_size,
    )
    print(
        f"    train={len(split.train_images)} val={len(split.val_images)} "
        f"test={len(split.test_images)}"
    )

    print(f"[2/3] Training for up to {args.epochs} epochs ...")
    history = train_model(
        split,
        model_path=args.model_out,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )
    print(
        f"    done: best_val_loss={history['best_val_loss']:.4f} "
        f"epochs_run={history['epochs_run']}"
    )

    print(f"[3/3] Model saved to {args.model_out}")
    print("\nTraining complete. You can now run inference with the dashboard or API.")


if __name__ == "__main__":
    main()

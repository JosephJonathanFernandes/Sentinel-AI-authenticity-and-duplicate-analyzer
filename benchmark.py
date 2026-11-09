"""CPU performance benchmark for Sentinel's duplicate detection pipeline.

Measures:
- SHA-256 hashing throughput (MB/s)
- Perceptual hash computation time per image
- ORB descriptor computation time per image
- Near-duplicate comparison speed for a synthetic collection

Run with:

    python benchmark.py

All benchmarks run on CPU only and use synthetically generated images so no
external dataset is required.
"""

from __future__ import annotations

import hashlib
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from sentinel.app.core.image_loader import load_image, close_image
from sentinel.app.duplicate_detection.perceptual_hash import compute_hashes
from sentinel.app.duplicate_detection.feature_matching import compute_orb, match_orb
from sentinel.app.duplicate_detection.exact_hash import compute_sha256


def _make_image(path: Path, size: tuple[int, int] = (512, 512), seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, format="PNG")
    return path


def bench_sha256(n_files: int = 20, size_mb: int = 5) -> dict:
    """SHA-256 streaming hash throughput."""
    sizes = []
    times = []
    with tempfile.TemporaryDirectory() as td:
        for i in range(n_files):
            p = Path(td) / f"file_{i}.bin"
            # Create a file of roughly size_mb MB
            data = np.random.default_rng(i).bytes(size_mb * 1024 * 1024)
            p.write_bytes(data)
            sizes.append(p.stat().st_size)

            t0 = time.perf_counter()
            compute_sha256(p)
            times.append(time.perf_counter() - t0)

    total_bytes = sum(sizes)
    total_sec = sum(times)
    mbps = (total_bytes / (1024 * 1024)) / total_sec if total_sec else 0
    return {
        "files": n_files,
        "size_per_file_mb": size_mb,
        "total_mb": total_bytes / (1024 * 1024),
        "total_seconds": round(total_sec, 3),
        "throughput_mb_per_sec": round(mbps, 2),
        "avg_ms_per_file": round((total_sec / n_files) * 1000, 2) if n_files else 0,
    }


def bench_phash(n_images: int = 30) -> dict:
    """Perceptual hash computation time per image."""
    times = []
    with tempfile.TemporaryDirectory() as td:
        paths = [_make_image(Path(td) / f"img_{i}.png", seed=i) for i in range(n_images)]
        for p in paths:
            t0 = time.perf_counter()
            compute_hashes(p)
            times.append(time.perf_counter() - t0)

    return {
        "images": n_images,
        "total_seconds": round(sum(times), 3),
        "avg_ms_per_image": round((sum(times) / n_images) * 1000, 2),
    }


def bench_orb(n_images: int = 30) -> dict:
    """ORB descriptor computation time per image."""
    times = []
    with tempfile.TemporaryDirectory() as td:
        paths = [_make_image(Path(td) / f"orb_{i}.png", seed=i) for i in range(n_images)]
        for p in paths:
            t0 = time.perf_counter()
            compute_orb(p)
            times.append(time.perf_counter() - t0)

    return {
        "images": n_images,
        "total_seconds": round(sum(times), 3),
        "avg_ms_per_image": round((sum(times) / n_images) * 1000, 2),
    }


def bench_orb_matching(n_pairs: int = 20) -> dict:
    """ORB matching time per pair."""
    times = []
    with tempfile.TemporaryDirectory() as td:
        paths = [_make_image(Path(td) / f"match_{i}.png", seed=i) for i in range(n_pairs * 2)]
        for i in range(n_pairs):
            desc_a = compute_orb(paths[i * 2]).descriptors
            desc_b = compute_orb(paths[i * 2 + 1]).descriptors
            t0 = time.perf_counter()
            match_orb(desc_a, desc_b)
            times.append(time.perf_counter() - t0)

    return {
        "pairs": n_pairs,
        "total_seconds": round(sum(times), 3),
        "avg_ms_per_pair": round((sum(times) / n_pairs) * 1000, 2) if n_pairs else 0,
    }


def bench_end_to_end_scan(n_images: int = 50) -> dict:
    """End-to-end: load + phash + orb for a batch of images."""
    total_t0 = time.perf_counter()
    phash_times = []
    orb_times = []
    with tempfile.TemporaryDirectory() as td:
        paths = [_make_image(Path(td) / f"e2e_{i}.png", seed=i) for i in range(n_images)]
        for p in paths:
            t0 = time.perf_counter()
            compute_hashes(p)
            phash_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            compute_orb(p)
            orb_times.append(time.perf_counter() - t0)

    total_sec = time.perf_counter() - total_t0
    return {
        "images": n_images,
        "total_seconds": round(total_sec, 3),
        "avg_ms_per_image": round((total_sec / n_images) * 1000, 2),
        "phash_avg_ms": round((sum(phash_times) / n_images) * 1000, 2),
        "orb_avg_ms": round((sum(orb_times) / n_images) * 1000, 2),
    }


def main() -> None:
    print("=" * 60)
    print("SENTINEL CPU Performance Benchmark")
    print("=" * 60)
    print()

    print("[1/5] SHA-256 hashing benchmark ...")
    r = bench_sha256()
    print(f"    Files: {r['files']} x {r['size_per_file_mb']} MB each")
    print(f"    Total: {r['total_mb']:.1f} MB in {r['total_seconds']} s")
    print(f"    Throughput: {r['throughput_mb_per_sec']} MB/s")
    print(f"    Avg: {r['avg_ms_per_file']} ms/file")
    print()

    print("[2/5] Perceptual hash (pHash/dHash/aHash) benchmark ...")
    r = bench_phash()
    print(f"    Images: {r['images']}")
    print(f"    Total: {r['total_seconds']} s")
    print(f"    Avg: {r['avg_ms_per_image']} ms/image")
    print()

    print("[3/5] ORB descriptor benchmark ...")
    r = bench_orb()
    print(f"    Images: {r['images']}")
    print(f"    Total: {r['total_seconds']} s")
    print(f"    Avg: {r['avg_ms_per_image']} ms/image")
    print()

    print("[4/5] ORB matching benchmark ...")
    r = bench_orb_matching()
    print(f"    Pairs: {r['pairs']}")
    print(f"    Total: {r['total_seconds']} s")
    print(f"    Avg: {r['avg_ms_per_pair']} ms/pair")
    print()

    print("[5/5] End-to-end scan benchmark ...")
    r = bench_end_to_end_scan()
    print(f"    Images: {r['images']}")
    print(f"    Total: {r['total_seconds']} s")
    print(f"    Avg: {r['avg_ms_per_image']} ms/image")
    print(f"    pHash avg: {r['phash_avg_ms']} ms/image")
    print(f"    ORB avg: {r['orb_avg_ms']} ms/image")
    print()

    print("=" * 60)
    print("Benchmark complete. All tests ran on CPU only.")
    print("=" * 60)


if __name__ == "__main__":
    main()

<div align="center">

# 🛡️ Sentinel

**AI Authenticity & Intelligent Duplicate Image Analyzer**

A local-first, CPU-first Python application for detecting duplicate images,
near-duplicates, and AI-generated imagery — with forensic analysis, smart
retention recommendations, safe quarantine management, and a professional
Streamlit dashboard.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![SQLite](https://img.shields.io/badge/DB-SQLite-blue.svg)](https://www.sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 54](https://img.shields.io/badge/Tests-54%20passing-brightgreen.svg)](tests/)
[![Windows 11](https://img.shields.io/badge/OS-Windows%2011-0078D6.svg)](https://www.microsoft.com/windows)
[![CPU Only](https://img.shields.io/badge/Acceleration-CPU%20only-gray.svg)](https://pytorch.org/)

</div>

---

## 📖 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Target Environment](#target-environment)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Dashboard Walkthrough](#dashboard-walkthrough)
- [API Reference](#api-reference)
- [Safety & Ethics](#safety--ethics)
- [Limitations](#limitations)
- [Testing](#testing)
- [Benchmarks](#benchmarks)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🔍 Overview

**Sentinel** is an intelligent image analysis and management platform that
combines two independent detection engines:

1. **Duplicate Detection** — finds byte-identical images (SHA-256) and
   visually similar near-duplicates (perceptual hashing + ORB feature
   matching), even across resized, compressed, cropped, rotated, or
   re-encoded variants.

2. **AI-Generated Image Detection** — classifies images as likely authentic,
   likely AI-generated, or uncertain, using a lightweight CNN trained on CPU
   plus image-forensic features (FFT / DCT / noise / metadata).

The two engines are **kept independent**: an image can be AI-generated *and*
have duplicates; an authentic photo can also be edited or compressed. Their
scores are never merged into a single "authenticity" score.

**All processing is local.** No images are uploaded to external servers. No
paid APIs are required. No dedicated GPU is needed.

---

## ✨ Key Features

| Capability | Details |
|------------|---------|
| **Exact Duplicate Detection** | SHA-256 byte hashing, cross-directory grouping, redundant-storage calculation, highest-resolution primary recommendation |
| **Near-Duplicate Detection** | pHash / dHash / aHash + Hamming distance, ORB keypoint matching (Lowe's ratio test), weighted combined similarity, configurable threshold |
| **AI-Generated Detection** | Lightweight CNN (~0.5M params), explicit "uncertain" confidence band, accuracy/precision/recall/F1/ROC-AUC metrics, Grad-CAM explainability heatmaps |
| **Image Forensics** | EXIF/metadata extraction, FFT/DCT frequency analysis, noise-residual analysis, composite anomaly score |
| **Smart Retention** | Per-group recommendation scoring (resolution, file size, metadata, sharpness) with human-readable explanations |
| **Safe Management** | Quarantine-by-default, explicit confirmation for permanent deletion, restore from quarantine, full audit logging |
| **Interactive Dashboard** | Dark-themed Streamlit UI with 8 sections: Overview, Scan, Exact Duplicates, Near Duplicates, AI Detection, Forensics, Reports, Settings |
| **Reports** | CSV, JSON, and PDF export of scan summaries, duplicate groups, near-duplicate pairs, and AI results |
| **Database** | SQLite via SQLAlchemy 2.0; paths and metadata only — image binaries are never stored |
| **Performance** | Batched scanning, hash caching, incremental scans, progress indicators, graceful corruption handling |

---

## 🏗️ Architecture

Sentinel follows a modular, layered design. Duplicate detection and AI
detection are completely independent engines.

```
sentinel/
├── app/
│   ├── main.py              # FastAPI application & REST endpoints
│   ├── config.py            # Centralized settings & logging
│   ├── core/                # Scanner, image loader, file manager
│   ├── database/            # SQLAlchemy models, session, repository
│   ├── duplicate_detection/ # SHA-256, pHash/dHash/aHash, ORB, similarity engine
│   ├── ai_detection/        # Dataset, preprocessing, CNN, training, inference, evaluation
│   ├── forensics/           # Metadata, FFT/DCT, noise, forensic engine
│   ├── management/          # Recommendations, quarantine, safe deletion
│   ├── reports/             # PDF report generation
│   └── ui/                  # Streamlit dashboard pages
├── train_ai.py              # CLI model training entrypoint
├── benchmark.py             # CPU performance benchmark
├── demo.py                  # End-to-end demonstration
└── tests/                   # 54 unit tests
```

**Design principles:**

- **CPU-first** — every component runs on an Intel i5-class laptop with
  8–16 GB RAM; no dedicated GPU required.
- **Memory-bounded** — images are streamed in batches; the full collection
  is never loaded into RAM at once.
- **Data safety** — quarantine by default, explicit confirmation before
  deletion, audit log of all destructive actions.
- **No fabrication** — AI detection reports "uncertain" when confidence is
  low; never claims 100% certainty.

---

## 💻 Target Environment

| Item | Requirement |
|------|-------------|
| OS | Windows 11 (also runs on Linux/macOS) |
| Python | 3.10+ (developed & tested on 3.14.2) |
| CPU | Intel i5-class or equivalent |
| RAM | 8–16 GB |
| GPU | Not required (CPU-only) |
| Paid APIs | None required |

---

## 📦 Installation

```powershell
# 1. Clone the repository
git clone https://github.com/JosephJonathanFernandes/Sentinel-AI-authenticity-and-duplicate-analyzer.git
cd Sentinel-AI-authenticity-and-duplicate-analyzer

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

All required wheels are available for Python 3.14 on Windows. If you use a
different Python version and a package lacks a wheel, fall back to
`opencv-python-headless` (no GUI backend needed).

---

## 🚀 Quick Start

```powershell
# From the sentinel/ folder

# Run the full test suite (54 tests)
python -m pytest tests/ -v

# Launch the Streamlit dashboard
python run.py
# Then open: http://127.0.0.1:8501

# Or launch the FastAPI backend only
python run.py --api --port 8000
# Then open: http://127.0.0.1:8000/docs

# Train the AI classifier on synthetic data
python train_ai.py

# Run an end-to-end demonstration
python demo.py

# Run the CPU performance benchmark
python benchmark.py

# Run with Docker
docker-compose up --build
# Then open: http://localhost:8501
```

---

## 🖥️ Dashboard Walkthrough

The Streamlit dashboard provides 8 navigation sections:

| Section | Purpose |
|---------|---------|
| **Overview** | Summary cards: total images, exact groups, near-duplicate pairs, storage savings, AI classification counts |
| **Scan Images** | Select a folder, run the scan with progress bar, view indexing and detection results |
| **Exact Duplicates** | Side-by-side previews of byte-identical groups, primary-copy recommendation, quarantine/delete |
| **Near Duplicates** | Visually similar pairs with similarity scores, matching method, Hamming distance |
| **AI Image Detection** | Model status, classification results table, single-image inference with uncertain band |
| **Image Forensics** | Per-image metadata, FFT/DCT frequency features, noise-residual analysis, composite anomaly score |
| **Reports** | CSV, JSON, and PDF report generation with download buttons |
| **Settings** | Configuration paths, detection thresholds, quarantine file management |

### Safety in the UI

- Files are **quarantined by default**, never permanently deleted without explicit action.
- Permanent deletion requires a confirmation dialog.
- All destructive actions are logged to the `user_actions` table in SQLite.

---

## 🔌 API Reference

Sentinel includes a FastAPI backend. Launch with `python run.py --api` and
visit `http://127.0.0.1:8000/docs` for interactive documentation.

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/scan` | POST | Scan a directory for images and detect duplicates |
| `/duplicates/exact` | GET | List exact duplicate groups |
| `/duplicates/near` | GET | List near-duplicate pairs |
| `/dashboard/summary` | GET | Aggregate dashboard metrics |
| `/images` | GET | List indexed images |

---

## 🛡️ Safety & Ethics

- **Local-first:** All image processing happens on your machine. Nothing is
  uploaded to external servers or paid APIs.
- **No silent deletion:** Files are quarantined, not deleted, by default.
- **Explicit confirmation:** Permanent deletion requires user confirmation.
- **Audit trail:** All destructive actions (quarantine, delete, restore) are
  logged in the database.
- **No false certainty:** AI detection results are probabilistic. Low
  confidence yields an "uncertain" label — never a forced guess.
- **Missing metadata ≠ AI-generated:** EXIF absence is treated as supporting
  forensic evidence only, never as proof of AI generation.

---

## ⚠️ Limitations

- **AI detection needs a real dataset for production use.** The classifier
  trains on procedurally generated synthetic images (fast, no download). For
  research-grade results, point `load_directory_dataset` at a CIFAKE-extracted
  folder and run `python train_ai.py`.
- **Metadata analysis is supporting evidence only.** Missing or unusual EXIF
  is flagged as suspicious but is never definitive proof of AI generation.
- **pHash limitation:** Solid/flat-colour images hash identically — a known
  perceptual-hashing artifact. Real photographic content distinguishes normally.
- **Near-duplicate comparison is O(n²)** with a pHash prefilter. Very large
  collections (100k+ images) would benefit from a FAISS index
  (`faiss-cpu` is already installed as an option).
- **`exiftool` is not installed.** Metadata extraction uses Pillow only.
  Install exiftool separately for richer EXIF parsing.
- **AI inference runs on CPU only.** The model is small (~0.5M parameters at
  128×128) so latency is acceptable on an i5-class laptop.
- **Recommendation scoring is heuristic** (resolution + file size + metadata +
  sharpness) and advisory only — the user always makes the final decision.
- **CI** — GitHub Actions runs the full test suite on Python 3.10–3.13 for
  every push and pull request (`.github/workflows/test.yml`).
- **Docker** — `Dockerfile` and `docker-compose.yml` for containerized
  deployment (`docker-compose up --build`).
- **Grad-CAM** — Gradient-based explainability heatmaps show which image
  regions influenced the AI detection prediction
  (`app/ai_detection/explainability.py`).

---

## 🧪 Testing

```powershell
python -m pytest tests/ -v
```

**54 tests** covering:

- SHA-256 exact duplicates across directories
- Resized, compressed, cropped, rotated near-duplicate detection
- Unrelated-image rejection
- ORB keypoint matching
- Synthetic dataset generation (AI module)
- CNN forward pass, training, inference
- Corrupt and missing file handling
- Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Recommendation scoring and ranking
- Quarantine, restore, and safe deletion
- Forensic metadata, frequency, and noise analysis

---

## 📊 Benchmarks

Run `python benchmark.py` for CPU performance numbers. Typical results on an
Intel i5-1235U:

| Benchmark | Result |
|-----------|--------|
| SHA-256 hashing | ~472 MB/s |
| Perceptual hashing | ~17 ms/image |
| ORB descriptors | ~24 ms/image |
| ORB matching | ~1 ms/pair |
| End-to-end scan | ~73 ms/image |
| Grad-CAM heatmap | ~100 ms/image |

---

## 📁 Project Structure

```
sentinel/
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── requirements.txt
├── run.py
├── train_ai.py
├── benchmark.py
├── demo.py
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── sentinel.db           # created at runtime (SQLite)
├── quarantine/           # created at runtime
├── reports/              # created at runtime
├── app/
│   ├── main.py           # FastAPI app + REST endpoints
│   ├── config.py         # settings, paths, logging
│   ├── core/
│   │   ├── scanner.py
│   │   ├── image_loader.py
│   │   └── file_manager.py
│   ├── database/
│   │   ├── models.py
│   │   ├── database.py
│   │   └── repository.py
│   ├── duplicate_detection/
│   │   ├── exact_hash.py
│   │   ├── perceptual_hash.py
│   │   ├── feature_matching.py
│   │   └── similarity_engine.py
│   ├── ai_detection/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   ├── classifier.py
│   │   ├── training.py
│   │   ├── inference.py
│   │   └── evaluation.py
│   ├── forensics/
│   │   ├── metadata.py
│   │   ├── frequency_analysis.py
│   │   ├── noise_analysis.py
│   │   └── forensic_engine.py
│   ├── management/
│   │   ├── recommendations.py
│   │   ├── quarantine.py
│   │   └── deletion.py
│   ├── reports/
│   │   └── pdf_report.py
│   └── ui/
│       ├── common.py
│       ├── dashboard.py
│       ├── scan_page.py
│       ├── duplicate_page.py
│       ├── ai_detection_page.py
│       ├── forensic_page.py
│       ├── reports_page.py
│       └── settings_page.py
├── models/               # trained AI models (runtime)
├── datasets/             # training datasets (runtime)
├── tests/
│   ├── test_exact_duplicate.py
│   ├── test_near_duplicate.py
│   ├── test_ai_dataset.py
│   ├── test_ai_classifier.py
│   ├── test_ai_evaluation.py
│   ├── test_forensics_metadata.py
│   ├── test_forensics_frequency_noise.py
│   ├── test_forensic_engine.py
│   ├── test_management_recommendations.py
│   ├── test_management_quarantine.py
│   └── test_management_deletion.py
├── sample_images/
└── ...
```

---

## 🗺️ Roadmap

- [x] Milestone 1 — Project foundation
- [x] Milestone 2 — Exact & near-duplicate detection
- [x] Milestone 5 — AI-generated image detection (baseline)
- [x] Milestone 6 — Image forensics (metadata, FFT/DCT, noise)
- [x] Milestone 7 — Streamlit dashboard UI
- [x] Milestone 8 — Image management (recommendations, quarantine, safe deletion)
- [x] Milestone 9 — Reports (CSV, JSON, PDF)
- [x] Milestone 10 — Final validation, benchmarking, documentation, demonstration
- [x] CI — GitHub Actions workflow (`.github/workflows/test.yml`)
- [x] Docker — `Dockerfile` + `docker-compose.yml`
- [x] Grad-CAM — Explainability heatmaps for AI detection (`app/ai_detection/explainability.py`)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for
details on how to submit bugs, propose features, and send pull requests.

All contributors are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/), [FastAPI](https://fastapi.tiangolo.com/),
  [PyTorch](https://pytorch.org/), [OpenCV](https://opencv.org/),
  [Pillow](https://python-pillow.org/), [ImageHash](https://github.com/JohannesBuchner/imagehash),
  [SQLAlchemy](https://www.sqlalchemy.org/), [scikit-learn](https://scikit-learn.org/),
  [ReportLab](https://www.reportlab.com/), and [FAISS](https://github.com/facebookresearch/faiss).
- Dataset guidance: CIFAKE and related real-vs-synthetic image collections
  (verify licences before use).

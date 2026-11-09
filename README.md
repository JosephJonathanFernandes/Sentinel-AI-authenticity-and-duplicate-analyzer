# Sentinel — AI Authenticity & Intelligent Duplicate Image Analyzer

Sentinel is a local, CPU-first application for analysing image collections. It
combines two independent detection engines:

1. **Duplicate Detection** — byte-identical (SHA-256) and near-duplicate
   (perceptual hashing + ORB feature matching) image detection.
2. **AI-Generated Image Detection** — a separately trained, lightweight CNN
   classifier plus image-forensic features (FFT / DCT / noise / metadata).

The two engines are kept independent: an image can be AI-generated *and* have
duplicates; an authentic photo can also be edited or compressed. Their scores
are never combined into a single "authenticity" score.

All processing is local. No images are uploaded to external servers. No paid
APIs are required.

## Target environment

- Windows 11
- Python 3.10+ (developed on Python 3.14.2)
- CPU-only (Intel i5-class, 8–16 GB RAM). No dedicated GPU required.
- `torch` CPU build is used.

## Installation

```powershell
# 1. Create a folder for the project and place this repository inside it.
cd C:\Users\Joseph\Desktop\projects\sentinel

# 2. Create and activate a virtual environment.
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

All required wheels are available for Python 3.14 on Windows. If you use a
different Python version and a package lacks a wheel, fall back to
`opencv-python-headless` (no GUI backend needed).

## Running the application

### Option A — Streamlit dashboard (recommended UI)

```powershell
python run.py
# or explicitly:
python run.py --host 127.0.0.1 --port 8501
```

Then open http://127.0.0.1:8501 in your browser.

### Option B — FastAPI backend only

```powershell
python run.py --api --port 8000
```

Then open http://127.0.0.1:8000/docs for the interactive API docs.

### Option C — Run tests

```powershell
python -m pytest tests/ -v
```

## Project structure (Milestones 1 & 2)

```
sentinel/
├── README.md
├── requirements.txt
├── run.py
├── .gitignore
├── sentinel.db                 # created at runtime (SQLite)
├── quarantine/                 # created at runtime
├── reports/                    # created at runtime
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app + REST endpoints
│   ├── config.py               # settings, paths, logging
│   ├── core/
│   │   ├── scanner.py          # directory walker, batched
│   │   ├── image_loader.py     # safe PIL image loading
│   │   └── file_manager.py     # quarantine / restore / safe delete
│   ├── database/
│   │   ├── models.py           # SQLAlchemy ORM models
│   │   ├── database.py         # engine, session, init_db
│   │   └── repository.py       # CRUD / query helpers
│   ├── duplicate_detection/
│   │   ├── exact_hash.py       # SHA-256 exact-duplicate engine
│   │   ├── perceptual_hash.py  # pHash / dHash / aHash + Hamming
│   │   ├── feature_matching.py # ORB keypoints + matching
│   │   └── similarity_engine.py# combined near-duplicate engine
│   ├── ai_detection/           # (Milestone 5 — complete)
│   │   ├── dataset.py          # synthetic + directory dataset loaders
│   │   ├── preprocessing.py    # transforms + PyTorch DataLoaders
│   │   ├── classifier.py       # lightweight CNN + save/load
│   │   ├── training.py         # training loop + early stopping
│   │   ├── inference.py        # single/batch prediction with uncertain band
│   │   └── evaluation.py       # accuracy, precision, recall, F1, ROC-AUC
│   ├── forensics/              # (Milestone 6 — complete)
│   │   ├── metadata.py         # EXIF / file metadata extraction (Pillow)
│   │   ├── frequency_analysis.py   # FFT / DCT frequency features
│   │   ├── noise_analysis.py       # noise-residual features
│   │   └── forensic_engine.py      # orchestrator + composite score
│   ├── management/             # (Milestone 8 — complete)
│   │   ├── recommendations.py  # retention scoring + explanations
│   │   ├── quarantine.py       # quarantine list / restore / delete-from-quarantine
│   │   └── deletion.py         # safe deletion with confirmation + audit logging
│   ├── reports/                # (Milestone 9 — complete)
│   │   ├── __init__.py
│   │   └── pdf_report.py       # PDF report generation (reportlab)
│   └── ui/                     # (Milestone 7 — complete)
│   │   ├── common.py           # shared DB helpers, image display, formatting
│   │   ├── dashboard.py        # main entrypoint + sidebar navigation + overview
│   │   ├── scan_page.py        # directory scanning + progress
│   │   ├── duplicate_page.py   # exact + near duplicates with side-by-side compare
│   │   │                       # + recommendation scores & reasons
│   │   ├── ai_detection_page.py# AI classification + inference
│   │   ├── forensic_page.py    # metadata / FFT-DCT / noise forensics
│   │   ├── reports_page.py     # CSV / JSON / PDF report generation
│   │   └── settings_page.py    # config, quarantine management, safety
├── train_ai.py                 # CLI training entrypoint (Milestone 5)
├── benchmark.py                # CPU performance benchmark (Milestone 10)
├── demo.py                     # End-to-end demonstration (Milestone 10)
├── models/                     # trained AI models (runtime)
├── datasets/                   # training datasets (runtime)
├── tests/
│   ├── __init__.py
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

## What works now (Milestones 1, 2, 5, 6, 7, 8, 9 & 10)

- Project foundation: configuration, logging, SQLite schema, FastAPI app,
  Streamlit entrypoint, virtual-environment-ready `requirements.txt`.
- **Exact duplicate detection**: streaming SHA-256 hashing (files never loaded
  wholly into RAM), hash reuse from the DB (incremental scanning), grouping of
  byte-identical files across directories, redundant-storage calculation,
  highest-resolution primary-copy recommendation.
- **Near-duplicate detection**: pHash, dHash, aHash with Hamming-distance
  similarity; ORB keypoint detection + BFMatcher + Lowe's ratio test; a
  weighted combined similarity score with a configurable threshold; candidate
  prefiltering by pHash to keep comparisons fast.
- **AI-generated image detection (baseline)**: lightweight CNN (~0.5 M params)
  for authentic vs AI-generated classification; synthetic dataset generator for
  pipeline verification; training loop with early stopping; single/batch
  inference with an explicit "uncertain" band; accuracy, precision, recall,
  F1, ROC-AUC, confusion matrix, FPR/FNR metrics.
- **Image forensics (Milestone 6)**: EXIF/metadata extraction via Pillow
  (camera make/model, software, timestamps, orientation, etc.); FFT frequency
  analysis (energy ratios across low/mid/high bands, spectral flatness);
  DCT analysis (energy distribution); noise-residual analysis (std, skew,
  kurtosis, local variance, consistency score).  A forensic engine combines
  all signals into a composite anomaly score (0..1).  All signals are
  supporting evidence only — missing or unusual metadata does NOT prove
  AI generation.
- **Interactive Streamlit dashboard (Milestone 7)**: dark-themed UI with
  sidebar navigation; Overview (summary cards + recent results); Scan Images
  (folder selection + progress bar); Exact Duplicates + Near Duplicates
  (side-by-side image previews, similarity scores, quarantine/delete with
  confirmation); AI Image Detection (classification with uncertain band);
  Image Forensics (metadata/FFT/DCT/noise per image); Reports (CSV/JSON
  export); Settings (paths, thresholds, quarantine management).
- **Image management (Milestone 8)**: retention recommendation engine that
  scores each copy in a duplicate group on resolution, file size, metadata
  completeness, and sharpness — with human-readable explanations for *why*
  each image is recommended. Quarantine manager (list/restore/delete from
  quarantine) and safe deletion manager (validation, confirmation-required
  permanent deletion, audit logging). All destructive actions are logged to
  the `user_actions` table.
- **Database**: SQLite via SQLAlchemy 2.0, tables for images, metadata, exact
  duplicate groups/members, near-duplicate edges, AI results, forensics, scan
  history, and user actions.
- **Safe file management**: quarantine-by-default, restore, optional permanent
  delete, every action logged (audit trail).
- **Tests**: 54 tests covering SHA-256 exact duplicates, resized/cropped/rotated
  near-duplicate detection, ORB matching, synthetic dataset generation, CNN
  forward/training/inference, corrupt/missing file handling, evaluation metrics,
  recommendation scoring, quarantine/restore/delete, and forensic analysis.
  All pass.

## How to run (end-to-end)

```powershell
# From the sentinel/ folder
python -m pytest tests/ -v          # run the full test suite (54 tests)
python run.py                       # launch the Streamlit dashboard
python train_ai.py                  # train the AI classifier on synthetic data
python demo.py                      # end-to-end demonstration
python benchmark.py                 # CPU performance benchmark
```

In the dashboard you can:
1. Point the scanner at a folder of images.
2. View exact duplicate groups with side-by-side paths and sizes.
3. View near-duplicate pairs with similarity scores and the method used.
4. See retention recommendations with scores and reasons for each copy.
5. Quarantine or (explicitly) delete duplicates, with confirmation.
6. Restore or delete files from quarantine.

## Safety rules (engineering rules)

- Images are processed locally; nothing leaves your machine.
- Files are quarantined, not deleted, by default.
- Deletion requires explicit user action; destructive actions are logged.
- Corrupted and unsupported files are handled gracefully (never crash the
  scan).
- The binary image is never stored in the database — only paths and metadata.

## Known limitations (all milestones)

- **AI detection needs a real dataset for production use.** The classifier
  trains on procedurally generated synthetic images (fast, no download). For
  research-grade results, swap `generate_synthetic_images` for
  `load_directory_dataset` pointed at a CIFAKE-extracted folder (see
  `sentinel/app/ai_detection/dataset.py`), then run `python train_ai.py`.
  Missing metadata does not prove an image is AI-generated.
- **Metadata analysis is supporting evidence only.** Missing or unusual EXIF
  is flagged as "suspicious" but is never treated as definitive proof of AI
  generation.
- **Reports (CSV/JSON/PDF)** — Milestone 9 complete. CSV and JSON exports plus
  PDF report generation via ReportLab, all downloadable from the Reports page.
- **Benchmarking & demonstration** — Milestone 10 complete. `benchmark.py`
  measures SHA-256, pHash, ORB, and end-to-end scan throughput on CPU.
  `demo.py` runs an end-to-end pipeline on synthetic images.
- Recommendation scoring is heuristic (resolution + file size + metadata +
  sharpness) and is advisory only — the user always makes the final decision.
- pHash treats all solid/flat-colour images as identical (a known perceptual
  hashing limitation); real-world photo content distinguishes them normally.
- Near-duplicate comparison is O(n²) over the scanned collection with a pHash
  prefilter; very large collections (100k+ images) would benefit from a FAISS
  index (faiss-cpu is already installed as an option).
- `exiftool` binary is not installed; metadata uses Pillow only. Install
  exiftool separately for richer EXIF parsing.
- AI inference runs on CPU only. Model is small (~0.5 M parameters at 128x128)
  so latency is acceptable on an i5-class laptop.

## Next milestones

Milestone 9 — PDF report export (optional), Milestone 10 — final validation,
benchmarking, documentation, demonstration.

# Contributing to Sentinel

Thank you for considering a contribution to Sentinel! This project is a
locally-executable, CPU-first image authenticity and duplicate-detection
platform. We welcome improvements of all kinds.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [License](#license)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork:

   ```powershell
   git clone https://github.com/<your-username>/Sentinel-AI-authenticity-and-duplicate-analyzer.git
   cd Sentinel-AI-authenticity-and-duplicate-analyzer
   ```

3. Create a virtual environment and install dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Run the test suite to verify everything works:

   ```powershell
   python -m pytest tests/ -v
   ```

## How to Contribute

### Bug Fixes

1. Create an issue describing the bug.
2. Fork the repo and create a branch: `git checkout -b fix/issue-description`
3. Add a test that reproduces the bug.
4. Fix the bug and verify the test passes.
5. Submit a pull request.

### New Features

1. Open an issue to discuss the feature first.
2. Implement on a feature branch: `git checkout -b feature/your-feature`
3. Add tests covering the new functionality.
4. Update the README if the feature changes usage.
5. Submit a pull request.

### Documentation

Improvements to the README, docstrings, or comments are always welcome.
No code changes required — just submit a PR with the documentation updates.

## Coding Standards

- **Python 3.10+** with type hints on all public functions.
- Follow PEP 8 where reasonable (4-space indentation, 100-char line limit).
- Docstrings on all modules, classes, and public methods (Google/NumPy style).
- Meaningful variable and function names — no single-letter names except in
  tight loops.
- No placeholder functions for core features (engineering rule).
- Never fabricate AI detection results; never claim an untrained model is
  operational.
- Never silently delete user files — quarantine by default, confirm before
  deletion.
- Keep engines independent: duplicate detection and AI detection never share
  scores.

## Testing

- Run the full suite before submitting: `python -m pytest tests/ -v`
- Add tests for any new functionality.
- Use synthetic test images (generated in `tests/`); do not commit real user
  images.
- Tests must run on CPU only and complete in reasonable time.

## Pull Request Process

1. Ensure the test suite passes.
2. Update the README if behavior or installation changes.
3. Provide a clear description of the change in the PR body.
4. Link to any related issues.
5. A maintainer will review and merge.

## Reporting Issues

Use GitHub Issues. Include:

- Operating system and Python version.
- Steps to reproduce.
- Expected vs. actual behavior.
- Any error messages or tracebacks.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

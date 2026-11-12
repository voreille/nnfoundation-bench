# CLAUDE.md

Benchmark of the nnFoundation 3D radiology foundation models (arXiv:2609.26924) on
downstream tasks. The current scope is **classification**: reproduce the paper's
protocol, one dataset at a time. KiTS Liver comes first.
Paper notes and open questions: `references/nnfoundation_protocol.md`. Read it
before changing the protocol.

## Environment
- Python via **uv** only: `uv sync`, then `uv run ...`. Do not use pip or create venvs.
- GPUs: 2x H100 NVL. TotalSegmentator (ROI masks) runs on the GPU (`--device gpu:1`, ...).
- `NNFB_DATA_DIR` moves the data root. `TOTALSEG_HOME_DIR` defaults to `models/totalsegmentator`.

## Commands
- `make requirements | test | lint | format`
- `make data TASK=kits_liver [DEVICE=gpu:0]` runs download + prepare.
- `uv run nnfb-data list | download <task> | prepare <task> [--device ... --seed ... --overwrite]`

## Layout (Cookiecutter Data Science)
- `data/raw/<source>/`: immutable downloads, pinned to fixed revisions and checksummed where possible.
- `data/interim/<source>/totalseg/`: cached ROI masks, shared by all tasks of one source.
- `data/processed/<task>/`: `images/`, `masks/`, `labels.csv`, `splits.json`, `dataset.json`.
- `nnfoundation_bench/`: the package. `config.py` holds paths and protocol constants
  (1 mm, 192^3, 40/10/50, seed). `datasets/` has one module per source plus the
  `TASKS` registry. `preprocessing.py` holds the shared crop and split code. `roi.py`
  wraps TotalSegmentator. `dataset.py` is the typer CLI.
- Training and evaluation code goes in `nnfoundation_bench/modeling/` (`train.py`, `predict.py`, not written yet).
- Notebooks: `notebooks/<n>-<initials>-<topic>.ipynb`. Exploration only; reusable code goes in the package.

## Adding a classification dataset
1. Create `nnfoundation_bench/datasets/<source>.py` with `download()`, `image_path(case_id)`,
   a label function returning `case_id,label`, and a `ClassificationTask` per target.
2. Pin download revisions (commit SHA / checksum). Make downloads resumable and idempotent.
3. Register the task in `datasets/__init__.py::TASKS`.
4. Document the label definition, and mark it ASSUMPTION if the paper does not specify it.
   Add a row to the README task table.
5. Add a unit test for the label mapping (no network, no real data in tests).

## Conventions
- Keep preprocessing protocol-faithful. Record any deviation from the paper in
  `references/nnfoundation_protocol.md`.
- Processed CT is stored as int16 HU and is not normalised. Normalisation belongs to the training pipeline.
- Crops keep the source direction cosines. Outside-FOV CT is padded with -1024.
- Ruff, line length 99. Run `make lint test` before committing.
- Git: commit locally, never push. **No Claude/AI attribution** in commits or PRs
  (no `Co-Authored-By: Claude`, no "Generated with Claude Code").

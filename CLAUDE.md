# CLAUDE.md

Benchmark of the nnFoundation 3D radiology foundation models (arXiv:2609.26924) on
downstream tasks. The current scope is **classification**: reproduce the paper's
protocol, one dataset at a time. Tasks so far: KiTS Liver (paper reproduction) and
HECKTOR HPV (PET, GTVp; our own baseline, not in the paper).
**Focus on datasets that ship their own ROI masks**: the user's pipeline relies on ROIs.
**Purpose:** we want to use the *pretrained* nnFoundation models in our own pipeline. The
benchmark is a testbed for design choices around them, e.g. how to pool/average
features within a ROI. Pretrained runs are the default; from-scratch baselines are
optional context, not the goal.
Fine-tuning uses the authors' trainer, **glovita**
(github.com/MIC-DKFZ/image_classification, pinned in pyproject.toml). We prepare the
data and launch and evaluate runs; we don't reimplement training.
**Target: HECKTOR HPV at AUROC >= 0.99** (centre-only reference is 0.84).
Paper notes and open questions: `references/nnfoundation_protocol.md`. Read it
before changing the protocol.
**Keep `docs/` up to date as part of the work, not afterwards:**
- `docs/progress.md`: status table, next steps, results tables (mean ± std AUROC per run),
  and a dated log. Update it whenever a task, run or result changes.
- `docs/decisions.md`: numbered decision records (choice / why / rejected / open). Add a
  new Dn for every design choice. Never rewrite an old entry; supersede it.

## Environment
- **Long-running or heavy jobs are run by the user on the host, not by Claude.** This
  covers full dataset downloads, `prepare` over a whole dataset, training runs, full
  inference, and anything that takes more than a few minutes or uses lots of GPU, disk or
  network. Don't launch them. Give the exact commands (copy-pasteable, with
  `uv run` / `scripts/run_benchmark.sh` and the GPU/fold arguments) and wait for the user to report back.
  Claude may still run quick checks: unit tests, lint, and smoke tests on a few cases or
  a couple of steps.
- Python via **uv** only: `uv sync`, then `uv run ...`. Do not use pip or create venvs.
- GPUs: 2x H100 NVL. torch/torchvision come from the cu128 index because the host
  driver is CUDA 12.9 and PyPI wheels need CUDA 13. TotalSegmentator runs on the GPU (`--device gpu:1`).
- **/dev/shm is 64 MB in claude-box**, too small for DataLoader workers on 192^3 volumes.
  Use `--num-workers 0` for smoke tests in here. Real training runs on the host.
- Memory at batch size 1, 192^3, bf16: ViT peaks at ~48 GB and fits on one H100.
- okapy (the user's PET/CT DICOM conversion library, SUV computation) is an editable path
  dependency on `../okapy`. Fix okapy bugs in okapy, not with workarounds here.
- `NNFB_DATA_DIR` moves the data root. `TOTALSEG_HOME_DIR` defaults to `models/totalsegmentator`.

## Commands
There is no Makefile. Everything is a `uv run` CLI; reruns are cheap because the Python
code skips work that is already done.
- Dev: `uv sync`, `uv run pytest`, `uv run ruff check`, `uv run ruff format`
- `uv run nnfb-data list | download <task> | prepare <task>`
- `uv run nnfb-train <task> --model cnn|vit --fold k [--gpus 0,1 --scratch --lr --epochs --tag]`
- `uv run nnfb-predict predict <task> --model ... --fold k` and `uv run nnfb-predict summarize <task>`
- `uv run nnfb-predict group-baseline <task>`: reference run `group_prior` that predicts from the
  `group` column alone (HECKTOR: centre). Always compare model AUROC against it.
- `scripts/run_benchmark.sh <task>` (host only) runs download, prepare, every
  pretrained model x fold spread over `GPUS` (`SCRATCH=1` adds random-init baselines),
  then the summary. It skips
  runs that already have `metrics.json`. Settings come from env vars (see the script header).

## Layout (Cookiecutter Data Science)
- `data/raw/<source>/`: immutable downloads, pinned to fixed revisions and checksummed where possible.
- `data/raw/HECKTOR_EUVIP2024_HPV/`: symlink to the read-only NAS copy (NFS) of the curated
  HECKTOR DICOM + `hecktor2022_HPV_outcomesBalanced_nospaces.csv` (registration-gated, placed by hand).
- `data/interim/<source>/`: per-source caches shared by its tasks: `totalseg/` (KiTS ROI
  masks), `converted/<case>/` (okapy NIfTI: CT HU, PET SUVbw, one mask per SEG segment + manifest.json).
- `data/interim/<task>/{images,masks}/`: QC crops as NIfTI (float32 in HU/SUV), viewable in ITK-SNAP.
- `data/processed/<task>/`: glovita `generic_3d_dataset` format: `images/*.b2nd` (float32
  (1,z,y,x), z-scored), `labels.json`, `splits.json` (list = folds), `dataset.json`, `labels.csv`,
  plus `masks/*.b2nd` (uint8 ROI on the same grid, for ROI-pooling experiments; glovita ignores it).
- `models/pretrained/`: nnFoundation checkpoints (HF, pinned revisions; downloaded on first use).
- `models/finetuned/<task>/generic_3d_dataset/<run>/<fold>/`: glovita run dirs (`best.pt`, `last.pt`).
  MLflow store: `models/finetuned/<task>/mlflow.db`.
- `reports/results/<task>/<run>/fold<k>/`: `metrics.json` (AUROC), `predictions.csv`; plus `summary.csv`.
- `nnfoundation_bench/`: the package. `config.py` holds paths and protocol constants
  (1 mm, 192^3, 40/10/50, seed). `datasets/` has one module per source plus the
  `TASKS` registry. `preprocessing.py` holds the shared crop and split code. `roi.py`
  wraps TotalSegmentator. `dataset.py` is the typer CLI.
- `weights.py`: pretrained checkpoints and paper LRs (CNN 3e-4, ViT 1e-4).
- `modeling/train.py` builds the `accelerate launch` command for glovita.
  `modeling/glovita_entry.py` wraps glovita's train main with two patches: fixed-length
  epochs (50 x 32 samples) and keeping only best/last checkpoints.
  `modeling/predict.py` runs `glovita.cli.infer` and computes AUROC (glovita's infer
  reports only acc/F1). Run names come from `modeling/runs.py`: `<model>[_scratch]_lr<lr>[_tag]`.
- Notebooks: `notebooks/<n>-<initials>-<topic>.ipynb`. Exploration only; reusable code goes in the package.

## Adding a classification dataset
1. Create `nnfoundation_bench/datasets/<source>.py` with `download()`, a label function returning
   `case_id,label`, a `case_inputs(case_id, device) -> CaseInputs(image, roi_mask, roi_label)`
   (cache any conversion/segmentation under `data/interim/<source>/`), and a
   `ClassificationTask` per (target, modality, ROI) combination. Prefer the dataset's own
   ROI masks (`require_roi=True`: cases without ROI are excluded and listed in dataset.json).
   Multi-centre data: also return `group` (centre) and `stratum` (centre x label) columns.
   Splits then stratify on `stratum`, and `group-baseline` measures confounding.
2. Pin download revisions (commit SHA / checksum). Make downloads resumable and idempotent.
3. Register the task in `datasets/__init__.py::TASKS`.
4. Document the label definition, and mark it ASSUMPTION if the paper does not specify it.
   Add a row to the README task table.
5. Add a unit test for the label mapping (no network, no real data in tests).

## Conventions
- Keep preprocessing protocol-faithful. Record any deviation from the paper in
  `references/nnfoundation_protocol.md`.
- Preprocessing must match the nnFoundation `adaptation_plan.json`: nnU-Net z-score
  (whole-scan mean/std, no mask), 1 mm, 192^3, arrays as SimpleITK (z,y,x).
- Crops are reoriented to an identity (LPS) direction so axis order is the same for every
  case. Outside-FOV CT is padded with -1024 HU before z-scoring.
- Do not edit glovita. If its behaviour needs to change, patch it in `glovita_entry.py`
  and update the pin consciously.
- Ruff, line length 99. Run `uv run ruff format && uv run ruff check && uv run pytest` before committing.
- Git: commit locally, never push. **No Claude/AI attribution** in commits or PRs
  (no `Co-Authored-By: Claude`, no "Generated with Claude Code").

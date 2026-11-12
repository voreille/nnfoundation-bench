# nnfoundation-bench

Benchmarking the nnFoundation 3D radiology foundation models
([arXiv:2609.26924](https://arxiv.org/abs/2609.26924);
weights: [CNN](https://huggingface.co/MIC-DKFZ/nnFoundationCNN),
[ViT](https://huggingface.co/MIC-DKFZ/nnFoundationViT)) on downstream tasks. The
current focus is the **classification** subtask. Fine-tuning uses the authors'
trainer, [glovita](https://github.com/MIC-DKFZ/image_classification/blob/master/docs/nnfoundation.md).

## Quickstart

```bash
uv sync
uv run nnfb-data list                                     # registered tasks
uv run nnfb-data download kits_liver
uv run nnfb-data prepare kits_liver --device gpu:0        # TotalSegmentator ROI, 1 mm, 192^3, splits
uv run nnfb-train kits_liver --model cnn --fold 0 --gpus 0   # add --scratch for the baseline
uv run nnfb-predict predict kits_liver --model cnn --fold 0
uv run nnfb-predict summarize kits_liver                  # mean/std test AUROC over folds
uv run pytest && uv run ruff check && uv run ruff format --check
```

Or run everything (every pretrained model x fold, spread over the GPUs; `SCRATCH=1` also
trains the random-init baselines):

```bash
GPUS="0 1" scripts/run_benchmark.sh kits_liver
```

The script can be rerun safely. Downloads and preprocessing skip files that already
exist, and runs that already have test metrics are skipped. Logs go to `reports/logs/<task>/`.

Training needs a large `/dev/shm` for DataLoader workers (one sample is about 28 MB).
In a small-shm container, pass `--num-workers 0`.

Set `NNFB_DATA_DIR=/big/disk/nnfb` to keep data outside the repo. KiTS23 raw
images are about 100 GB.

## Tasks

| Task         | Source | Cases | ROI   | Label                                            | Status |
|--------------|--------|-------|-------|--------------------------------------------------|--------|
| `kits_liver` | KiTS23 | 489   | liver | Charlson liver-disease comorbidity (mild or moderate/severe) | pipeline smoke-tested; label definition is our assumption |

See [references/nnfoundation_protocol.md](references/nnfoundation_protocol.md) for
the paper's classification protocol and the full list of its datasets.

## Project Organization

```
├── CLAUDE.md          <- Notes for coding agents working on this repo
├── README.md
├── data
│   ├── external       <- Data from third-party sources
│   ├── interim        <- Intermediate data (e.g. TotalSegmentator ROI masks)
│   ├── processed      <- Final per-task datasets used for training/evaluation
│   └── raw            <- Original, immutable downloads
├── models             <- pretrained/ (nnFoundation), finetuned/ (glovita runs), totalsegmentator/
├── notebooks          <- Exploration. Naming: `<number>-<initials>-<description>.ipynb`
├── references         <- Paper notes, protocol, data dictionaries
├── reports
│   ├── figures        <- Generated figures
│   ├── logs           <- run_benchmark.sh logs (not versioned)
│   └── results        <- Per-run/fold metrics.json + predictions.csv, summary.csv
├── pyproject.toml     <- Dependencies and the nnfb-data / nnfb-train / nnfb-predict CLIs
├── scripts            <- run_benchmark.sh: end-to-end run for one task
├── tests
└── nnfoundation_bench <- Source code
    ├── config.py         <- Paths and protocol constants
    ├── dataset.py        <- CLI: `nnfb-data list|download|prepare`
    ├── preprocessing.py  <- ROI-centred 1 mm / 192^3 crops, z-score, .b2nd export, splits
    ├── roi.py            <- TotalSegmentator wrapper
    ├── weights.py        <- nnFoundation checkpoints (pinned HF revisions)
    ├── datasets          <- One module per source dataset + task registry
    └── modeling          <- glovita launch (train.py), evaluation (predict.py)
```

Layout follows [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/).

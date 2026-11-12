# nnfoundation-bench

Benchmarking the nnFoundation 3D radiology foundation models
([arXiv:2609.26924](https://arxiv.org/abs/2609.26924);
weights: [CNN](https://huggingface.co/MIC-DKFZ/nnFoundationCNN),
[ViT](https://huggingface.co/MIC-DKFZ/nnFoundationViT)) on downstream tasks. The
current focus is the **classification** subtask. Fine-tuning uses the authors'
trainer, [glovita](https://github.com/MIC-DKFZ/image_classification/blob/master/docs/nnfoundation.md).

## Quickstart

```bash
make requirements          # uv sync
make tasks                 # list registered tasks
make data TASK=kits_liver  # download + prepare (needs a GPU for TotalSegmentator)
make train MODEL=cnn FOLD=0 GPUS=0
make predict MODEL=cnn FOLD=0
make summarize             # mean/std test AUROC over folds
make test
```

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
├── Makefile           <- `make data`, `make test`, `make lint`, ...
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
│   └── results        <- Per-run/fold metrics.json + predictions.csv, summary.csv
├── pyproject.toml
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

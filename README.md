# nnfoundation-bench

Benchmarking the nnFoundation 3D radiology foundation models
([arXiv:2609.26924](https://arxiv.org/abs/2609.26924)) on downstream tasks. The
current focus is the **classification** subtask.

## Quickstart

```bash
make requirements          # uv sync
make tasks                 # list registered tasks
make data TASK=kits_liver  # download + prepare (needs a GPU for TotalSegmentator)
make test
```

Set `NNFB_DATA_DIR=/big/disk/nnfb` to keep data outside the repo. KiTS23 raw
images are about 100 GB.

## Tasks

| Task         | Source | Cases | ROI   | Label                                            | Status |
|--------------|--------|-------|-------|--------------------------------------------------|--------|
| `kits_liver` | KiTS23 | 489   | liver | Charlson liver-disease comorbidity (mild or moderate/severe) | data prep done; label definition is our assumption |

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
├── models             <- Model weights (foundation checkpoints, TotalSegmentator, fine-tuned)
├── notebooks          <- Exploration. Naming: `<number>-<initials>-<description>.ipynb`
├── references         <- Paper notes, protocol, data dictionaries
├── reports/figures    <- Generated figures
├── pyproject.toml
├── tests
└── nnfoundation_bench <- Source code
    ├── config.py         <- Paths and protocol constants
    ├── dataset.py        <- CLI: `nnfb-data list|download|prepare`
    ├── preprocessing.py  <- ROI-centred 1 mm / 192^3 cropping, stratified splits
    ├── roi.py            <- TotalSegmentator wrapper
    └── datasets          <- One module per source dataset + task registry
```

Layout follows [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/).

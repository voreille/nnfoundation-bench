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
uv run nnfb-features extract kits_liver --model cnn       # frozen encoder, pooled features
uv run nnfb-features probe kits_liver --model cnn         # logistic-regression probe per pooling
uv run nnfb-predict summarize kits_liver                  # mean/std test AUROC over folds
uv run nnfb-radiomics extract kits_liver                  # pyradiomics baseline in the ROI
uv run nnfb-radiomics probe kits_liver --corr-threshold 0.9  # or --classifier enet
uv run nnfb-predict covariate-baseline kits_liver --column age  # reference from covariates alone
uv run nnfb-features fuse task_a task_b --name fused --model vit  # late fusion across modalities
uv run nnfb-cox probe upenn_os_t1gd --model vit          # Cox probe, C-index (survival tasks)
uv run nnfb-cox probe hecktor22_rfs_pet --model covariates --column age --column hpv
uv run nnfb-data prepare hecktor22_rfs_pet_official      # same data, challenge train/test split (D37)
uv run nnfb-predict table                                 # design grid over all tasks (CNN / ViT)
uv run pytest && uv run ruff check && uv run ruff format --check
```

Or run everything (every pretrained model x fold, spread over the GPUs; `SCRATCH=1` also
trains the random-init baselines):

```bash
GPUS="0 1" scripts/run_benchmark.sh kits_liver
```

Frozen-feature probes and radiomics baselines for a prepared task (no fine-tuning):

```bash
scripts/run_probes.sh hecktor_hpv_pet_gtvp
```

The scripts can be rerun safely. Downloads and preprocessing skip files that already
exist, and runs that already have test metrics are skipped. Results go to `outputs/results/<task>/` and logs to `outputs/logs/<task>/` (not versioned).

Training needs a large `/dev/shm` for DataLoader workers (one sample is about 28 MB).
In a small-shm container, pass `--num-workers 0`.

Set `NNFB_DATA_DIR=/big/disk/nnfb` to keep data outside the repo. KiTS23 raw
images are about 100 GB.

## Tasks

| Task | Source | Cases | ROI | Label | Status |
|------|--------|-------|-----|-------|--------|
| `kits_liver` | KiTS23 | 489   | liver (TotalSegmentator) | Charlson liver-disease comorbidity (mild or moderate/severe) | pipeline smoke-tested; label definition is our assumption |
| `hecktor_hpv_pet_gtvp` | HECKTOR (curated EUVIP 2024 HPV subset), PET SUVbw | 102 (52 +) | GTVp (dataset SEG) | HPV positive | ready for prepare; **centre alone gives AUROC 0.84** |
| `mrnet_<abnormal\|acl\|meniscus>_<sagittal\|coronal\|axial>` | MRNet public release (knee MR), one series per task | 1250 (abnormal 1008, ACL 262, meniscus 449) | field of view (no masks) | per target | code + smoke test; spacing assumed (no geometry in the data) |
| `lld_mmri_malignant_<pre\|arterial\|venous\|delay\|t2\|dwi\|inphase\|outphase>` | LLD-MMRI 2023 (MedSAM2 release), liver MR, one series per task | 498 (266 +) | liver lesion (dataset mask, per series) | malignant vs benign (our assumption) | code + smoke test on 2 cases x 3 phases |
| `stoic_<severe\|covid>` | STOIC 2021 public training set, chest CT | 2000 | lungs (TotalSegmentator) | `probSevere` / `probCOVID` | code + smoke test on 1 case; download unfinished |
| `rsna_spine_fracture` | RSNA 2022 cervical spine fracture (Kaggle zip), CT | 2019 | C1-C7 (TotalSegmentator) | `patient_overall` | code + unit tests only; download unfinished |
| `upenn_os_<t1\|t1gd\|t2\|flair>` | UPENN-GBM (TCIA), pre-operative brain MR, one sequence per task; OS labels placed by hand | 452 (226 survived 1 year; all deceased) | whole tumour (expert mask, else automatic) | 1-year overall survival (AUROC) + time/event for the Cox C-index | probes, Cox, fusion done |
| `hecktor22_rfs_<pet\|ct>` | HECKTOR 2022 challenge (training + test pooled, 9 centres), PET SUV or CT HU | 692 (134 recur within 2 years; 113 censored before 2 years dropped) | GTVp (challenge mask) | recurrence within 2 years (AUROC) + time/event for the Cox C-index | code + smoke test on 4 cases x 2 modalities |
| `brats_gbm_<t1c\|t1n\|t2f\|t2w>` | BraTS 2024 post-treatment glioma (Synapse, 5 sites), brain MR, one sequence per task | 632 patients (384 +), earliest scan each | whole tumour (dataset seg, labels 1+2+3) | glioblastoma vs astrocytoma/oligodendroglioma (our assumption) | code + smoke test on 6 cases x 2 sequences |
| `hecktor2022_rfs_<ct\|pet>` | HECKTOR 2022 training + test (NIfTI), CT or PET (SUV), one task per modality | 827 with RFS (174 relapses), 9 centres | GTVp ∪ GTVn (dataset masks; mask keeps 1 = GTVp, 2 = GTVn) | recurrence-free survival (time, event) | code + smoke test on 5 cases |

Progress and results: [docs/progress.md](docs/progress.md). Design decisions:
[docs/decisions.md](docs/decisions.md). See [references/nnfoundation_protocol.md](references/nnfoundation_protocol.md) for
the paper's classification protocol and the full list of its datasets.

## Project Organization

```
├── CLAUDE.md          <- Notes for coding agents working on this repo
├── README.md
├── docs               <- progress.md (status, results, log), decisions.md (design records)
├── data
│   ├── external       <- Data from third-party sources
│   ├── interim        <- Intermediate data (e.g. TotalSegmentator ROI masks)
│   ├── processed      <- Final per-task datasets used for training/evaluation
│   └── raw            <- Original, immutable downloads
├── models             <- pretrained/ (nnFoundation), finetuned/ (glovita runs), totalsegmentator/
├── notebooks          <- Exploration. Naming: `<number>-<initials>-<description>.ipynb`
├── references         <- Paper notes, protocol, data dictionaries
├── outputs            <- Not versioned: results/ (per-run/fold metrics.json, predictions.csv), logs/
├── reports
│   └── figures        <- Generated figures (only what people read goes in reports/)
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
    └── modeling          <- glovita launch (train.py), evaluation (predict.py), frozen probes (features.py)
```

Layout follows [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/).

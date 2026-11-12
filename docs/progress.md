# Progress

Where the benchmark stands, what's next, and results as they come in. Design choices
and their reasons go in [decisions.md](decisions.md). Notes on the paper's protocol go in
[../references/nnfoundation_protocol.md](../references/nnfoundation_protocol.md).

## Goal

Use the pretrained nnFoundation models (CNN and ViT, arXiv:2609.26924) in our own
ROI-based pipeline. This benchmark is where we make design choices around them, such
as input channels, ROI handling, and how to pool features within a ROI.

**Target: HECKTOR HPV classification at AUROC ≥ 0.99.**

## Current status (2026-11-12)

| Task | Data | Pipeline | Results |
|------|------|----------|---------|
| `kits_liver` | 2/489 raw images downloaded | prepare, train and predict smoke-tested | none yet |
| `hecktor_hpv_pet_gtvp` | 102 patients on the NAS, labels found | okapy conversion and crop smoke-tested on 3 patients | centre-only reference: AUROC 0.839 ± 0.011 |
| `mrnet_*` (9 tasks) | 1250 public exams on the NAS; all file sizes and 40 md5s checked | npy-to-NIfTI conversion and crop smoke-tested on 2 exams per plane | none yet |

Next steps:
1. Run `uv run nnfb-data prepare hecktor_hpv_pet_gtvp` on the host, then check
   `dataset.json` for failed cases and cases without GTVp.
2. Frozen-feature probes (D13): `nnfb-features extract` then `probe`, for cnn and vit
   (designs F1-F3 below).
3. Fine-tuning baseline with glovita: `FOLDS=0 scripts/run_benchmark.sh hecktor_hpv_pet_gtvp`
   (design T1).
4. Input-size designs S2-S7: add `--fov` / `--mode` to `nnfb-features extract`, then run
   the FOV sweep.
5. KiTS Liver full run (background, paper reproduction).
6. MRNet: prepare and run `mrnet_abnormal_sagittal` first (closest to the paper's
   "pathology"). Each task needs about 12 GB in `data/processed`. See D12.

## Design benchmark

The designs we compare for using the pretrained models in our pipeline, run on every
dataset to get a broad view. Each cell gives test AUROC as `CNN / ViT`, each value the
mean ± std over the 5 splits.

| Dataset (task) | Modalities | ROI | Default input | Reference |
|---|---|---|---|---|
| HECKTOR HPV (`hecktor_hpv_pet_gtvp`) | PET, CT (same patient, registered) | GTVp (dataset SEG) | PET, 192³ at 1 mm centred on the GTVp | centre only **0.839 ± 0.011**; target ≥ 0.99 |
| MRNet (`mrnet_abnormal_*`) | sagittal T2 FS, coronal T1, axial PD FS (separate series, no shared geometry) | field of view | sagittal, 192³ at 1 mm (whole series) | MRNet paper (all 3 planes): abnormal 0.94, ACL 0.97, meniscus 0.85 (official test set, not comparable to our splits) |
| KiTS Liver (`kits_liver`) | CT | liver (TotalSegmentator), often larger than 192 mm | CT, 192³ at 1 mm centred on the liver | about 5 test positives per split: noisy |

Unless a row says otherwise:
- **Input:** the dataset's default input above, z-scored over the whole scan.
- **Frozen rows:** the probe is a logistic regression (D13).
- **Several modalities:** embeddings are extracted per modality, each from its own
  single-channel task, then fused late (M2-M5, D14).

Status: **ready** = code exists and the run is waiting for the host; **planned** = code
still to write; **done** = results filled in. n/a = the design does not apply.

| ID | Axis | Design | Run name | HECKTOR | MRNet | KiTS Liver | Status |
|----|------|--------|----------|---------|-------|------------|--------|
| F1 | Pooling (frozen) | Mean over the whole crop (what glovita's head sees) | `<m>_frozen_mean_logreg` | | | | ready |
| F2 | Pooling (frozen) | Mean over the ROI, cells weighted by ROI fraction | `<m>_frozen_roi_mean_logreg` | | n/a (ROI = FOV, same as F1 up to padding) | | ready |
| F3 | Pooling (frozen) | Mean over the ROI + 10 mm | `<m>_frozen_roi_margin_mean_logreg` | | n/a | | ready |
| F4 | Pooling (frozen) | ROI mean over earlier CNN stages (16 mm, 8 mm cells); ViT n/a | | | | | planned |
| T1 | Adaptation | glovita full fine-tuning, paper recipe (D1) | `cnn_lr0.0003`, `vit_lr0.0001` | | | | ready |
| S1 | Input size | Fixed 192³ at 1 mm (current preprocessing; same as F1-F3) | see F1-F3 | | | | ready |
| S2 | Input size | Native 1 mm, input size = FOV; ViT absolute position embedding interpolated (glovita default) | | | | | planned |
| S3 | Input size | Same as S2, but the ViT absolute position embedding is cut out (FOV < 192) instead of interpolated, consistent with RoPE; CNN n/a | | | | | planned |
| S4 | Input size | Rescale the FOV to 192³ (spacing = FOV / 192) | | | | | planned |
| S5 | Input size | Tiling: 192³ windows at 1 mm covering the FOV, ROI-weighted mean of features (FOV > 192) | | | n/a (knee fits in 192) | | planned |
| S6 | Input size | ROI-adaptive native: per-case FOV = ROI extent + margin, as in S2 | | | n/a | | planned |
| S7 | Input size | ROI-adaptive rescale: per-case FOV = ROI extent + margin, as in S4 | | | n/a | | planned |
| I1 | Input | Other single modality: CT (HECKTOR), coronal or axial series (MRNet) | | | | n/a | planned |
| I3 | Input | ROI = GTVp ∪ GTVn | | | n/a | n/a | planned |
| M1 | Modality fusion | Early fusion: modalities as channels on one grid, stem weights repeated / n (glovita). Not planned: we extract embeddings per modality (D14) | | n/a | n/a | n/a | dropped |
| M2 | Modality fusion (late) | Feature-level: per-modality embeddings concatenated, one probe | | | | n/a | planned |
| M3 | Modality fusion (late) | Feature-level: mean of the per-modality embeddings (one encoder, so one space), one probe | | | | n/a | planned |
| M4 | Modality fusion (late) | Score-level: one probe per modality, scores combined by a logistic regression fitted on val (the MRNet paper's scheme) | | | | n/a | planned |
| M5 | Modality fusion (late) | Learned attention across the per-modality embeddings or token maps (needs `--save-maps` and a small trained head) | | | | n/a | planned |
| E1 | Evaluation | Leave-one-centre-out (removes the centre shortcut, D9) | | | n/a (single centre) | n/a (single centre) | planned |

### FOV sweep (designs S2-S5)

Frozen features with ROI-mean pooling (F2). The FOV is the cube edge in mm, centred on
the ROI. Each cell: `CNN / ViT` test AUROC (S3: ViT only). HECKTOR first. KiTS Liver is
the large-ROI case (S5 tiling). MRNet's FOV is fixed by the data (about 160 mm).

| Dataset | FOV (mm) | S2 native | S3 native, PE cut (ViT) | S4 rescale | S5 tiling |
|---------|---------:|-----------|-------------------------|------------|-----------|
| HECKTOR | 64  | | | | n/a |
| HECKTOR | 96  | | | | n/a |
| HECKTOR | 128 | | | | n/a |
| HECKTOR | 192 | | n/a | | n/a |
| HECKTOR | 256 | | n/a | | |
| HECKTOR | 320 | | n/a | | |
| KiTS Liver | 192 | | n/a | | n/a |
| KiTS Liver | 256 | | n/a | | |
| KiTS Liver | 320 | | n/a | | |

At 192 mm, S2 and S4 are the same as S1. The S2 ViT at 256-320 mm runs at 32.8k-64k
tokens: fine for frozen inference, too large to fine-tune.

## Results

Test AUROC, mean ± std over the 5 repeated 40/10/50 splits unless noted.

### hecktor_hpv_pet_gtvp

| Run | AUROC | Notes |
|-----|-------|-------|
| `group_prior` (centre only) | 0.839 ± 0.011 | reference, no image; see D9 in decisions.md |

### kits_liver

None yet.

### mrnet_*

None yet.

## Log

### 2026-11-12
- Set up the repo in the Cookiecutter Data Science layout, managed with uv. KiTS23
  download is pinned to fixed versions. KiTS Liver labels come from the clinical
  comorbidity flags, which is our assumption.
- Found the authors' code and weights: nnssl, glovita, and Hugging Face checkpoints.
  Switched fine-tuning to glovita.
- Smoke-tested KiTS end to end on 2 cases. The CNN and ViT both train, and the ViT fits
  at batch size 1 (peak about 48 GB).
- Found and worked around glovita issues: epochs were too short, it wrote a checkpoint
  every epoch, and MLflow rejected its default store.
- Replaced the Makefile with uv CLIs and `scripts/run_benchmark.sh`. The benchmark
  defaults to pretrained models only.
- Shifted focus to datasets that ship their own ROI masks. Added HECKTOR HPV (PET, GTVp),
  converted with okapy. ROI masks are now exported next to the images.
- Found the HPV labels. HPV status is strongly confounded with centre (centre-only
  AUROC 0.84). Splits now stratify on centre × label, and a centre-only baseline was added.
- Added MRNet: 9 tasks (abnormal/ACL/meniscus x sagittal/coronal/axial). The data has no
  spacing and no ROI masks, so we assume 0.625 x 0.625 x 4 mm and use the field of view as
  the ROI (D12). Found that the Redivis label CSVs have no header row.
- Added frozen-feature extraction with ROI poolings and linear probes (`nnfb-features`,
  D13). Smoke-tested both encoders on 2 HECKTOR crops on the GPU: `mean` matches glovita's
  pooled feature, and peak memory is about 3 GB.

- Added a design-benchmark table to this file: pooling, adaptation, input size, input
  channels and evaluation, plus an FOV sweep grid for the input-size designs. Found that
  for inputs other than 192³, glovita interpolates the ViT's absolute position embedding
  but rebuilds RoPE at native token spacing, so the two position signals disagree (S2 vs S3).
- Made the design benchmark multi-dataset (HECKTOR, MRNet, KiTS Liver, one column each)
  and added modality-fusion designs M1-M5. With a frozen encoder, early fusion with
  repeated / n stem weights only encodes the channel average, so M1 is fine-tuning only.
- Decided to always extract embeddings per modality and fuse late (D14). Early fusion (M1)
  is dropped from the plan.


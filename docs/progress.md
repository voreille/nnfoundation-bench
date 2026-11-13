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
| `hecktor_hpv_pet_gtvp` | 102 patients on the NAS, labels found | prepared: 102 cases, 52 positive, no failures | centre-only reference: AUROC 0.839 ± 0.011 |
| `mrnet_*` (9 tasks) | 1250 public exams on the NAS; all file sizes and 40 md5s checked | npy-to-NIfTI conversion and crop smoke-tested on 2 exams per plane | none yet |
| `lld_mmri_malignant_*` (8 phases) | 498 patients x 8 series on the NAS, all files present | crop around the dataset lesion mask smoke-tested on 2 cases x 3 phases | none yet |
| `stoic_severe`, `stoic_covid` | 2000 scans | `stoic_severe` prepared: 2000 cases, 301 positive, no failures | none yet |
| `rsna_spine_fracture` | 2019 studies unpacked on the NAS, checked (961 positive) | conversion, slice order and C1-C7 ROI smoke-tested on 3 studies | none yet |

Next steps:
1. Frozen-feature probes (D13) on the prepared tasks (`hecktor_hpv_pet_gtvp`,
   `stoic_severe`): `nnfb-features extract` then `probe`, for cnn and vit (designs F1-F3).
2. Prepare on the host: the 8 `lld_mmri_malignant_<phase>` tasks, `stoic_covid` (reuses the
   lung masks) and `rsna_spine_fracture`. Then check each `dataset.json` for failed cases.
3. Fine-tuning baseline with glovita: `FOLDS=0 scripts/run_benchmark.sh hecktor_hpv_pet_gtvp`
   (design T1).
4. Input-size designs S2-S7: add `--fov` / `--mode` to `nnfb-features extract`, then run
   the FOV sweep.
5. LLD-MMRI: frozen probes per phase, then late fusion (D14, D15). It has lesion masks, so
   it fits the ROI focus.
6. KiTS Liver full run (background, paper reproduction).
7. MRNet: prepare and run `mrnet_abnormal_sagittal` first (closest to the paper's
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
| STOIC 2021 (`stoic_severe`) | CT | lungs (TotalSegmentator, 3 mm masks, D18); wider than 192 mm | CT, 192³ at 1 mm centred on the lungs | 301/2000 severe; STOIC challenge (severe, hidden test set): about 0.8 |

Unless a row says otherwise:
- **Input:** the dataset's default input above, z-scored over the whole scan.
- **Frozen rows:** the probe is a logistic regression (D13).
- **Several modalities:** embeddings are extracted per modality, each from its own
  single-channel task, then fused late (M2-M5, D14).

Cells can be regenerated with `uv run nnfb-predict table [tasks...]`.

Status: **ready** = code exists and the run is waiting for the host; **planned** = code
still to write; **done** = results filled in. n/a = the design does not apply.

| ID | Axis | Design | Run name | HECKTOR | MRNet | KiTS Liver | STOIC severe | Status |
|----|------|--------|----------|---------|-------|------------|------------|--------|
| F1 | Pooling (frozen) | Mean over the whole crop (what glovita's head sees) | `<m>_frozen_mean_logreg` | 0.514 ± 0.061 / 0.765 ± 0.047 | | | | HECKTOR done; STOIC ready |
| F2 | Pooling (frozen) | Mean over the ROI, cells weighted by ROI fraction | `<m>_frozen_roi_mean_logreg` | 0.459 ± 0.053 / 0.710 ± 0.058 | n/a (ROI = FOV, same as F1 up to padding) | | | HECKTOR done; STOIC ready |
| F3 | Pooling (frozen) | Mean over the ROI + 10 mm | `<m>_frozen_roi_margin_mean_logreg` | 0.498 ± 0.050 / 0.742 ± 0.015 | n/a | | | HECKTOR done; STOIC ready |
| F4 | Pooling (frozen) | ROI mean over earlier CNN stages (16 mm, 8 mm cells); ViT n/a | | | | | | planned |
| T1 | Adaptation | glovita full fine-tuning, paper recipe (D1) | `cnn_lr0.0003`, `vit_lr0.0001` | | | | | ready |
| T1-last | Adaptation | Same runs as T1, evaluated at the last checkpoint (end of the cosine schedule) instead of the best val AUROC (D20) | `cnn_lr0.0003_last`, `vit_lr0.0001_last` | | | | | ready |
| R1 | Baseline (handcrafted) | pyradiomics in the ROI, original image (107 features), same probe and splits (D21); one value per cell | `radiomics_original_logreg` | 0.744 ± 0.053 | | | | HECKTOR done |
| R2 | Baseline (handcrafted) | Same as R1, plus LoG (1, 3 mm) and wavelet images (1037 features) | `radiomics_filtered_logreg` | 0.786 ± 0.008 | | | | HECKTOR done |
| R3 | Baseline (handcrafted) | R2 features, redundant ones dropped per fold (Spearman \|rho\| > 0.9, fit on train; ~305 of 1037 kept), L2 probe (D22) | `radiomics_filtered_corr0.9_logreg` | 0.814 ± 0.018 | | | | HECKTOR done |
| R4 | Baseline (handcrafted) | R2 features, elastic-net probe (L1 share 0.1/0.5/0.9 and C chosen on val) | `radiomics_filtered_enet` | 0.817 ± 0.045 | | | | HECKTOR done |
| R5 | Baseline (handcrafted) | Correlation filter (R3) then elastic net (R4) | `radiomics_filtered_corr0.9_enet` | 0.810 ± 0.044 | | | | HECKTOR done |
| S1 | Input size | Fixed 192³ at 1 mm (current preprocessing; same as F1-F3) | see F1-F3 | | | | | ready |
| S2 | Input size | Native 1 mm, input size = FOV; ViT absolute position embedding interpolated (glovita default) | | | | | | planned |
| S3 | Input size | Same as S2, but the ViT absolute position embedding is cut out (FOV < 192) instead of interpolated, consistent with RoPE; CNN n/a | | | | | | planned |
| S4 | Input size | Rescale the FOV to 192³ (spacing = FOV / 192) | | | | | | planned |
| S5 | Input size | Tiling: 192³ windows at 1 mm covering the FOV, ROI-weighted mean of features (FOV > 192) | | | n/a (knee fits in 192) | | | planned |
| S6 | Input size | ROI-adaptive native: per-case FOV = ROI extent + margin, as in S2 | | | n/a | | | planned |
| S7 | Input size | ROI-adaptive rescale: per-case FOV = ROI extent + margin, as in S4 | | | n/a | | | planned |
| I1 | Input | Other single modality: CT (HECKTOR), coronal or axial series (MRNet) | | | | n/a | n/a (CT only) | planned |
| I3 | Input | ROI = GTVp ∪ GTVn | | | n/a | n/a | n/a | planned |
| M1 | Modality fusion | Early fusion: modalities as channels on one grid, stem weights repeated / n (glovita). Not planned: we extract embeddings per modality (D14) | | n/a | n/a | n/a | n/a | dropped |
| M2 | Modality fusion (late) | Feature-level: per-modality embeddings concatenated, one probe | | | | n/a | n/a | planned |
| M3 | Modality fusion (late) | Feature-level: mean of the per-modality embeddings (one encoder, so one space), one probe | | | | n/a | n/a | planned |
| M4 | Modality fusion (late) | Score-level: one probe per modality, scores combined by a logistic regression fitted on val (the MRNet paper's scheme) | | | | n/a | n/a | planned |
| M5 | Modality fusion (late) | Learned attention across the per-modality embeddings or token maps (needs `--save-maps` and a small trained head) | | | | n/a | n/a | planned |
| E1 | Evaluation | Leave-one-centre-out (removes the centre shortcut, D9) | | | n/a (single centre) | n/a (single centre) | n/a (no centre released) | planned |

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
| `radiomics_filtered_enet` (R4) | 0.817 ± 0.045 | best image-only mean; the chosen C and L1 share vary a lot between folds |
| `radiomics_filtered_corr0.9_logreg` (R3) | 0.814 ± 0.018 | about as good, and more stable |
| `radiomics_filtered_corr0.9_enet` (R5) | 0.810 ± 0.044 | |
| `radiomics_filtered_logreg` (R2) | 0.786 ± 0.008 | |
| `vit_frozen_mean_logreg` (F1) | 0.765 ± 0.047 | best embedding run |
| `radiomics_original_logreg` (R1) | 0.744 ± 0.053 | |
| `radiomics_original_corr0.9_logreg` | 0.706 ± 0.047 | ~33 of 107 features kept |
| `radiomics_original_enet` | 0.689 ± 0.109 | |
| `radiomics_original_corr0.9_enet` | 0.679 ± 0.102 | |
| `vit_frozen_roi_margin_mean_logreg` (F3) | 0.742 ± 0.015 | |
| `vit_frozen_roi_mean_logreg` (F2) | 0.710 ± 0.058 | |
| `cnn_frozen_mean_logreg` (F1) | 0.514 ± 0.061 | chance level, see below |
| `cnn_frozen_roi_margin_mean_logreg` (F3) | 0.498 ± 0.050 | |
| `cnn_frozen_roi_mean_logreg` (F2) | 0.459 ± 0.053 | |

Frozen probes (D13), extracted on 2026-11-13: logistic regression on 40 train cases, C
chosen on 10 val cases, tested on 51. Small, so expect noise.
- **ViT > CNN by a wide margin.** The CNN's pooled last-stage features hardly vary between
  patients (case-to-case correlation 0.998, between-case std about 1.5% of the feature
  magnitude, 30 of 320 dims constant). bf16 is not the cause: its error is 16% of that
  between-case spread (checked against fp32 on 6 cases). Next: earlier CNN stages (F4).
- **ROI pooling does not help yet**: whole crop ≥ ROI + 10 mm ≥ ROI, for both models. With a
  GTVp of a few cm, the ROI covers few cells; the context around the tumour seems to carry
  signal (or centre information, D9).
- **Handcrafted radiomics (D21) is on par with or better than the frozen ViT.** R2 (filtered,
  1037 features) beats every embedding probe; R1 (107 features in the ROI) matches the ViT's
  ROI pooling (0.744 vs 0.710-0.742). Both look at the GTVp only, so the ViT's advantage of
  whole-crop over ROI pooling is not just "more tissue".
- **Handling the redundancy of the filtered set helps (D22):** the correlation filter (R3)
  and the elastic net (R4) both take R2 from 0.786 to about 0.81. The filter is more stable
  across splits. On the 107 original features both hurt (0.68-0.71 vs 0.744). With few
  features, dropping or zeroing them loses information that L2 can still use.
- The probe's chosen C swings from 1e-4 to 10 between folds: 10 val cases cannot pick it
  reliably. This applies to every probe row.
- Nothing beats `group_prior` (0.84). Image features may partly encode the centre too; the
  leave-one-centre-out evaluation (E1) is needed to tell.

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
- Added LLD-MMRI 2023 (8 tasks, one per MR series, ROI = the dataset's MedSAM2 lesion mask,
  target malignant vs benign, D15), STOIC 2021 (`severe`, `covid`, TotalSegmentator lungs,
  D16) and RSNA Spine (`fracture`, DICOM read from the Kaggle zip, TotalSegmentator C1-C7,
  D17). ROIs can now be a union of labels. LLD-MMRI case IDs end with the lesion category,
  so they carry the label. STOIC and RSNA were still downloading.
- STOIC prepare ran at 1-2 min per case. TotalSegmentator's fast model now gets a 3 mm
  input, resampled with SimpleITK, through an uncompressed temporary NIfTI. Per case it
  now takes ~18 s instead of ~50 s, with the same lung centre (< 2 mm apart). Masks made
  before the change stay valid: they are resampled in physical space either way. The masks
  are close to, but not identical to, TotalSegmentator's own output (D18).

### 2026-11-13
- `hecktor_hpv_pet_gtvp` and `stoic_severe` are prepared on the host, with no failed cases
  and no missing ROIs.
- RSNA Spine was unpacked on the NAS. The reader now uses the folders (D19). On 3 studies,
  slice order, spacing and the C1-C7 ROI are correct. About 22% of each scan is scanner
  padding at -2048 to -3669, which distorted the whole-scan z-score in a scanner-dependent
  way, so it is now clamped to -1024 HU.
- Ran the first frozen probes (F1-F3, CNN and ViT) on HECKTOR and added
  `nnfb-predict table`, which prints the design grid from all results. ViT 0.71-0.77,
  CNN at chance; all below the centre-only 0.84. Added a STOIC severe column to the design
  table. Fixed the CLI name: `pyproject.toml` declared `nfb-features`, now `nnfb-features`.
- `nnfb-predict predict --checkpoint last` now writes to `<run>_last`, so it no longer
  overwrites the best-checkpoint results. `run_benchmark.sh` evaluates both, and the
  table has a T1-last row (D20).
- Added the radiomics baseline `nnfb-radiomics` (pyradiomics in the ROI of the 1 mm QC
  crops, same probe and splits as the embeddings, D21). HECKTOR: R1 0.744, R2 0.786, so
  the best image-only result is handcrafted. pyradiomics comes from its GitHub tag v3.1.0,
  because the PyPI 3.1.0 sdist does not build on Python 3.12.
- Radiomics probe options (D22): `--corr-threshold` (Spearman filter fit per fold on
  train) and `--classifier enet` (elastic net). HECKTOR filtered set: corr 0.9 + L2 0.814,
  elastic net 0.817, both up from 0.786. On the original 107 features both hurt.

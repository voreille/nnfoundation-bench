# Progress

Where the benchmark stands, what's next, and results as they come in. Design choices
and their reasons go in [decisions.md](decisions.md). Notes on the paper's protocol go in
[../references/nnfoundation_protocol.md](../references/nnfoundation_protocol.md).

## Goal

Use the pretrained nnFoundation models (CNN and ViT, arXiv:2609.26924) in our own
ROI-based pipeline. This benchmark is where we make design choices around them, such
as input channels, ROI handling, and how to pool features within a ROI.

**Target: HECKTOR HPV classification at AUROC ≥ 0.99.**

## Current status (2026-11-19)

| Task | Data | Pipeline | Results |
|------|------|----------|---------|
| `kits_liver` | 2/489 raw images downloaded | prepare, train and predict smoke-tested | none yet |
| `hecktor_hpv_pet_gtvp` | 102 patients on the NAS, labels found | prepared: 102 cases, 52 positive, no failures | frozen probes, radiomics and centre-only reference done |
| `mrnet_*` (9 tasks) | 1250 public exams on the NAS; all file sizes and 40 md5s checked | npy-to-NIfTI conversion and crop smoke-tested on 2 exams per plane | none yet |
| `lld_mmri_malignant_*` (8 phases) | 498 patients x 8 series on the NAS, all files present | all 8 prepared: 498 cases, 266 malignant, no failures, no missing ROI | frozen probes and radiomics done for all 8 |
| `stoic_severe`, `stoic_covid` | 2000 scans | both prepared: 2000 cases, no failures; `stoic_covid` links the crops, features and radiomics of `stoic_severe` | frozen probes and radiomics done for both |
| `rsna_spine_fracture` | 2019 studies unpacked on the NAS, checked (961 positive) | prepare on the host stopped at 1792 of 2019 (tmux session killed 2026-11-18 08:13); rerun resumes | none yet |
| `upenn_os_*` (4 sequences) | 452 patients on the NAS (`UPENN_GBM`) + OS labels placed by hand (`GBM-UPENN-OS`), all 4 sequences, a mask for each (145 expert) | all 4 prepared on the host: 452 cases, 226 survived 1 year, no failures, no missing ROI; identical labels and splits (D31) | probes, radiomics, age reference, Cox / C-index (D32), fusion (`upenn_os_fused`) done |
| `brats_gbm_*` (4 sequences) | 1809 scans on the NAS, 1621 with seg; metadata xlsx read | all 4 prepared on the host: 632 cases, 384 GBM, no failures, no missing ROI; identical labels and splits (D28) | probes, radiomics, site and age references, fusion (`brats_gbm_fused`) done |

Next steps:
1. Frozen probes and radiomics are done on HECKTOR, both STOIC tasks and the 8 LLD-MMRI
   phases.
2. BraTS: age references and fusion done. Next: a generated age-and-site stratified AUROC
   (see Results) and age-adjusted probes (D29 open).
3. UPENN survival: probes, Cox and fusion done; age alone beats every imaging probe. Next:
   age + imaging probes (AUROC and Cox) to test whether imaging adds anything over age.
4. RSNA: resume `prepare rsna_spine_fracture`, then `RADIOMICS="original"
   scripts/run_probes.sh rsna_spine_fracture`.
5. Fine-tuning baseline with glovita: `FOLDS=0 scripts/run_benchmark.sh hecktor_hpv_pet_gtvp`
   (design T1).
6. Input-size designs S2-S7: add `--fov` / `--mode` to `nnfb-features extract`, then run
   the FOV sweep.
7. LLD-MMRI late fusion over the phases (M2-M4, D14, D15, D30): `nnfb-features fuse` is ready.
8. KiTS Liver full run (background, paper reproduction).
9. MRNet: prepare and run `mrnet_abnormal_sagittal` first (closest to the paper's
   "pathology"). Each task needs about 12 GB in `data/processed`. See D12.

## Backlog

- **Age-and-site stratified AUROC as a generated table** (pairs from the same centre and age
  band, like D24 with a finer group). On BraTS it separated the age shortcut from tumour
  signal (Results). *Trigger:* another task with an `age` column, or reporting BraTS.

Ideas parked until results on more datasets show they are worth it. Each item says what
would trigger it. Move an item to "Next steps" when it is picked up.

- **Nested CV for model selection on small datasets.** Choosing the probe pipeline or its
  hyperparameters on 10 val cases is noise: on HECKTOR the chosen C ranges from 1e-4 to 10
  across splits (D22). Two options:
  - **A.** Keep the 5 outer 40/10/50 splits. Merge train + val and choose the whole
    sklearn pipeline by repeated stratified 5-fold CV inside it (`GridSearchCV`; the
    correlation filter as a transformer, refitted per inner fold). Refit the best on
    train + val, then score the same test cases as today. Runs suffixed `_ncv`. Optional
    one-standard-error rule. Applies to the embedding probes as well as radiomics.
  - **B.** Repeated nested k-fold over all cases (outer 5-fold x 5 repeats, inner CV as
    in A). More training data per fit, but the test sets differ from the fine-tuning splits.
  - *Trigger:* first run the radiomics and embedding probes on LLD-MMRI (498 cases, 50 val)
    and STOIC (2000, 200 val). If val selection is unstable there too, or the gap between
    val and test AUROC is large on small tasks, implement A. B only for HECKTOR / LLD-MMRI.
- **Use `last.pt` and fold val back into train for fine-tuning** if T1-last matches T1
  on small datasets (D20).
- **Earlier CNN stages for pooling (F4):** the last CNN stage barely varies between
  patients on HECKTOR (case-to-case correlation 0.998).
- **Keep TotalSegmentator loaded between cases** (or segment several in parallel): model
  loading is most of the remaining ~18 s per case (D18). *Trigger:* the next big
  TotalSegmentator prepare (RSNA, KiTS).
- **Full TotalSegmentator model (`fast=False`) for RSNA vertebrae:** vertebrae are
  15-20 mm, so 3 mm masks are coarse next to 8 mm ViT cells (D18).
- **STOIC: check that dense consolidation is inside the lung mask** on a few severe
  cases (ITK-SNAP, `data/interim/stoic_severe/`). If it is excluded, ROI pooling drops signal
  more in severe cases than in the others.
- **Silence TotalSegmentator's "Output affine not equal" warning** by snapping the
  temporary 3 mm image's origin to float32-exact values (a sub-micrometre shift). Harmless,
  but about 15 lines per case in the logs.
- **Erase the centre from the features (LEACE, Belrose et al. 2023)** with a linear eraser
  fit per fold on train, then probe. Within-centre AUROC (D24) already removes the centre
  from the *metric*; LEACE would remove it from the *features*, which matters if centre-free
  embeddings are wanted downstream. Needs shrinkage or PCA (41 train cases vs 768-1037
  dims), and also removes real signal that correlates with the centre. *Trigger:* a model
  with clearly better than 0.5 within-centre AUROC on HECKTOR, or a multi-centre task where
  the embeddings feed our own pipeline.

## Design benchmark

The designs we compare for using the pretrained models in our pipeline, run on every
dataset to get a broad view. Each cell of the results grid gives test AUROC as
`CNN / ViT`, each value the mean ± std over the 5 splits.

| Dataset (task) | Modalities | ROI | Default input | Reference |
|---|---|---|---|---|
| HECKTOR HPV (`hecktor_hpv_pet_gtvp`) | PET, CT (same patient, registered) | GTVp (dataset SEG) | PET, 192³ at 1 mm centred on the GTVp | centre only: `ref` row of the grid; target ≥ 0.99 |
| MRNet (`mrnet_abnormal_*`) | sagittal T2 FS, coronal T1, axial PD FS (separate series, no shared geometry) | field of view | sagittal, 192³ at 1 mm (whole series) | MRNet paper (all 3 planes): abnormal 0.94, ACL 0.97, meniscus 0.85 (official test set, not comparable to our splits) |
| KiTS Liver (`kits_liver`) | CT | liver (TotalSegmentator), often larger than 192 mm | CT, 192³ at 1 mm centred on the liver | about 5 test positives per split: noisy |
| STOIC 2021 (`stoic_severe`, `stoic_covid`) | CT | lungs (TotalSegmentator, 3 mm masks, D18); wider than 192 mm | CT, 192³ at 1 mm centred on the lungs | 301/2000 severe, 1205/2000 COVID; STOIC challenge (severe, hidden test set): about 0.8 |
| UPENN-GBM survival (`upenn_os_*`) | T1, T1GD, T2, FLAIR (co-registered, one grid) | whole tumour (expert mask, else automatic) | each sequence, 192³ at 1 mm centred on the tumour | 1-year OS 226/452 (AUROC); C-index grid for the Cox probes; age reference |
| BraTS 2024 glioma (`brats_gbm_*`) | t1c, t1n, t2f, t2w (co-registered, one grid) | whole tumour (dataset seg, labels 1+2+3) | each sequence, 192³ at 1 mm centred on the tumour | 384/632 GBM; 5 sites, GBM share 38-100% by site: `ref` row (site only) |

Unless a row says otherwise:
- **Input:** the dataset's default input above, z-scored over the whole scan.
- **Frozen rows:** the probe is a logistic regression (D13).
- **Several modalities:** embeddings are extracted per modality, each from its own
  single-channel task, then fused late (M2-M5, D14).

Status: **ready** = code exists (the grid below shows which tasks have run); **planned**
= code still to write; **dropped** = not pursued (dropped rows leave the grids; their past
runs stay in the per-task runs tables).

| ID | Axis | Design | Run name | Applies to | Status |
|----|------|--------|----------|------------|--------|
| F1 | Pooling (frozen) | Mean over the whole crop (what glovita's head sees) | `<m>_frozen_mean_logreg` | all | ready |
| F2 | Pooling (frozen) | Mean over the ROI, cells weighted by ROI fraction | `<m>_frozen_roi_mean_logreg` | MRNet n/a (ROI = FOV, same as F1 up to padding) | ready |
| F3 | Pooling (frozen) | Mean over the ROI + 10 mm | `<m>_frozen_roi_margin_mean_logreg` | MRNet n/a | ready |
| F4 | Pooling (frozen) | ROI mean over earlier CNN stages (16 mm, 8 mm cells); ViT n/a |  | all | planned |
| T1 | Adaptation | glovita full fine-tuning, paper recipe (D1) | `cnn_lr0.0003`, `vit_lr0.0001` | all | ready |
| T1-last | Adaptation | Same runs as T1, evaluated at the last checkpoint (end of the cosine schedule) instead of the best val AUROC (D20) | `cnn_lr0.0003_last`, `vit_lr0.0001_last` | all | ready |
| R1 | Baseline (handcrafted) | pyradiomics in the ROI, original image (107 features), same probe and splits (D21); one value per cell | `radiomics_original_logreg` | MRNet n/a (no ROI) | ready |
| R2 | Baseline (handcrafted) | Same as R1, plus LoG (1, 3 mm) and wavelet images (1037 features) | `radiomics_filtered_logreg` | MRNet n/a (no ROI) | dropped (D25) |
| R3 | Baseline (handcrafted) | R2 features, redundant ones dropped per fold (Spearman \|rho\| > 0.9, fit on train; ~305 of 1037 kept), L2 probe (D22) | `radiomics_filtered_corr0.9_logreg` | MRNet n/a (no ROI) | dropped (D25) |
| R4 | Baseline (handcrafted) | R2 features, elastic-net probe (L1 share 0.1/0.5/0.9 and C chosen on val) | `radiomics_filtered_enet` | MRNet n/a (no ROI) | dropped (D25) |
| R5 | Baseline (handcrafted) | Correlation filter (R3) then elastic net (R4) | `radiomics_filtered_corr0.9_enet` | MRNet n/a (no ROI) | dropped (D25) |
| S1 | Input size | Fixed 192³ at 1 mm (current preprocessing; same as F1-F3) | see F1-F3 | all | ready |
| S2 | Input size | Native 1 mm, input size = FOV; ViT absolute position embedding interpolated (glovita default) |  | all | planned |
| S3 | Input size | Same as S2, but the ViT absolute position embedding is cut out (FOV < 192) instead of interpolated, consistent with RoPE; CNN n/a |  | all | planned |
| S4 | Input size | Rescale the FOV to 192³ (spacing = FOV / 192) |  | all | planned |
| S5 | Input size | Tiling: 192³ windows at 1 mm covering the FOV, ROI-weighted mean of features (FOV > 192) |  | MRNet n/a (knee fits in 192) | planned |
| S6 | Input size | ROI-adaptive native: per-case FOV = ROI extent + margin, as in S2 |  | MRNet n/a | planned |
| S7 | Input size | ROI-adaptive rescale: per-case FOV = ROI extent + margin, as in S4 |  | MRNet n/a | planned |
| I1 | Input | Other single modality: CT (HECKTOR), coronal or axial series (MRNet) |  | KiTS, STOIC n/a (CT only) | planned |
| I3 | Input | ROI = GTVp ∪ GTVn |  | HECKTOR only | planned |
| M1 | Modality fusion | Early fusion: modalities as channels on one grid, stem weights repeated / n (glovita). Not planned: we extract embeddings per modality (D14) |  | all | dropped |
| M2 | Modality fusion (late) | Feature-level: per-modality embeddings concatenated, one probe (grid: ROI mean; D30) | `<m>_fused_concat_<pooling>_logreg` on a fused task | HECKTOR, MRNet, LLD-MMRI, BraTS | ready |
| M3 | Modality fusion (late) | Feature-level: mean of the per-modality embeddings (one encoder, so one space), one probe (grid: ROI mean) | `<m>_fused_mean_<pooling>_logreg` | HECKTOR, MRNet, LLD-MMRI, BraTS | ready |
| M4 | Modality fusion (late) | Score-level: one probe per modality, scores combined by a logistic regression fitted on val (the MRNet paper's scheme; grid: ROI mean) | `<m>_fused_stack_<pooling>_logreg` | HECKTOR, MRNet, LLD-MMRI, BraTS | ready |
| M5 | Modality fusion (late) | Learned attention across the per-modality embeddings or token maps (needs `--save-maps` and a small trained head) |  | HECKTOR, MRNet, LLD-MMRI | planned |
| ref-age | Reference | Age alone, same probe and splits (D29) | `covariates_age_logreg` | tasks with an `age` column (BraTS) | ready |
| ref-age+site | Reference | Age and centre (one-hot), same probe | `covariates_age_group_logreg` | tasks with `age` and `group` | ready |
| E1 | Evaluation | Leave-one-centre-out (removes the centre shortcut, D9) |  | HECKTOR only (other tasks: single centre or no centre released) | planned |

Results grids, one column per task with results. Generated from `outputs/results/` by
`uv run nnfb-predict table --update docs/progress.md`; do not edit by hand. `-` = not
run yet; `ref` = `group_prior` (centre only), for tasks with centres.

<!-- generated: design hecktor_* stoic_* -->
| ID | hecktor_hpv_pet_gtvp | stoic_covid | stoic_severe |
|----|---|---|---|
| F1 | 0.514 ± 0.061 / 0.765 ± 0.047 | 0.621 ± 0.014 / 0.787 ± 0.029 | 0.656 ± 0.023 / 0.760 ± 0.013 |
| F2 | 0.459 ± 0.053 / 0.710 ± 0.058 | 0.698 ± 0.010 / 0.814 ± 0.012 | 0.697 ± 0.035 / 0.778 ± 0.008 |
| F3 | 0.498 ± 0.050 / 0.742 ± 0.015 | 0.696 ± 0.007 / 0.808 ± 0.017 | 0.707 ± 0.022 / 0.776 ± 0.009 |
| T1 | - / - | - / - | - / - |
| T1-last | - / - | - / - | - / - |
| R1 | 0.744 ± 0.053 | 0.738 ± 0.013 | 0.747 ± 0.016 |
| ref | 0.839 ± 0.011 | n/a | n/a |
<!-- end generated -->

**Within-centre AUROC** (D24), for tasks with centres: only (positive, negative) pairs from
the same centre count, so the centre shortcut is removed and `ref` is 0.5 by construction.
On HECKTOR a test split holds only 66-81 such pairs, more than half from CHUP (3 positives
x 15 negatives), so these values are noisy.

<!-- generated: design-within -->
| ID | hecktor_hpv_pet_gtvp |
|----|---|
| F1 | 0.351 ± 0.087 / 0.490 ± 0.123 |
| F2 | 0.431 ± 0.158 / 0.454 ± 0.036 |
| F3 | 0.428 ± 0.131 / 0.502 ± 0.099 |
| T1 | - / - |
| T1-last | - / - |
| R1 | 0.660 ± 0.100 |
| ref | 0.500 ± 0.000 |
<!-- end generated -->

LLD-MMRI, one column per MR phase (task `lld_mmri_malignant_<phase>`):

<!-- generated: design lld_mmri_* -->
| ID | lld_mmri_malignant_arterial | lld_mmri_malignant_delay | lld_mmri_malignant_dwi | lld_mmri_malignant_inphase | lld_mmri_malignant_outphase | lld_mmri_malignant_pre | lld_mmri_malignant_t2 | lld_mmri_malignant_venous |
|----|---|---|---|---|---|---|---|---|
| F1 | 0.546 ± 0.023 / 0.668 ± 0.082 | 0.548 ± 0.013 / 0.664 ± 0.060 | 0.546 ± 0.021 / 0.696 ± 0.056 | 0.512 ± 0.031 / 0.703 ± 0.055 | 0.562 ± 0.047 / 0.695 ± 0.062 | 0.532 ± 0.057 / 0.693 ± 0.049 | 0.553 ± 0.023 / 0.667 ± 0.056 | 0.522 ± 0.031 / 0.640 ± 0.071 |
| F2 | 0.553 ± 0.033 / 0.746 ± 0.025 | 0.554 ± 0.046 / 0.867 ± 0.014 | 0.579 ± 0.021 / 0.748 ± 0.034 | 0.533 ± 0.046 / 0.760 ± 0.028 | 0.499 ± 0.031 / 0.697 ± 0.022 | 0.558 ± 0.034 / 0.729 ± 0.033 | 0.570 ± 0.023 / 0.808 ± 0.020 | 0.524 ± 0.024 / 0.837 ± 0.021 |
| F3 | 0.547 ± 0.022 / 0.716 ± 0.040 | 0.557 ± 0.052 / 0.778 ± 0.009 | 0.573 ± 0.025 / 0.758 ± 0.014 | 0.565 ± 0.011 / 0.744 ± 0.024 | 0.516 ± 0.028 / 0.696 ± 0.035 | 0.544 ± 0.017 / 0.703 ± 0.022 | 0.560 ± 0.017 / 0.769 ± 0.025 | 0.504 ± 0.045 / 0.749 ± 0.019 |
| T1 | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - |
| T1-last | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - |
| R1 | 0.786 ± 0.040 | 0.823 ± 0.045 | 0.687 ± 0.037 | 0.739 ± 0.034 | 0.709 ± 0.026 | 0.705 ± 0.035 | 0.755 ± 0.037 | 0.828 ± 0.029 |
| ref | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
<!-- end generated -->

BraTS 2024 glioma, one column per MR sequence (task `brats_gbm_<seq>`), GBM vs
astrocytoma/oligodendroglioma. Test AUROC, then within-centre AUROC (D24):

<!-- generated: design brats_* -->
| ID | brats_gbm_fused | brats_gbm_t1c | brats_gbm_t1n | brats_gbm_t2f | brats_gbm_t2w |
|----|---|---|---|---|---|
| F1 | - / - | 0.728 ± 0.011 / 0.839 ± 0.016 | 0.728 ± 0.016 / 0.874 ± 0.015 | 0.725 ± 0.027 / 0.852 ± 0.019 | 0.730 ± 0.028 / 0.877 ± 0.014 |
| F2 | - / - | 0.689 ± 0.031 / 0.851 ± 0.013 | 0.668 ± 0.028 / 0.844 ± 0.015 | 0.730 ± 0.022 / 0.847 ± 0.013 | 0.661 ± 0.053 / 0.817 ± 0.029 |
| F3 | - / - | 0.669 ± 0.016 / 0.834 ± 0.017 | 0.692 ± 0.029 / 0.832 ± 0.018 | 0.739 ± 0.022 / 0.829 ± 0.017 | 0.683 ± 0.020 / 0.855 ± 0.017 |
| T1 | - / - | - / - | - / - | - / - | - / - |
| T1-last | - / - | - / - | - / - | - / - | - / - |
| R1 | - | 0.778 ± 0.046 | 0.756 ± 0.018 | 0.756 ± 0.020 | 0.759 ± 0.018 |
| M2 | 0.749 ± 0.007 / 0.864 ± 0.013 | - / - | - / - | - / - | - / - |
| M3 | 0.710 ± 0.044 / 0.851 ± 0.011 | - / - | - / - | - / - | - / - |
| M4 | 0.747 ± 0.027 / 0.854 ± 0.013 | - / - | - / - | - / - | - / - |
| ref-age | 0.772 ± 0.018 | 0.772 ± 0.018 | 0.772 ± 0.018 | 0.772 ± 0.018 | 0.772 ± 0.018 |
| ref-age+site | 0.814 ± 0.015 | 0.814 ± 0.015 | 0.814 ± 0.015 | 0.814 ± 0.015 | 0.814 ± 0.015 |
| ref | 0.667 ± 0.001 | 0.667 ± 0.001 | 0.667 ± 0.001 | 0.667 ± 0.001 | 0.667 ± 0.001 |
<!-- end generated -->

<!-- generated: design-within brats_* -->
| ID | brats_gbm_fused | brats_gbm_t1c | brats_gbm_t1n | brats_gbm_t2f | brats_gbm_t2w |
|----|---|---|---|---|---|
| F1 | - / - | 0.689 ± 0.014 / 0.790 ± 0.017 | 0.682 ± 0.027 / 0.829 ± 0.019 | 0.713 ± 0.020 / 0.809 ± 0.023 | 0.695 ± 0.020 / 0.825 ± 0.021 |
| F2 | - / - | 0.675 ± 0.030 / 0.833 ± 0.018 | 0.664 ± 0.023 / 0.833 ± 0.022 | 0.702 ± 0.019 / 0.829 ± 0.022 | 0.663 ± 0.052 / 0.800 ± 0.031 |
| F3 | - / - | 0.658 ± 0.015 / 0.804 ± 0.009 | 0.692 ± 0.023 / 0.810 ± 0.020 | 0.713 ± 0.020 / 0.799 ± 0.015 | 0.685 ± 0.020 / 0.823 ± 0.025 |
| T1 | - / - | - / - | - / - | - / - | - / - |
| T1-last | - / - | - / - | - / - | - / - | - / - |
| R1 | - | 0.751 ± 0.048 | 0.698 ± 0.015 | 0.714 ± 0.036 | 0.716 ± 0.031 |
| M2 | 0.724 ± 0.005 / 0.844 ± 0.016 | - / - | - / - | - / - | - / - |
| M3 | 0.696 ± 0.043 / 0.831 ± 0.014 | - / - | - / - | - / - | - / - |
| M4 | 0.725 ± 0.025 / 0.837 ± 0.015 | - / - | - / - | - / - | - / - |
| ref-age | 0.776 ± 0.023 | 0.776 ± 0.023 | 0.776 ± 0.023 | 0.776 ± 0.023 | 0.776 ± 0.023 |
| ref-age+site | 0.776 ± 0.023 | 0.776 ± 0.023 | 0.776 ± 0.023 | 0.776 ± 0.023 | 0.776 ± 0.023 |
| ref | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 |
<!-- end generated -->

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

Test AUROC, mean ± std over the 5 repeated 40/10/50 splits unless noted. The tables are
generated blocks (`uv run nnfb-predict table --update docs/progress.md`); the comments
under them are written by hand.

### hecktor_hpv_pet_gtvp

<!-- generated: runs hecktor_hpv_pet_gtvp -->
| Run | Design | AUROC | Within-centre AUROC |
|---|---|---|---|
| `group_prior` | ref | 0.839 ± 0.011 | 0.500 ± 0.000 |
| `radiomics_filtered_enet` |  | 0.817 ± 0.045 | 0.626 ± 0.118 |
| `radiomics_filtered_corr0.9_logreg` |  | 0.814 ± 0.017 | 0.620 ± 0.061 |
| `radiomics_filtered_corr0.9_enet` |  | 0.810 ± 0.044 | 0.628 ± 0.123 |
| `radiomics_filtered_logreg` |  | 0.786 ± 0.008 | 0.583 ± 0.076 |
| `vit_frozen_mean_logreg` | F1 | 0.765 ± 0.047 | 0.490 ± 0.123 |
| `radiomics_original_logreg` | R1 | 0.744 ± 0.053 | 0.660 ± 0.100 |
| `vit_frozen_roi_margin_mean_logreg` | F3 | 0.742 ± 0.015 | 0.502 ± 0.099 |
| `vit_frozen_roi_mean_logreg` | F2 | 0.710 ± 0.058 | 0.454 ± 0.036 |
| `radiomics_original_corr0.9_logreg` |  | 0.706 ± 0.047 | 0.628 ± 0.150 |
| `radiomics_original_enet` |  | 0.689 ± 0.109 | 0.604 ± 0.105 |
| `radiomics_original_corr0.9_enet` |  | 0.679 ± 0.102 | 0.562 ± 0.117 |
| `cnn_frozen_mean_logreg` | F1 | 0.514 ± 0.061 | 0.351 ± 0.087 |
| `cnn_frozen_roi_margin_mean_logreg` | F3 | 0.498 ± 0.050 | 0.428 ± 0.131 |
| `cnn_frozen_roi_mean_logreg` | F2 | 0.459 ± 0.053 | 0.431 ± 0.158 |
<!-- end generated -->

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
  reliably. This applies to every probe row; for R4 the chosen L1 share varies a lot too.
- Correlation filter at 0.9: about 305 of 1037 filtered features kept, about 33 of 107
  original ones.
- Nothing beats `group_prior` (0.84). Image features may partly encode the centre too; the
  leave-one-centre-out evaluation (E1) is needed to tell.
- **Within-centre AUROC (2026-11-18, D24): the frozen embeddings carry no HPV signal
  beyond the centre.** CNN and ViT probes drop to 0.35-0.50, i.e. chance: their 0.71-0.77
  was the centre shortcut. Radiomics keeps some (R1 0.66, R2-R5 0.58-0.63), and the
  original set now beats the filtered one, so the filtered set's gain in plain AUROC was
  centre signal too. With 66-81 pairs per split, the std is 0.04-0.16: read only the
  ordering, not the decimals.
- E1 (leave-one-centre-out) is not workable on these centres: MDA is 30/31 positive, CHUM
  7/8, CHUV 1/4, so most held-out centres have no AUROC. Within-centre AUROC replaces it
  for HECKTOR.

### stoic_severe

<!-- generated: runs stoic_severe -->
| Run | Design | AUROC |
|---|---|---|
| `vit_frozen_roi_mean_logreg` | F2 | 0.778 ± 0.008 |
| `vit_frozen_roi_margin_mean_logreg` | F3 | 0.776 ± 0.009 |
| `vit_frozen_mean_logreg` | F1 | 0.760 ± 0.013 |
| `radiomics_original_enet` |  | 0.756 ± 0.018 |
| `radiomics_filtered_corr0.9_logreg` |  | 0.755 ± 0.028 |
| `radiomics_filtered_enet` |  | 0.755 ± 0.026 |
| `radiomics_filtered_corr0.9_enet` |  | 0.753 ± 0.023 |
| `radiomics_original_logreg` | R1 | 0.747 ± 0.016 |
| `radiomics_original_corr0.9_enet` |  | 0.746 ± 0.014 |
| `radiomics_filtered_logreg` |  | 0.745 ± 0.032 |
| `radiomics_original_corr0.9_logreg` |  | 0.744 ± 0.021 |
| `cnn_frozen_roi_margin_mean_logreg` | F3 | 0.707 ± 0.022 |
| `cnn_frozen_roi_mean_logreg` | F2 | 0.697 ± 0.035 |
| `cnn_frozen_mean_logreg` | F1 | 0.656 ± 0.023 |
<!-- end generated -->

Frozen probes and radiomics, 2026-11-17: 800 train, 200 val, 1000 test cases per split
(about 150 positives in test), so much less noisy than HECKTOR.
- **ViT > radiomics > CNN.** Unlike HECKTOR, the frozen ViT beats the handcrafted
  baseline (0.78 vs 0.75).
- **ROI pooling helps here**, for both encoders: ROI mean and ROI + 10 mm > whole crop.
  The lungs fill most of the crop, so the whole-crop mean mixes in the chest wall and the
  mediastinum.
- **The filtered radiomics set brings nothing:** R2-R5 are within noise of R1
  (backlog item on dropping it).

### stoic_covid

<!-- generated: runs stoic_covid -->
| Run | Design | AUROC |
|---|---|---|
| `vit_frozen_roi_mean_logreg` | F2 | 0.814 ± 0.012 |
| `vit_frozen_roi_margin_mean_logreg` | F3 | 0.808 ± 0.017 |
| `vit_frozen_mean_logreg` | F1 | 0.787 ± 0.029 |
| `radiomics_filtered_enet` |  | 0.738 ± 0.017 |
| `radiomics_original_logreg` | R1 | 0.738 ± 0.013 |
| `radiomics_original_enet` |  | 0.737 ± 0.013 |
| `radiomics_filtered_corr0.9_enet` |  | 0.735 ± 0.019 |
| `radiomics_original_corr0.9_logreg` |  | 0.734 ± 0.012 |
| `radiomics_original_corr0.9_enet` |  | 0.728 ± 0.015 |
| `radiomics_filtered_logreg` |  | 0.726 ± 0.021 |
| `radiomics_filtered_corr0.9_logreg` |  | 0.725 ± 0.020 |
| `cnn_frozen_roi_mean_logreg` | F2 | 0.698 ± 0.010 |
| `cnn_frozen_roi_margin_mean_logreg` | F3 | 0.696 ± 0.007 |
| `cnn_frozen_mean_logreg` | F1 | 0.621 ± 0.014 |
<!-- end generated -->

Same scans, crops, features and ROIs as `stoic_severe` (linked, not recomputed); only
the label and the stratified splits differ. 1205 of 2000 positive.
- Same ordering as severe, with a wider gap: ViT ROI mean about 0.81 vs radiomics about
  0.73-0.74. ROI pooling again beats the whole crop for both encoders.
- Here the filtered radiomics set is slightly *worse* than `original` (R2, R3 vs R1).

### lld_mmri_malignant_*

Grid above (`design lld_mmri_*`). 498 lesions, 266 malignant; 199 train, 50 val, 249 test
per split. Single-phase probes, 2026-11-17.
- **ROI pooling is the big win:** ViT whole crop 0.64-0.70 → ROI mean (F2) 0.70-0.87. A
  lesion of a few cm in a 192 mm crop is a small part of the whole-crop mean.
- **Best embedding: ViT ROI mean on the delayed phase (0.867), then venous (0.837) and T2
  (0.808).** On delay and venous it is above or level with the best radiomics (R1 0.823,
  0.828). F3 (ROI + 10 mm) is below F2 on most phases: the liver around the lesion dilutes.
- **CNN at chance on every phase** (0.50-0.58), as on HECKTOR.
- **Filtered radiomics never beats original:** R2-R5 at or below R1 on all 8 phases (dropped, D25).
- Late fusion over phases (M2-M4) is the natural next step: the best phases differ
  between the ViT (delay, venous, T2) and radiomics (venous, delay, arterial).

### brats_gbm_*

Grids above (`design brats_*`, `design-within brats_*`). 632 patients, 384 GBM; 252 train,
64 val, 316 test per split. Single-sequence probes, 2026-11-18.
- **ViT well above the site-only reference (0.667):** 0.84-0.88 plain, 0.79-0.83 within-centre.
  Unlike HECKTOR, most of the signal is not the site.
- **Best: ViT whole-crop mean on t2w (0.877) and t1n (0.874).** Within-centre the ROI mean is
  level with or above the whole crop (t1c, t1n 0.833).
- **ROI pooling does not help in plain AUROC** (unlike LLD-MMRI). The crop holds most of the
  brain, so this may partly be patient age (atrophy, ventricles): GBM patients are older.
  Confirmed by the age references below.
- **ViT beats radiomics on every sequence** (R1 0.76-0.78 plain, 0.70-0.75 within). The CNN is
  below radiomics (0.66-0.74), as on the other tasks.

Late fusion over the 4 sequences (D30), every pooling and method, on the fused task
`brats_gbm_fused` (AUROC, then within-centre AUROC):

Age and fusion, 2026-11-18:
- **Age is a strong shortcut:** age alone 0.772 (`ref-age`), age + site 0.814, within-centre
  0.776. That is level with single-sequence radiomics (0.76-0.78). The ViT (0.84-0.88) is
  above it, but by much less than it is above the site-only `ref` (0.667).
- **The tumour ROI carries the signal beyond age.** One-off check (not a generated table):
  AUROC on pairs from the same site *and* age decade, where age alone falls to 0.62. ViT
  ROI-mean runs hold up (t1c F2 0.821, fused concat ROI mean 0.826); whole-crop runs drop more
  (t2w F1 0.877 → 0.762, fused concat whole crop 0.885 → 0.781). So the whole-crop advantage in
  plain AUROC is largely age; for this task, read ROI pooling as the better design.
- **Fusion gains are small:** best plain AUROC is the concat of whole-crop embeddings (0.885 vs
  0.877 single), best within-centre the concat of ROI means (M2 0.844 vs 0.833 single).
  M2 ≈ M4 > M3 (averaging the embeddings loses information). Fused radiomics 0.817.
- **CNN stays below age alone** in every variant (0.71-0.77).

<!-- generated: runs brats_gbm_fused -->
| Run | Design | AUROC | Within-centre AUROC |
|---|---|---|---|
| `vit_fused_concat_mean_logreg` |  | 0.885 ± 0.013 | 0.831 ± 0.018 |
| `vit_fused_mean_mean_logreg` |  | 0.878 ± 0.014 | 0.823 ± 0.018 |
| `vit_fused_stack_mean_logreg` |  | 0.877 ± 0.019 | 0.830 ± 0.020 |
| `vit_fused_concat_roi_mean_logreg` | M2 | 0.864 ± 0.013 | 0.844 ± 0.016 |
| `vit_fused_concat_roi_margin_mean_logreg` |  | 0.859 ± 0.019 | 0.830 ± 0.007 |
| `vit_fused_stack_roi_margin_mean_logreg` |  | 0.854 ± 0.022 | 0.824 ± 0.014 |
| `vit_fused_stack_roi_mean_logreg` | M4 | 0.854 ± 0.013 | 0.837 ± 0.015 |
| `vit_fused_mean_roi_mean_logreg` | M3 | 0.851 ± 0.011 | 0.831 ± 0.014 |
| `vit_fused_mean_roi_margin_mean_logreg` |  | 0.844 ± 0.025 | 0.813 ± 0.030 |
| `radiomics_original_fused_concat_logreg` |  | 0.817 ± 0.023 | 0.774 ± 0.034 |
| `covariates_age_group_logreg` | ref-age+site | 0.814 ± 0.015 | 0.776 ± 0.023 |
| `radiomics_original_fused_stack_logreg` |  | 0.790 ± 0.021 | 0.754 ± 0.031 |
| `covariates_age_logreg` | ref-age | 0.772 ± 0.018 | 0.776 ± 0.023 |
| `cnn_fused_stack_mean_logreg` |  | 0.771 ± 0.011 | 0.725 ± 0.018 |
| `cnn_fused_concat_mean_logreg` |  | 0.760 ± 0.032 | 0.697 ± 0.034 |
| `cnn_fused_concat_roi_margin_mean_logreg` |  | 0.750 ± 0.023 | 0.723 ± 0.030 |
| `cnn_fused_stack_roi_margin_mean_logreg` |  | 0.750 ± 0.037 | 0.732 ± 0.030 |
| `cnn_fused_concat_roi_mean_logreg` | M2 | 0.749 ± 0.007 | 0.724 ± 0.005 |
| `cnn_fused_stack_roi_mean_logreg` | M4 | 0.747 ± 0.027 | 0.725 ± 0.025 |
| `cnn_fused_mean_mean_logreg` |  | 0.727 ± 0.017 | 0.691 ± 0.021 |
| `cnn_fused_mean_roi_margin_mean_logreg` |  | 0.718 ± 0.036 | 0.708 ± 0.027 |
| `cnn_fused_mean_roi_mean_logreg` | M3 | 0.710 ± 0.044 | 0.696 ± 0.043 |
| `group_prior` | ref | 0.667 ± 0.001 | 0.500 ± 0.000 |
<!-- end generated -->

### upenn_os_*

452 patients, 226 survived 1 year; 180 train, 46 val, 226 test per split. AUROC
(1-year OS), then the C-index of the Cox probes (D32), then the fused runs:

Results, 2026-11-19:
- **Age alone beats every imaging probe:** 1-year AUROC 0.667, C-index 0.619. The best
  single-sequence ViT is T2 F3 (ROI + 10 mm) at 0.655 AUROC and 0.623 C-index; most ViT runs
  sit at 0.58-0.64 AUROC and 0.58-0.61 C-index. Survival from pre-operative imaging alone is
  weak, in line with the literature (~0.6-0.65).
- **The CNN is near chance** (AUROC 0.50-0.55, C-index 0.52-0.55). The ViT is consistently
  above it. Radiomics sits between them, best on T2 (0.623 AUROC, 0.567 C-index).
- **T2 is the most informative sequence** for the ViT and radiomics. ROI pooling helps a
  little on T2 and T1 (F3 > F1), not on T1GD.
- **Fusion barely helps:** best is the concat of the four ROI + 10 mm means (0.663), M2
  (concat ROI means) 0.661 ± 0.003, both level with age; mean fusion (M3) loses, as on
  BraTS. Fused radiomics 0.600 (concat) / 0.617 (stack).
- Open question: does imaging add anything over age? Needs age + imaging probes.

<!-- generated: design upenn_* -->
| ID | upenn_os_flair | upenn_os_fused | upenn_os_t1 | upenn_os_t1gd | upenn_os_t2 |
|----|---|---|---|---|---|
| F1 | 0.502 ± 0.038 / 0.619 ± 0.039 | - / - | 0.531 ± 0.026 / 0.630 ± 0.044 | 0.513 ± 0.034 / 0.603 ± 0.032 | 0.547 ± 0.037 / 0.607 ± 0.036 |
| F2 | 0.501 ± 0.039 / 0.622 ± 0.013 | - / - | 0.532 ± 0.019 / 0.627 ± 0.026 | 0.514 ± 0.032 / 0.581 ± 0.033 | 0.542 ± 0.015 / 0.638 ± 0.037 |
| F3 | 0.509 ± 0.012 / 0.610 ± 0.023 | - / - | 0.516 ± 0.021 / 0.643 ± 0.022 | 0.515 ± 0.034 / 0.587 ± 0.023 | 0.542 ± 0.010 / 0.655 ± 0.027 |
| T1 | - / - | - / - | - / - | - / - | - / - |
| T1-last | - / - | - / - | - / - | - / - | - / - |
| R1 | 0.569 ± 0.015 | - | 0.536 ± 0.052 | 0.538 ± 0.026 | 0.623 ± 0.030 |
| M2 | - / - | 0.516 ± 0.017 / 0.661 ± 0.003 | - / - | - / - | - / - |
| M3 | - / - | 0.505 ± 0.040 / 0.599 ± 0.027 | - / - | - / - | - / - |
| M4 | - / - | 0.528 ± 0.011 / 0.629 ± 0.052 | - / - | - / - | - / - |
| ref-age | 0.667 ± 0.031 | - | 0.667 ± 0.031 | 0.667 ± 0.031 | 0.667 ± 0.031 |
| ref-age+site | - | - | - | - | - |
| ref | n/a | n/a | n/a | n/a | n/a |
<!-- end generated -->

<!-- generated: cindex upenn_* -->
| ID | upenn_os_flair | upenn_os_t1 | upenn_os_t1gd | upenn_os_t2 |
|----|---|---|---|---|
| F1 | 0.523 ± 0.009 / 0.588 ± 0.024 | 0.520 ± 0.025 / 0.589 ± 0.024 | 0.524 ± 0.012 / 0.580 ± 0.017 | 0.554 ± 0.020 / 0.595 ± 0.030 |
| F2 | 0.521 ± 0.015 / 0.588 ± 0.010 | 0.549 ± 0.025 / 0.598 ± 0.018 | 0.526 ± 0.005 / 0.587 ± 0.014 | 0.542 ± 0.008 / 0.605 ± 0.029 |
| F3 | 0.524 ± 0.010 / 0.599 ± 0.008 | 0.550 ± 0.019 / 0.595 ± 0.022 | 0.519 ± 0.007 / 0.580 ± 0.018 | 0.536 ± 0.009 / 0.623 ± 0.011 |
| R1 | 0.544 ± 0.021 | 0.519 ± 0.027 | 0.527 ± 0.021 | 0.567 ± 0.013 |
| ref-age | 0.619 ± 0.019 | 0.619 ± 0.019 | 0.619 ± 0.019 | 0.619 ± 0.019 |
<!-- end generated -->

<!-- generated: runs upenn_os_fused -->
| Run | Design | AUROC |
|---|---|---|
| `vit_fused_concat_roi_margin_mean_logreg` |  | 0.663 ± 0.024 |
| `vit_fused_concat_roi_mean_logreg` | M2 | 0.661 ± 0.003 |
| `vit_fused_stack_roi_margin_mean_logreg` |  | 0.654 ± 0.042 |
| `vit_fused_concat_mean_logreg` |  | 0.639 ± 0.043 |
| `vit_fused_mean_mean_logreg` |  | 0.634 ± 0.034 |
| `vit_fused_stack_roi_mean_logreg` | M4 | 0.629 ± 0.052 |
| `vit_fused_stack_mean_logreg` |  | 0.628 ± 0.038 |
| `vit_fused_mean_roi_margin_mean_logreg` |  | 0.620 ± 0.016 |
| `radiomics_original_fused_stack_logreg` |  | 0.617 ± 0.039 |
| `radiomics_original_fused_concat_logreg` |  | 0.600 ± 0.031 |
| `vit_fused_mean_roi_mean_logreg` | M3 | 0.599 ± 0.027 |
| `cnn_fused_concat_mean_logreg` |  | 0.536 ± 0.019 |
| `cnn_fused_mean_mean_logreg` |  | 0.535 ± 0.019 |
| `cnn_fused_stack_roi_margin_mean_logreg` |  | 0.530 ± 0.014 |
| `cnn_fused_stack_roi_mean_logreg` | M4 | 0.528 ± 0.011 |
| `cnn_fused_stack_mean_logreg` |  | 0.520 ± 0.034 |
| `cnn_fused_concat_roi_mean_logreg` | M2 | 0.516 ± 0.017 |
| `cnn_fused_mean_roi_margin_mean_logreg` |  | 0.507 ± 0.035 |
| `cnn_fused_concat_roi_margin_mean_logreg` |  | 0.506 ± 0.026 |
| `cnn_fused_mean_roi_mean_logreg` | M3 | 0.505 ± 0.040 |
<!-- end generated -->

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
- Added a Backlog section to this file. Nested CV for probe selection is parked until the
  probes have run on LLD-MMRI and STOIC; other deferred items from today are listed there too.
- Added `scripts/run_probes.sh <task>`: frozen features + radiomics + all probes + design
  grid for one prepared task. Checked on HECKTOR: same numbers as the runs by hand.

### 2026-11-17
- `scripts/run_probes.sh` finished on `stoic_severe` and `stoic_covid` (see Results).
  On STOIC severe the frozen ViT beats radiomics and ROI pooling helps, unlike HECKTOR.
  `stoic_covid` reuses the crops, features and radiomics of `stoic_severe` through
  symlinks in `data/processed/stoic_covid/`; only its labels and splits are its own.
- Layout (D23): results and logs moved from `reports/` to `outputs/` (not versioned);
  `reports/` keeps only what people read. Result tables in this file are now generated
  blocks, rebuilt with `uv run nnfb-predict table --update docs/progress.md`.

### 2026-11-18
- LLD-MMRI: all 8 phases prepared and probed on the host (see Results). ViT ROI pooling
  reaches 0.867 on the delayed phase; the CNN is at chance.
- RSNA prepare stopped at 1792 of 2019 studies when the host's tmux session was killed
  (cause under investigation on the host). Finished cases are cached, so a rerun resumes.
- Added within-centre AUROC (D24): `nnfb-predict table --within`, a `design-within`
  generated grid, and a second AUROC column in the runs tables of tasks with centres. It
  is computed from the existing `predictions.csv`, so nothing was rerun. On HECKTOR it
  shows the frozen embeddings at chance once the centre is removed. LEACE parked in the
  backlog; E1 judged unworkable on HECKTOR.
- Dropped the filtered radiomics set and the R2-R5 rows (D25): `run_probes.sh` now runs
  `original` with the plain L2 probe only (R1). Removed from the backlog with it: the
  filter ablation item. Cached filtered features and past runs are kept.
- BraTS 2024 post-treatment glioma added (D28), on branch `brats-glioma-type`: four tasks
  `brats_gbm_<t1c|t1n|t2f|t2w>`, GBM vs astrocytoma/oligodendroglioma, earliest scan per
  patient, whole tumour (1+2+3) as ROI, site as `group`. 632 patients (384 GBM). Smoke test
  on 6 cases x 2 sequences: ROI fully inside every crop, ~2 s per case. `openpyxl` added to
  read the metadata xlsx.
- BraTS: all 4 sequences prepared on the host (632 cases each, no failures, no case
  without ROI, ~2.9 GB per task). Probes started.
- BraTS: frozen probes and radiomics on all 4 sequences. ViT 0.84-0.88 (within-centre
  0.79-0.83) vs site only 0.667 and radiomics 0.76-0.78; the whole crop is as good as the ROI.
- This server's `outputs/results/` holds only the BraTS runs (the other tasks ran in another
  workspace), so only the `brats_*` blocks were regenerated here. A full
  `table --update` needs all results in one `outputs/`.
- BraTS: added the age references (D29, `nnfb-predict covariate-baseline`; `brats.py` now
  writes `age`) and late fusion M2-M4 (D30, `nnfb-features fuse`, results under a fused task
  name such as `brats_gbm_fused`). Unit-tested on synthetic data; real runs pending on the host.
- BraTS on the host: labels.csv rewritten with `age` (splits unchanged, same hash), age
  references and fusion (CNN, ViT, radiomics) run. Age alone 0.772, age + site 0.814; best
  ViT fused 0.885 (concat, whole crop), 0.844 within-centre (concat, ROI mean). Within the
  same site and age decade the ROI-pooled ViT keeps ~0.82, the whole-crop ViT drops to ~0.78.
- UPENN-GBM survival added (D31): four tasks `upenn_os_<seq>`, 452 deceased patients (no
  censoring), label = 1-year OS (226/226), `time`/`event`/`age` in labels.csv, ROI = expert
  mask (145) else automatic (307). Cox probes with C-index (D32, `nnfb-cox`,
  scikit-survival); `run_probes.sh` runs them for tasks with `time`/`event`. Prepare
  smoke-tested on 4 cases (2 expert, 2 automatic masks): ROI fully inside, ~3 s per case.

### 2026-11-19
- UPENN on the host: all 4 sequences prepared (452 cases, no failures, identical splits),
  probes, radiomics, age reference, Cox probes and fusion run. Age alone 0.667 AUROC / 0.619
  C-index; best ViT T2 F3 0.655 / 0.623; CNN near chance; fused concat (M2) 0.661.
- `nnfb-cox` CLI fixed: a single-command typer app made `probe` an extra argument, so the
  first `run_probes.sh` pass skipped every Cox probe. A CLI test now covers it.

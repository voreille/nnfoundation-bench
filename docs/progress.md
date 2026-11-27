# Progress

Where the benchmark stands, what's next, and results as they come in. Design choices
and their reasons go in [decisions.md](decisions.md). Notes on the paper's protocol go in
[../references/nnfoundation_protocol.md](../references/nnfoundation_protocol.md).

## Goal

Use the pretrained nnFoundation models (CNN and ViT, arXiv:2609.26924) in our own
ROI-based pipeline. This benchmark is where we make design choices around them, such
as input channels, ROI handling, and how to pool features within a ROI.

**Target: HECKTOR HPV classification at AUROC ≥ 0.99.**

## Current status (2026-11-26)

| Task | Data | Pipeline | Results |
|------|------|----------|---------|
| `kits_liver` | 2/489 raw images downloaded | prepare, train and predict smoke-tested | none yet |
| `hecktor_hpv_pet_gtvp` | 102 patients on the NAS, labels found | prepared: 102 cases, 52 positive, no failures | frozen probes, radiomics and centre-only reference done |
| `mrnet_*` (9 tasks) | 1250 public exams on the NAS; all file sizes and 40 md5s checked | npy-to-NIfTI conversion and crop smoke-tested on 2 exams per plane | none yet |
| `lld_mmri_malignant_*` (8 phases) | 498 patients x 8 series on the NAS, all files present | all 8 prepared: 498 cases, 266 malignant, no failures, no missing ROI | frozen probes and radiomics done for all 8 |
| `stoic_severe`, `stoic_covid` | 2000 scans | both prepared: 2000 cases, no failures; `stoic_covid` links the crops, features and radiomics of `stoic_severe` | frozen probes and radiomics done for both |
| `hecktor2022_rfs_ct`, `hecktor2022_rfs_pet` | HECKTOR 2022 train + test on the NAS, all checked: 827 patients with RFS, 174 relapses, 9 centres | preprocessing smoke-tested on 5 cases (PET and CT crops on the same grid, labels 1/2 kept); survival probe to write | none yet |
| `rsna_spine_fracture` | 2019 studies unpacked on the NAS, checked (961 positive) | prepare on the host stopped at 1792 of 2019 (tmux session killed 2026-11-18 08:13); rerun resumes | none yet |
| `upenn_os_*` (4 sequences) | 452 patients on the NAS (`UPENN_GBM`) + OS labels placed by hand (`GBM-UPENN-OS`), all 4 sequences, a mask for each (145 expert) | all 4 prepared on the host: 452 cases, 226 survived 1 year, no failures, no missing ROI; identical labels and splits (D31) | probes, radiomics, age reference, Cox / C-index (D32), fusion (`upenn_os_fused`) done |
| `hecktor22_rfs_pet`, `hecktor22_rfs_ct` | HECKTOR 2022 challenge on the NAS (training + test with ground truth, `data/raw/HECKTOR2022/`), 827 patients with RFS, all CT, PET and masks present | labels checked on real data (714 labelled, 692 with GTVp); prepare both prepared on the host: 692 cases, 134 positive, 22 without GTVp excluded, no failures; identical labels and splits (D34) | probes, radiomics, age / age+HPV / centre references, Cox / C-index done; official challenge split (`_official`, D37) done |
| `totalseg_ct_organs`, `totalseg_mri_organs` (+ `_official`) | TotalSegmentator CT v2.0.1 / MRI v2.0.0 (Zenodo), to place by hand under `data/raw/totalsegmentator_<ct\|mri>/` | scan layout, extract-time organ crops and multi-class probe unit-tested on synthetic scans (D38); smoke test on real scans pending | none yet |
| `brats_gbm_*` (4 sequences) | 1809 scans on the NAS, 1621 with seg; metadata xlsx read | all 4 prepared on the host: 632 cases, 384 GBM, no failures, no missing ROI; identical labels and splits (D28) | probes, radiomics, site and age references, fusion (`brats_gbm_fused`) done |

Next steps:
1. Frozen probes and radiomics are done on HECKTOR, both STOIC tasks and the 8 LLD-MMRI
   phases.
2. BraTS: age references and fusion done. Next: a generated age-and-site stratified AUROC
   (see Results) and age-adjusted probes (D29 open).
3. UPENN survival: probes, Cox, fusion and image + age blends (D36) done; the images add
   little over age (C-index 0.629 vs 0.619).
4. HECKTOR 2022, our 2-year variant `hecktor22_rfs_*` (D34): probes, Cox, within-centre
   C-index (D35), image + covariate blends (D36) and PET + CT fusion done.
   Official challenge split (D37) done: ViT 0.678-0.690 single, 0.683-0.691 PET + CT fused;
   radiomics level with it (0.662-0.667). Next: bootstrap CI on the single test set.
5. HECKTOR 2022 RFS (D26): prepare both modalities on the host; meanwhile write the Cox
   probe, the C-index (plain, within-centre) and the per-ROI poolings (GTVp, GTVn,
   concatenation) are written (D27); then `scripts/run_probes.sh` on both tasks. Next:
   PET + CT late fusion (M2-M4) and the official train -> test view.
6. RSNA: resume `prepare rsna_spine_fracture`, then `RADIOMICS="original"
   scripts/run_probes.sh rsna_spine_fracture`.
7. Fine-tuning baseline with glovita: `FOLDS=0 scripts/run_benchmark.sh hecktor_hpv_pet_gtvp`
   (design T1).
8. Input-size designs S2-S7: add `--fov` / `--mode` to `nnfb-features extract`, then run
   the FOV sweep.
9. LLD-MMRI late fusion over the phases (M2-M4, D14, D15, D30): `nnfb-features fuse` is ready.
10. KiTS Liver full run (background, paper reproduction).
11. TotalSegmentator organ recognition (D38): download and unzip both Zenodo records, smoke-test
   `prepare` + `extract` on 2 scans, then `RADIOMICS="" scripts/run_probes.sh totalseg_ct_organs`
   (and MRI), then `prepare totalseg_<ct|mri>_organs_official` and its probes.
12. MRNet: prepare and run `mrnet_abnormal_sagittal` first (closest to the paper's
   "pathology"). Each task needs about 12 GB in `data/processed`. See D12.

## Backlog

- **Cap the samples per class for the organ probes** (D38): CT has ~45k train samples x 98
  classes, ribs and vertebrae dominate. *Trigger:* `nnfb-features probe totalseg_ct_organs`
  too slow on the host.
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
| HECKTOR 2022 RFS (`hecktor2022_rfs_ct`, `hecktor2022_rfs_pet`) | CT, PET (same patient, same crop grid) | GTVp ∪ GTVn (dataset masks, labels kept) | CT or PET, 192³ at 1 mm centred on GTVp ∪ GTVn | survival: C-index, not AUROC (D27); 174 relapses in 827; challenge Task 2 winners about 0.68 on the test set (official split, not comparable to our splits) |
| STOIC 2021 (`stoic_severe`, `stoic_covid`) | CT | lungs (TotalSegmentator, 3 mm masks, D18); wider than 192 mm | CT, 192³ at 1 mm centred on the lungs | 301/2000 severe, 1205/2000 COVID; STOIC challenge (severe, hidden test set): about 0.8 |
| UPENN-GBM survival (`upenn_os_*`) | T1, T1GD, T2, FLAIR (co-registered, one grid) | whole tumour (expert mask, else automatic) | each sequence, 192³ at 1 mm centred on the tumour | 1-year OS 226/452 (AUROC); C-index grid for the Cox probes; age reference |
| BraTS 2024 glioma (`brats_gbm_*`) | t1c, t1n, t2f, t2w (co-registered, one grid) | whole tumour (dataset seg, labels 1+2+3) | each sequence, 192³ at 1 mm centred on the tumour | 384/632 GBM; 5 sites, GBM share 38-100% by site: `ref` row (site only) |

Unless a row says otherwise:
- **Input:** the dataset's default input above, z-scored over the whole scan.
- **Frozen rows:** the probe is a logistic regression (D13).
- **Several modalities:** embeddings are extracted per modality, each from its own
  single-channel task, then fused late (M2-M5, D14).

`<p>` is the probe: `logreg`, or `cox` (ridge Cox, D27) on survival tasks, whose cells are
C-index instead of AUROC.

Status: **ready** = code exists (the grid below shows which tasks have run); **planned**
= code still to write; **dropped** = not pursued (dropped rows leave the grids; their past
runs stay in the per-task runs tables).

| ID | Axis | Design | Run name | Applies to | Status |
|----|------|--------|----------|------------|--------|
| F1 | Pooling (frozen) | Mean over the whole crop (what glovita's head sees) | `<m>_frozen_mean_logreg` | all | ready |
| F2 | Pooling (frozen) | Mean over the ROI, cells weighted by ROI fraction | `<m>_frozen_roi_mean_logreg` | MRNet n/a (ROI = FOV, same as F1 up to padding) | ready |
| F3 | Pooling (frozen) | Mean over the ROI + 10 mm | `<m>_frozen_roi_margin_mean_logreg` | MRNet n/a | ready |
| F4 | Pooling (frozen) | ROI mean over earlier CNN stages (16 mm, 8 mm cells); ViT n/a |  | all | planned |
| F5 | Pooling (frozen, several ROIs) | Mean over the first ROI only (HECKTOR 2022: GTVp) | `<m>_frozen_gtvp_mean_<p>` | tasks with named ROIs (HECKTOR 2022) | ready |
| F6 | Pooling (frozen, several ROIs) | Mean over the second ROI only (GTVn) | `<m>_frozen_gtvn_mean_<p>` | tasks with named ROIs | ready |
| F7 | Pooling (frozen, several ROIs) | Concatenation of the per-ROI means (GTVp, GTVn), one probe; a missing ROI is imputed (train median). F2 is the union (one mean over GTVp ∪ GTVn) | `<m>_frozen_roi_concat_mean_<p>` | tasks with named ROIs | ready |
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

<!-- generated: design hecktor_hpv_* stoic_* -->
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

HECKTOR 2022 recurrence-free survival, **C-index** (D26, D27):

<!-- generated: design hecktor2022_rfs_ct hecktor2022_rfs_pet -->
No results yet.
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

### hecktor2022_rfs_ct, hecktor2022_rfs_pet (C-index)

<!-- generated: runs hecktor2022_rfs_pet -->
No results yet.
<!-- end generated -->

<!-- generated: runs hecktor2022_rfs_ct -->
No results yet.
<!-- end generated -->

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
- **Image + age (D36):** small AUROC gains over age alone (0.667): best ViT ROI mean + age
  0.688 (T2) and 0.689 (FLAIR), radiomics + age up to 0.689 (T2); several blends land at or
  below age alone. **No C-index gain:** age alone 0.619, best blend ViT T2 F3 + age 0.629
  (the image alone 0.623); most blends 0.58-0.62. The val weight of the images varies
  between folds (T2 ROI mean: 0-0.8; 46 val cases). On UPENN the frozen features add
  little or nothing to age for survival.

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

Image + age (D36), one table per sequence:

<!-- generated: adjusted upenn_os_t1 -->
| Run | AUROC | C-index |
|---|---|---|
| `covariates_age` | 0.667 ± 0.031 | 0.619 ± 0.019 |
| `cnn_frozen_mean` | 0.531 ± 0.026 | 0.520 ± 0.025 |
| `cnn_frozen_mean_plus_age` | 0.653 ± 0.036 | 0.589 ± 0.038 |
| `cnn_frozen_roi_margin_mean` | 0.516 ± 0.021 | 0.550 ± 0.019 |
| `cnn_frozen_roi_margin_mean_plus_age` | 0.639 ± 0.025 | 0.595 ± 0.033 |
| `cnn_frozen_roi_mean` | 0.532 ± 0.019 | 0.549 ± 0.025 |
| `cnn_frozen_roi_mean_plus_age` | 0.654 ± 0.038 | 0.596 ± 0.041 |
| `radiomics_original` | 0.536 ± 0.052 | 0.519 ± 0.027 |
| `radiomics_original_plus_age` | 0.643 ± 0.053 | 0.581 ± 0.045 |
| `vit_frozen_mean` | 0.630 ± 0.044 | 0.589 ± 0.024 |
| `vit_frozen_mean_plus_age` | 0.673 ± 0.032 | 0.603 ± 0.013 |
| `vit_frozen_roi_margin_mean` | 0.643 ± 0.022 | 0.595 ± 0.022 |
| `vit_frozen_roi_margin_mean_plus_age` | 0.678 ± 0.028 | 0.616 ± 0.018 |
| `vit_frozen_roi_mean` | 0.627 ± 0.026 | 0.598 ± 0.018 |
| `vit_frozen_roi_mean_plus_age` | 0.666 ± 0.016 | 0.607 ± 0.023 |
<!-- end generated -->

<!-- generated: adjusted upenn_os_t1gd -->
| Run | AUROC | C-index |
|---|---|---|
| `covariates_age` | 0.667 ± 0.031 | 0.619 ± 0.019 |
| `cnn_frozen_mean` | 0.513 ± 0.034 | 0.524 ± 0.012 |
| `cnn_frozen_mean_plus_age` | 0.634 ± 0.030 | 0.604 ± 0.020 |
| `cnn_frozen_roi_margin_mean` | 0.515 ± 0.034 | 0.519 ± 0.007 |
| `cnn_frozen_roi_margin_mean_plus_age` | 0.638 ± 0.055 | 0.568 ± 0.045 |
| `cnn_frozen_roi_mean` | 0.514 ± 0.032 | 0.526 ± 0.005 |
| `cnn_frozen_roi_mean_plus_age` | 0.625 ± 0.066 | 0.583 ± 0.037 |
| `radiomics_original` | 0.538 ± 0.026 | 0.527 ± 0.021 |
| `radiomics_original_plus_age` | 0.659 ± 0.040 | 0.586 ± 0.051 |
| `vit_frozen_mean` | 0.603 ± 0.032 | 0.580 ± 0.017 |
| `vit_frozen_mean_plus_age` | 0.654 ± 0.023 | 0.593 ± 0.019 |
| `vit_frozen_roi_margin_mean` | 0.587 ± 0.023 | 0.580 ± 0.018 |
| `vit_frozen_roi_margin_mean_plus_age` | 0.631 ± 0.043 | 0.603 ± 0.029 |
| `vit_frozen_roi_mean` | 0.581 ± 0.033 | 0.587 ± 0.014 |
| `vit_frozen_roi_mean_plus_age` | 0.657 ± 0.026 | 0.611 ± 0.024 |
<!-- end generated -->

<!-- generated: adjusted upenn_os_t2 -->
| Run | AUROC | C-index |
|---|---|---|
| `covariates_age` | 0.667 ± 0.031 | 0.619 ± 0.019 |
| `cnn_frozen_mean` | 0.547 ± 0.037 | 0.554 ± 0.020 |
| `cnn_frozen_mean_plus_age` | 0.656 ± 0.029 | 0.610 ± 0.018 |
| `cnn_frozen_roi_margin_mean` | 0.542 ± 0.010 | 0.536 ± 0.009 |
| `cnn_frozen_roi_margin_mean_plus_age` | 0.659 ± 0.040 | 0.569 ± 0.036 |
| `cnn_frozen_roi_mean` | 0.542 ± 0.015 | 0.542 ± 0.008 |
| `cnn_frozen_roi_mean_plus_age` | 0.644 ± 0.034 | 0.584 ± 0.025 |
| `radiomics_original` | 0.623 ± 0.030 | 0.567 ± 0.013 |
| `radiomics_original_plus_age` | 0.689 ± 0.040 | 0.615 ± 0.034 |
| `vit_frozen_mean` | 0.607 ± 0.036 | 0.595 ± 0.030 |
| `vit_frozen_mean_plus_age` | 0.660 ± 0.025 | 0.610 ± 0.033 |
| `vit_frozen_roi_margin_mean` | 0.655 ± 0.027 | 0.623 ± 0.011 |
| `vit_frozen_roi_margin_mean_plus_age` | 0.681 ± 0.026 | 0.629 ± 0.007 |
| `vit_frozen_roi_mean` | 0.638 ± 0.037 | 0.605 ± 0.029 |
| `vit_frozen_roi_mean_plus_age` | 0.688 ± 0.007 | 0.624 ± 0.011 |
<!-- end generated -->

<!-- generated: adjusted upenn_os_flair -->
| Run | AUROC | C-index |
|---|---|---|
| `covariates_age` | 0.667 ± 0.031 | 0.619 ± 0.019 |
| `cnn_frozen_mean` | 0.502 ± 0.038 | 0.523 ± 0.009 |
| `cnn_frozen_mean_plus_age` | 0.625 ± 0.020 | 0.588 ± 0.030 |
| `cnn_frozen_roi_margin_mean` | 0.509 ± 0.012 | 0.524 ± 0.010 |
| `cnn_frozen_roi_margin_mean_plus_age` | 0.613 ± 0.027 | 0.584 ± 0.047 |
| `cnn_frozen_roi_mean` | 0.501 ± 0.039 | 0.521 ± 0.015 |
| `cnn_frozen_roi_mean_plus_age` | 0.629 ± 0.062 | 0.599 ± 0.037 |
| `radiomics_original` | 0.569 ± 0.015 | 0.544 ± 0.021 |
| `radiomics_original_plus_age` | 0.645 ± 0.047 | 0.607 ± 0.031 |
| `vit_frozen_mean` | 0.619 ± 0.039 | 0.588 ± 0.024 |
| `vit_frozen_mean_plus_age` | 0.655 ± 0.049 | 0.603 ± 0.022 |
| `vit_frozen_roi_margin_mean` | 0.610 ± 0.023 | 0.599 ± 0.008 |
| `vit_frozen_roi_margin_mean_plus_age` | 0.670 ± 0.027 | 0.619 ± 0.013 |
| `vit_frozen_roi_mean` | 0.622 ± 0.013 | 0.588 ± 0.010 |
| `vit_frozen_roi_mean_plus_age` | 0.689 ± 0.028 | 0.603 ± 0.018 |
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

### hecktor22_rfs_*

692 patients (GTVp present), 134 recurrences within 2 years, 169 events overall;
9 centres. AUROC (2-year recurrence), within-centre AUROC, then the C-index (D32, D34):

Results, 2026-11-19:
- **Centre is the strongest single predictor:** centre alone 0.649 AUROC (`ref`) and 0.613
  C-index (`ref-site`), level with the best ViT probes (0.651-0.658 AUROC). Recurrence rates
  and follow-up differ by centre (USZ 37/82 positive, MDA 53/365).
- **The ViT keeps some signal within centres:** within-centre AUROC 0.59-0.62 (best PET F1
  0.616, CT F2 0.590), where centre is 0.5 by construction. C-index up to 0.637 (PET F3)
  and 0.634 (CT F2), a little above centre alone; within-centre C-index (D35) up to 0.603
  (PET F1) and 0.594 (CT F2), centre alone 0.5.
- **PET and CT are close.** PET is better on the whole crop (F1), CT catches up with ROI
  pooling (F2 0.651 vs F1 0.614). Radiomics 0.60-0.61 AUROC, ~0.60 C-index, weaker within
  centres (0.55-0.56).
- **The CNN is near chance** (0.51-0.57), as on UPENN.
- **Age carries nothing** (0.49 AUROC, 0.51 C-index). Age + HPV only 0.518 / 0.536: HPV is
  unknown for a third of patients (median-imputed) and nearly all-positive at MDA, so it
  adds little in this cohort.
- **Image + centre (D36):** blending the image probe with the centre lifts AUROC above
  centre alone (0.644 logistic, 0.649 `group_prior`): ViT PET F3 + centre 0.683, CT F2 +
  centre 0.674, radiomics + centre 0.668-0.678. Same for the C-index (centre alone 0.613):
  ViT PET F3 + centre 0.657, CT F2 + centre 0.650, radiomics + centre 0.631-0.643, each
  also above the image alone (0.637, 0.634). The CNN adds little (AUROC 0.63-0.65, C-index
  0.58-0.63). Adding age and HPV to the centre changes little (AUROC 0.650, C-index 0.614).
- **PET + CT fusion (D30):** score stacking of the two ViT ROI-mean probes (M4) 0.669, concat
  (M2) 0.657, mean (M3) 0.652; within-centre 0.58-0.60, no better than the best single
  modality. Fusion helps less than adding the centre.

<!-- generated: design hecktor22_rfs_ct hecktor22_rfs_fused hecktor22_rfs_pet -->
| ID | hecktor22_rfs_ct | hecktor22_rfs_fused | hecktor22_rfs_pet |
|----|---|---|---|
| F1 | 0.533 ± 0.039 / 0.614 ± 0.012 | - / - | 0.548 ± 0.029 / 0.656 ± 0.040 |
| F2 | 0.508 ± 0.025 / 0.651 ± 0.015 | - / - | 0.568 ± 0.053 / 0.647 ± 0.016 |
| F3 | 0.523 ± 0.022 / 0.636 ± 0.022 | - / - | 0.550 ± 0.039 / 0.658 ± 0.040 |
| T1 | - / - | - / - | - / - |
| T1-last | - / - | - / - | - / - |
| R1 | 0.602 ± 0.061 | - | 0.611 ± 0.025 |
| M2 | - / - | 0.557 ± 0.039 / 0.657 ± 0.039 | - / - |
| M3 | - / - | 0.544 ± 0.029 / 0.652 ± 0.038 | - / - |
| M4 | - / - | 0.533 ± 0.040 / 0.669 ± 0.017 | - / - |
| ref-age | 0.490 ± 0.014 | - | 0.490 ± 0.014 |
| ref-age+site | 0.640 ± 0.009 | - | 0.640 ± 0.009 |
| ref-age+hpv | 0.518 ± 0.006 | - | 0.518 ± 0.006 |
| ref | 0.649 ± 0.003 | n/a | 0.649 ± 0.003 |
<!-- end generated -->

<!-- generated: design-within hecktor22_rfs_ct hecktor22_rfs_fused hecktor22_rfs_pet -->
| ID | hecktor22_rfs_ct | hecktor22_rfs_fused | hecktor22_rfs_pet |
|----|---|---|---|
| F1 | 0.508 ± 0.054 / 0.526 ± 0.044 | - / - | 0.529 ± 0.053 / 0.616 ± 0.040 |
| F2 | 0.493 ± 0.051 / 0.590 ± 0.028 | - / - | 0.531 ± 0.051 / 0.587 ± 0.029 |
| F3 | 0.504 ± 0.052 / 0.580 ± 0.037 | - / - | 0.513 ± 0.033 / 0.595 ± 0.038 |
| T1 | - / - | - / - | - / - |
| T1-last | - / - | - / - | - / - |
| R1 | 0.558 ± 0.041 | - | 0.553 ± 0.024 |
| M2 | - / - | 0.523 ± 0.022 / 0.583 ± 0.022 | - / - |
| M3 | - / - | 0.505 ± 0.015 / 0.591 ± 0.029 | - / - |
| M4 | - / - | 0.509 ± 0.049 / 0.595 ± 0.011 | - / - |
| ref-age | 0.488 ± 0.035 | - | 0.488 ± 0.035 |
| ref-age+site | 0.488 ± 0.035 | - | 0.488 ± 0.035 |
| ref-age+hpv | 0.499 ± 0.031 | - | 0.499 ± 0.031 |
| ref | 0.500 ± 0.000 | n/a | 0.500 ± 0.000 |
<!-- end generated -->

<!-- generated: cindex-within hecktor22_rfs_ct hecktor22_rfs_pet -->
| ID | hecktor22_rfs_ct | hecktor22_rfs_pet |
|----|---|---|
| F1 | 0.507 ± 0.017 / 0.525 ± 0.033 | 0.539 ± 0.038 / 0.603 ± 0.026 |
| F2 | 0.500 ± 0.046 / 0.594 ± 0.034 | 0.558 ± 0.027 / 0.558 ± 0.028 |
| F3 | 0.483 ± 0.044 / 0.578 ± 0.037 | 0.535 ± 0.037 / 0.572 ± 0.025 |
| R1 | 0.564 ± 0.030 | 0.555 ± 0.043 |
| ref-age | 0.503 ± 0.021 | 0.503 ± 0.021 |
| ref-age+hpv | 0.512 ± 0.027 | 0.512 ± 0.027 |
| ref-site | 0.500 ± 0.000 | 0.500 ± 0.000 |
<!-- end generated -->

Image + covariates (D36), PET then CT:

<!-- generated: adjusted hecktor22_rfs_pet -->
| Run | AUROC | Within-centre AUROC | C-index | Within-centre C-index |
|---|---|---|---|---|
| `covariates_age_hpv_group` | 0.650 ± 0.014 | 0.499 ± 0.031 | 0.614 ± 0.015 | 0.499 ± 0.016 |
| `cnn_frozen_mean` | 0.548 ± 0.029 | 0.529 ± 0.053 | 0.540 ± 0.020 | 0.539 ± 0.038 |
| `cnn_frozen_mean_plus_age_hpv_group` | 0.642 ± 0.031 | 0.525 ± 0.057 | - | - |
| `covariates_group` | 0.644 ± 0.006 | 0.500 ± 0.000 | 0.613 ± 0.012 | 0.500 ± 0.000 |
| `cnn_frozen_mean_plus_group` | 0.644 ± 0.024 | 0.529 ± 0.053 | 0.622 ± 0.024 | 0.539 ± 0.038 |
| `cnn_frozen_roi_margin_mean` | 0.550 ± 0.039 | 0.513 ± 0.033 | 0.559 ± 0.032 | 0.535 ± 0.037 |
| `cnn_frozen_roi_margin_mean_plus_age_hpv_group` | 0.647 ± 0.033 | 0.502 ± 0.031 | - | - |
| `cnn_frozen_roi_margin_mean_plus_group` | 0.642 ± 0.032 | 0.513 ± 0.033 | 0.622 ± 0.026 | 0.535 ± 0.037 |
| `cnn_frozen_roi_mean` | 0.568 ± 0.053 | 0.531 ± 0.051 | 0.575 ± 0.033 | 0.558 ± 0.027 |
| `cnn_frozen_roi_mean_plus_age_hpv_group` | 0.649 ± 0.021 | 0.529 ± 0.050 | - | - |
| `cnn_frozen_roi_mean_plus_group` | 0.650 ± 0.026 | 0.531 ± 0.051 | 0.633 ± 0.019 | 0.558 ± 0.027 |
| `radiomics_original` | 0.611 ± 0.025 | 0.553 ± 0.024 | 0.596 ± 0.019 | 0.555 ± 0.043 |
| `radiomics_original_plus_age_hpv_group` | 0.661 ± 0.018 | 0.534 ± 0.020 | - | - |
| `radiomics_original_plus_group` | 0.668 ± 0.021 | 0.553 ± 0.024 | 0.631 ± 0.013 | 0.555 ± 0.043 |
| `vit_frozen_mean` | 0.656 ± 0.040 | 0.616 ± 0.040 | 0.633 ± 0.038 | 0.603 ± 0.026 |
| `vit_frozen_mean_plus_age_hpv_group` | 0.680 ± 0.024 | 0.615 ± 0.036 | - | - |
| `vit_frozen_mean_plus_group` | 0.679 ± 0.014 | 0.616 ± 0.040 | 0.638 ± 0.027 | 0.603 ± 0.026 |
| `vit_frozen_roi_margin_mean` | 0.658 ± 0.040 | 0.595 ± 0.038 | 0.637 ± 0.024 | 0.572 ± 0.025 |
| `vit_frozen_roi_margin_mean_plus_age_hpv_group` | 0.679 ± 0.028 | 0.596 ± 0.038 | - | - |
| `vit_frozen_roi_margin_mean_plus_group` | 0.683 ± 0.023 | 0.596 ± 0.037 | 0.657 ± 0.018 | 0.572 ± 0.025 |
| `vit_frozen_roi_mean` | 0.647 ± 0.016 | 0.587 ± 0.029 | 0.622 ± 0.032 | 0.558 ± 0.028 |
| `vit_frozen_roi_mean_plus_age_hpv_group` | 0.678 ± 0.020 | 0.586 ± 0.028 | - | - |
| `vit_frozen_roi_mean_plus_group` | 0.674 ± 0.016 | 0.587 ± 0.029 | 0.640 ± 0.022 | 0.551 ± 0.038 |
<!-- end generated -->

<!-- generated: adjusted hecktor22_rfs_ct -->
| Run | AUROC | Within-centre AUROC | C-index | Within-centre C-index |
|---|---|---|---|---|
| `covariates_age_hpv_group` | 0.650 ± 0.014 | 0.499 ± 0.031 | 0.614 ± 0.015 | 0.499 ± 0.016 |
| `cnn_frozen_mean` | 0.533 ± 0.039 | 0.508 ± 0.054 | 0.533 ± 0.015 | 0.507 ± 0.017 |
| `cnn_frozen_mean_plus_age_hpv_group` | 0.641 ± 0.032 | 0.490 ± 0.040 | - | - |
| `covariates_group` | 0.644 ± 0.006 | 0.500 ± 0.000 | 0.613 ± 0.012 | 0.500 ± 0.000 |
| `cnn_frozen_mean_plus_group` | 0.634 ± 0.030 | 0.491 ± 0.032 | 0.596 ± 0.029 | 0.509 ± 0.015 |
| `cnn_frozen_roi_margin_mean` | 0.523 ± 0.022 | 0.504 ± 0.052 | 0.513 ± 0.035 | 0.483 ± 0.044 |
| `cnn_frozen_roi_margin_mean_plus_age_hpv_group` | 0.645 ± 0.025 | 0.506 ± 0.038 | - | - |
| `cnn_frozen_roi_margin_mean_plus_group` | 0.627 ± 0.029 | 0.504 ± 0.052 | 0.582 ± 0.049 | 0.481 ± 0.040 |
| `cnn_frozen_roi_mean` | 0.508 ± 0.025 | 0.493 ± 0.051 | 0.523 ± 0.028 | 0.500 ± 0.046 |
| `cnn_frozen_roi_mean_plus_age_hpv_group` | 0.649 ± 0.018 | 0.496 ± 0.024 | - | - |
| `cnn_frozen_roi_mean_plus_group` | 0.637 ± 0.019 | 0.489 ± 0.049 | 0.606 ± 0.023 | 0.494 ± 0.035 |
| `radiomics_original` | 0.602 ± 0.061 | 0.558 ± 0.041 | 0.606 ± 0.019 | 0.564 ± 0.030 |
| `radiomics_original_plus_age_hpv_group` | 0.667 ± 0.017 | 0.531 ± 0.042 | - | - |
| `radiomics_original_plus_group` | 0.678 ± 0.016 | 0.558 ± 0.041 | 0.643 ± 0.006 | 0.564 ± 0.030 |
| `vit_frozen_mean` | 0.614 ± 0.012 | 0.526 ± 0.044 | 0.589 ± 0.013 | 0.525 ± 0.033 |
| `vit_frozen_mean_plus_age_hpv_group` | 0.653 ± 0.016 | 0.524 ± 0.048 | - | - |
| `vit_frozen_mean_plus_group` | 0.651 ± 0.013 | 0.520 ± 0.046 | 0.614 ± 0.014 | 0.521 ± 0.035 |
| `vit_frozen_roi_margin_mean` | 0.636 ± 0.022 | 0.580 ± 0.037 | 0.623 ± 0.010 | 0.578 ± 0.037 |
| `vit_frozen_roi_margin_mean_plus_age_hpv_group` | 0.661 ± 0.027 | 0.557 ± 0.046 | - | - |
| `vit_frozen_roi_margin_mean_plus_group` | 0.665 ± 0.020 | 0.558 ± 0.045 | 0.641 ± 0.022 | 0.573 ± 0.046 |
| `vit_frozen_roi_mean` | 0.651 ± 0.015 | 0.590 ± 0.028 | 0.634 ± 0.014 | 0.594 ± 0.034 |
| `vit_frozen_roi_mean_plus_age_hpv_group` | 0.675 ± 0.011 | 0.584 ± 0.021 | - | - |
| `vit_frozen_roi_mean_plus_group` | 0.674 ± 0.011 | 0.590 ± 0.028 | 0.650 ± 0.004 | 0.594 ± 0.034 |
<!-- end generated -->

PET + CT fusion (D30), every run:

<!-- generated: runs hecktor22_rfs_fused -->
| Run | Design | AUROC | Within-centre AUROC |
|---|---|---|---|
| `vit_fused_stack_roi_mean_logreg` | M4 | 0.669 ± 0.017 | 0.595 ± 0.011 |
| `vit_fused_concat_roi_margin_mean_logreg` |  | 0.666 ± 0.035 | 0.598 ± 0.025 |
| `vit_fused_stack_roi_margin_mean_logreg` |  | 0.664 ± 0.033 | 0.595 ± 0.033 |
| `vit_fused_mean_roi_margin_mean_logreg` |  | 0.662 ± 0.042 | 0.608 ± 0.025 |
| `vit_fused_concat_roi_mean_logreg` | M2 | 0.657 ± 0.039 | 0.583 ± 0.022 |
| `vit_fused_stack_mean_logreg` |  | 0.653 ± 0.023 | 0.599 ± 0.025 |
| `vit_fused_mean_roi_mean_logreg` | M3 | 0.652 ± 0.038 | 0.591 ± 0.029 |
| `vit_fused_mean_mean_logreg` |  | 0.650 ± 0.029 | 0.596 ± 0.030 |
| `vit_fused_concat_mean_logreg` |  | 0.635 ± 0.031 | 0.586 ± 0.031 |
| `radiomics_original_fused_stack_logreg` |  | 0.612 ± 0.028 | 0.557 ± 0.025 |
| `radiomics_original_fused_concat_logreg` |  | 0.600 ± 0.055 | 0.564 ± 0.037 |
| `cnn_fused_concat_roi_mean_logreg` | M2 | 0.557 ± 0.039 | 0.523 ± 0.022 |
| `cnn_fused_concat_roi_margin_mean_logreg` |  | 0.555 ± 0.023 | 0.511 ± 0.024 |
| `cnn_fused_stack_roi_margin_mean_logreg` |  | 0.545 ± 0.042 | 0.506 ± 0.038 |
| `cnn_fused_mean_roi_mean_logreg` | M3 | 0.544 ± 0.029 | 0.505 ± 0.015 |
| `cnn_fused_mean_roi_margin_mean_logreg` |  | 0.538 ± 0.023 | 0.515 ± 0.022 |
| `cnn_fused_stack_roi_mean_logreg` | M4 | 0.533 ± 0.040 | 0.509 ± 0.049 |
| `cnn_fused_stack_mean_logreg` |  | 0.531 ± 0.033 | 0.508 ± 0.068 |
| `cnn_fused_mean_mean_logreg` |  | 0.529 ± 0.013 | 0.503 ± 0.035 |
| `cnn_fused_concat_mean_logreg` |  | 0.523 ± 0.043 | 0.501 ± 0.061 |
<!-- end generated -->

<!-- generated: cindex hecktor22_rfs_ct hecktor22_rfs_pet -->
| ID | hecktor22_rfs_ct | hecktor22_rfs_pet |
|----|---|---|
| F1 | 0.533 ± 0.015 / 0.589 ± 0.013 | 0.540 ± 0.020 / 0.633 ± 0.038 |
| F2 | 0.523 ± 0.028 / 0.634 ± 0.014 | 0.575 ± 0.033 / 0.622 ± 0.032 |
| F3 | 0.513 ± 0.035 / 0.623 ± 0.010 | 0.559 ± 0.032 / 0.637 ± 0.024 |
| R1 | 0.606 ± 0.019 | 0.596 ± 0.019 |
| ref-age | 0.508 ± 0.004 | 0.508 ± 0.004 |
| ref-age+hpv | 0.536 ± 0.014 | 0.536 ± 0.014 |
| ref-site | 0.613 ± 0.012 | 0.613 ± 0.012 |
<!-- end generated -->

### hecktor22_rfs_*_official (challenge train/test split, D37)

Same patients, crops and features as `hecktor22_rfs_*`, one fold: the challenge's training
cohort (7 centres) as train + val, its test cohort (CHB, MDA, USZ) as test. 342 train, 86
val, 264 test; 67 test positives (25%, vs 15% in train). CHB and USZ are not in training.
One test set, so no ± std: with 67 positives and 197 negatives the AUROC standard error
is about 0.04, so only differences above ~0.08 are clear. Comparable in kind with the
challenge Task 2 leaderboard (about 0.68 C-index), which used all RFS patients rather than
our 2-year cohort.

Results, 2026-11-25:
- **The ViT holds up on unseen centres:** ROI pooling 0.678-0.690 AUROC (PET F3 0.690, CT
  F2 0.678), at or above the random splits (0.636-0.658); within-centre 0.606-0.642.
  C-index 0.659-0.664 (PET F3, CT F3), near the leaderboard's ~0.68.
- **Radiomics catches up:** R1 0.662-0.667 AUROC (random splits 0.60-0.61), within-centre
  0.643-0.645, C-index 0.634-0.653. On this split the ViT and radiomics are level.
- **PET + CT fusion helps a little:** the M2-M4 fusions (ROI mean) 0.683-0.691; the best of
  the 9 ViT fusions, the mean of the ROI + 10 mm embeddings, 0.709 (0.662 within-centre), but
  picking the best of 9 on one test set is optimistic. Radiomics fusion 0.671-0.673.
- **The centre-only reference is weaker here** (0.641 AUROC, 0.621 C-index): a centre not
  in training gets the overall training rate, so it can only separate MDA from the rest.
  Age + centre falls to 0.544 (unseen centres get no centre column). Age alone is at chance.
- **CT whole-crop ViT drops** (F1 0.562 vs ROI 0.678); PET does not (0.660). The CNN is
  at chance throughout (0.49-0.55).

AUROC, within-centre AUROC, C-index, within-centre C-index, then every fused run:

<!-- generated: design hecktor22_*_official -->
| ID | hecktor22_rfs_ct_official | hecktor22_rfs_fused_official | hecktor22_rfs_pet_official |
|----|---|---|---|
| F1 | 0.487 (n=1) / 0.562 (n=1) | - / - | 0.530 (n=1) / 0.660 (n=1) |
| F2 | 0.488 (n=1) / 0.678 (n=1) | - / - | 0.547 (n=1) / 0.654 (n=1) |
| F3 | 0.490 (n=1) / 0.679 (n=1) | - / - | 0.554 (n=1) / 0.690 (n=1) |
| T1 | - / - | - / - | - / - |
| T1-last | - / - | - / - | - / - |
| R1 | 0.667 (n=1) | - | 0.662 (n=1) |
| M2 | - / - | 0.506 (n=1) / 0.691 (n=1) | - / - |
| M3 | - / - | 0.506 (n=1) / 0.683 (n=1) | - / - |
| M4 | - / - | 0.534 (n=1) / 0.690 (n=1) | - / - |
| ref-age | 0.494 (n=1) | - | 0.494 (n=1) |
| ref-age+site | 0.544 (n=1) | - | 0.544 (n=1) |
| ref-age+hpv | 0.489 (n=1) | - | 0.489 (n=1) |
| ref | 0.641 (n=1) | n/a | 0.641 (n=1) |
<!-- end generated -->

<!-- generated: design-within hecktor22_*_official -->
| ID | hecktor22_rfs_ct_official | hecktor22_rfs_fused_official | hecktor22_rfs_pet_official |
|----|---|---|---|
| F1 | 0.492 (n=1) / 0.578 (n=1) | - / - | 0.567 (n=1) / 0.624 (n=1) |
| F2 | 0.506 (n=1) / 0.638 (n=1) | - / - | 0.546 (n=1) / 0.606 (n=1) |
| F3 | 0.520 (n=1) / 0.613 (n=1) | - / - | 0.557 (n=1) / 0.642 (n=1) |
| T1 | - / - | - / - | - / - |
| T1-last | - / - | - / - | - / - |
| R1 | 0.643 (n=1) | - | 0.645 (n=1) |
| M2 | - / - | 0.518 (n=1) / 0.642 (n=1) | - / - |
| M3 | - / - | 0.535 (n=1) / 0.641 (n=1) | - / - |
| M4 | - / - | 0.541 (n=1) / 0.640 (n=1) | - / - |
| ref-age | 0.452 (n=1) | - | 0.452 (n=1) |
| ref-age+site | 0.452 (n=1) | - | 0.452 (n=1) |
| ref-age+hpv | 0.456 (n=1) | - | 0.456 (n=1) |
| ref | 0.500 (n=1) | n/a | 0.500 (n=1) |
<!-- end generated -->

<!-- generated: cindex hecktor22_*_official -->
| ID | hecktor22_rfs_ct_official | hecktor22_rfs_pet_official |
|----|---|---|
| F1 | 0.478 (n=1) / 0.535 (n=1) | 0.559 (n=1) / 0.637 (n=1) |
| F2 | 0.510 (n=1) / 0.662 (n=1) | 0.484 (n=1) / 0.620 (n=1) |
| F3 | 0.504 (n=1) / 0.664 (n=1) | 0.584 (n=1) / 0.659 (n=1) |
| R1 | 0.653 (n=1) | 0.634 (n=1) |
| ref-age | 0.505 (n=1) | 0.505 (n=1) |
| ref-age+hpv | 0.502 (n=1) | 0.502 (n=1) |
| ref-site | 0.621 (n=1) | 0.621 (n=1) |
<!-- end generated -->

<!-- generated: cindex-within hecktor22_*_official -->
| ID | hecktor22_rfs_ct_official | hecktor22_rfs_pet_official |
|----|---|---|
| F1 | 0.503 (n=1) / 0.556 (n=1) | 0.587 (n=1) / 0.587 (n=1) |
| F2 | 0.535 (n=1) / 0.631 (n=1) | 0.470 (n=1) / 0.580 (n=1) |
| F3 | 0.540 (n=1) / 0.618 (n=1) | 0.588 (n=1) / 0.603 (n=1) |
| R1 | 0.625 (n=1) | 0.622 (n=1) |
| ref-age | 0.477 (n=1) | 0.477 (n=1) |
| ref-age+hpv | 0.481 (n=1) | 0.481 (n=1) |
| ref-site | 0.500 (n=1) | 0.500 (n=1) |
<!-- end generated -->

<!-- generated: runs hecktor22_rfs_fused_official -->
| Run | Design | AUROC | Within-centre AUROC |
|---|---|---|---|
| `vit_fused_mean_roi_margin_mean_logreg` |  | 0.709 (n=1) | 0.662 (n=1) |
| `vit_fused_concat_roi_margin_mean_logreg` |  | 0.701 (n=1) | 0.641 (n=1) |
| `vit_fused_stack_roi_margin_mean_logreg` |  | 0.698 (n=1) | 0.647 (n=1) |
| `vit_fused_concat_roi_mean_logreg` | M2 | 0.691 (n=1) | 0.642 (n=1) |
| `vit_fused_stack_roi_mean_logreg` | M4 | 0.690 (n=1) | 0.640 (n=1) |
| `vit_fused_mean_roi_mean_logreg` | M3 | 0.683 (n=1) | 0.641 (n=1) |
| `radiomics_original_fused_stack_logreg` |  | 0.673 (n=1) | 0.652 (n=1) |
| `radiomics_original_fused_concat_logreg` |  | 0.671 (n=1) | 0.653 (n=1) |
| `vit_fused_stack_mean_logreg` |  | 0.653 (n=1) | 0.609 (n=1) |
| `vit_fused_concat_mean_logreg` |  | 0.649 (n=1) | 0.577 (n=1) |
| `vit_fused_mean_mean_logreg` |  | 0.611 (n=1) | 0.592 (n=1) |
| `cnn_fused_concat_mean_logreg` |  | 0.541 (n=1) | 0.572 (n=1) |
| `cnn_fused_stack_roi_mean_logreg` | M4 | 0.534 (n=1) | 0.541 (n=1) |
| `cnn_fused_stack_roi_margin_mean_logreg` |  | 0.520 (n=1) | 0.537 (n=1) |
| `cnn_fused_mean_roi_margin_mean_logreg` |  | 0.516 (n=1) | 0.541 (n=1) |
| `cnn_fused_mean_mean_logreg` |  | 0.513 (n=1) | 0.519 (n=1) |
| `cnn_fused_concat_roi_mean_logreg` | M2 | 0.506 (n=1) | 0.518 (n=1) |
| `cnn_fused_mean_roi_mean_logreg` | M3 | 0.506 (n=1) | 0.535 (n=1) |
| `cnn_fused_stack_mean_logreg` |  | 0.501 (n=1) | 0.514 (n=1) |
| `cnn_fused_concat_roi_margin_mean_logreg` |  | 0.499 (n=1) | 0.513 (n=1) |
<!-- end generated -->

### totalseg_*_organs (multi-class organ recognition, D38)

Balanced accuracy, accuracy and macro one-vs-rest AUROC; cells `CNN / ViT`.

<!-- generated: multiclass totalseg_* -->
No results yet.
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
- Added HECKTOR 2022 (D26): training and test pooled for recurrence-free survival, one
  task per modality (`hecktor2022_rfs_ct`, `hecktor2022_rfs_pet`). Checked all 883
  patients: masks on the CT grid (48 with one extra row or slice), PET in SUV, the
  GTVp ∪ GTVn union under 192 mm everywhere. Preprocessing can now keep the ROI labels in
  the exported mask and clamp intensities per task.
- Survival probes (D27): ridge Cox (scikit-survival 0.28) with the penalty chosen on val
  C-index, Harrell's and within-centre C-index, a survival `group_prior` (centre's events
  per day of follow-up). New poolings for tasks with named ROIs: GTVp only (F5), GTVn only
  (F6), concatenation (F7). Radiomics now always uses the union of the mask labels.
  Smoke-tested end to end on 30 cases (CNN on CPU: CUDA does not initialise in the
  container since its restart this morning); the smoke results were deleted.

### 2026-11-19
- UPENN on the host: all 4 sequences prepared (452 cases, no failures, identical splits),
  probes, radiomics, age reference, Cox probes and fusion run. Age alone 0.667 AUROC / 0.619
  C-index; best ViT T2 F3 0.655 / 0.623; CNN near chance; fused concat (M2) 0.661.
- `nnfb-cox` CLI fixed: a single-command typer app made `probe` an extra argument, so the
  first `run_probes.sh` pass skipped every Cox probe. A CLI test now covers it.
- Merged `main` (Valentin: HECKTOR 2022 RFS D26, survival probes D27) into
  `brats-glioma-type`. Our decisions renumbered D26-D30 -> D28-D32. New D33: survival-only
  tasks (label = event) get Cox probes in `nnfb-features probe`; tasks with a binary label
  plus time/event (UPENN) keep logistic probes and get Cox from `nnfb-cox`. BraTS and UPENN
  tables regenerated with the merged code: same numbers.
- HECKTOR 2022 recurrence-free survival added (D34): `hecktor22_rfs_<pet|ct>` on the
  challenge release Valentin linked (training + test pooled, 9 centres). Label = recurrence
  within 2 years; 113 patients censored earlier dropped; 714 labelled, 692 with GTVp
  (134 positive, 169 events). New references age + HPV (AUROC and Cox) and centre (Cox).
  Prepare smoke-tested on 4 cases x 2 modalities: GTVp fully inside the crop, ~7-20 s per
  case; the case without GTVp is flagged for exclusion.
- HECKTOR 2022 on the host: both tasks prepared (692 cases, 22 without GTVp excluded, no
  failures, identical splits), probes, radiomics, references and Cox run. Centre alone 0.649
  AUROC / 0.613 C-index, level with the best ViT (0.658 PET F3); ViT within-centre AUROC up
  to 0.616; age and age + HPV near chance.
- Within-centre C-index (D35, `cindex-within` blocks) and image + covariate blends (D36,
  `nnfb-features combine` / `nnfb-cox combine`, `adjusted <task>` blocks). AUROC blends run
  here (seconds each): HECKTOR 2022 image + centre up to 0.683 vs centre alone 0.649; UPENN
  image + age up to 0.689 vs age alone 0.667. Cox blends left for the host (~10 min per
  ViT pooling).
- Cox blends (D36) on the host: HECKTOR 2022 image + centre C-index up to 0.657 (ViT PET
  F3) vs centre alone 0.613 and image alone 0.637; UPENN image + age at most 0.629 vs age
  alone 0.619 (no real gain).
- Merged `main` (PR #1 merged: BraTS, UPENN, Valentin's HECKTOR 2022 and survival probes)
  into `hecktor2022-rfs`. Our HECKTOR 2022 decisions renumbered D31-D33 -> D34-D36, the
  2-year variant moved to `datasets/hecktor2022_rfs2y.py` (Valentin's task keeps
  `hecktor2022.py`), the within-centre C-index of `nnfb-cox` runs now uses
  `survival.within_group_cindex`. BraTS, UPENN and `hecktor22_*` tables regenerated with
  the merged code: same numbers.

### 2026-11-25
- Official train/test split for HECKTOR 2022 (D37): tasks `<task>_official` for
  `hecktor22_rfs_<pet|ct>` and `hecktor2022_rfs_<ct|pet>`, one fold (challenge training
  cohort as train + val 80/20, test cohort CHB/MDA/USZ as test). `prepare` links the parent
  task's crops, features and radiomics, so only the probes run again. The 2-year variant's
  `cohort` column is now `official_split`, as in `hecktor2022.py`. The `hecktor22_*` and
  `hecktor2022_*` generated blocks now name their tasks, so the `_official` tasks get their
  own blocks. Unit-tested; to run on the host.
- `hecktor2022-rfs` merged into `main` (PR #2). Official split run on lxbelwahoo: prepare
  linked the parent tasks (342 train / 86 val / 264 test, 67 test positives), then probes,
  references, Cox and PET + CT fusion; no errors. ViT ROI pooling 0.678-0.690 AUROC on the
  challenge test cohort (two unseen centres), radiomics 0.662-0.667, centre alone 0.641,
  PET + CT ViT fusion 0.683-0.691 (M2-M4). C-index up to 0.664 (ViT CT F3). Tables generated on
  lxbelwahoo (official blocks only, its outputs/ being partial).

### 2026-11-26
- TotalSegmentator CT/MRI organ recognition (issue #4, D38): tasks `totalseg_ct_organs` (98
  classes) and `totalseg_mri_organs` (50) and their `_official` variants. `prepare` stores
  whole scans at 1 mm with a merged label map (scan layout); `extract` cuts each organ's
  192³ crop at extract time; the probe is a multinomial logistic regression scored by
  balanced accuracy, accuracy and macro AUROC (`multiclass` generated block). Other
  commands refuse organ tasks. Unit-tested on synthetic scans; to run on the host.

### 2026-11-27
- TotalSegmentator CT on lxbelwahoo: dataset unzipped on the NAS (1228 CT / 616 MRI subjects, linked
  in the shared data root). Smoke prepare on 4 CT scans: 3 failed in SimpleITK ("ITK only supports
  orthonormal direction cosines"): their sform directions are off orthonormal by ~2e-4 (in the CT
  and ~26 of 118 masks per scan). `preprocessing.read_image` now reads such files with nibabel and
  the nearest orthonormal direction (corner shift < 0.05 mm); nibabel declared as a dependency.
  Rerun: 4 scans, 124 samples, 79 classes, no failures.

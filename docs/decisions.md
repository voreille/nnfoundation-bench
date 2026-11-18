# Design decisions

One entry per decision: what we chose, why, and what we rejected. Numbering is
permanent. If a decision is revisited, add a new entry that supersedes the old one
instead of editing the old one. Status is one of: **active**, **superseded by Dn**,
**open**.

---

## D1. Fine-tune with the authors' trainer, glovita (active, 2026-11-12)
- **Choice:** MIC-DKFZ/image_classification (`glovita`), pinned to a fixed commit, with
  the official nnFoundation CNN/ViT checkpoints from Hugging Face, also pinned.
- **Why:** it's the authors' own fine-tuning path, so results are comparable with the
  paper. We only write data preparation, launching and evaluation.
- **Rejected:** our own training loop (risk of drifting from the paper's protocol).

## D2. Patch glovita from outside, never edit it (active, 2026-11-12)
- **Choice:** `modeling/glovita_entry.py` wraps glovita's train entry point:
  (a) fixed-length epochs of 50 × 32 samples, repeating the training cases as needed;
  (b) keep only `best.pt` and `last.pt`.
- **Why:** (a) glovita's `train_steps_per_epoch` only shortens epochs. With about 200
  training cases, 100 epochs would be about 600 optimizer steps instead of the paper's
  5000. (b) A checkpoint per epoch is about 40 GB per CNN run.
- **Rejected:** forking glovita (harder to follow upstream).

## D3. Preprocessing matches the checkpoints' `adaptation_plan.json` (active, 2026-11-12)
- **Choice:** 1 mm isotropic, 192³ crop, z-score with the whole-scan mean/std (nnU-Net
  `ZScoreNormalization`, no mask), arrays in SimpleITK (z, y, x) order. Crops are
  reoriented to an identity (LPS) direction. Areas outside the field of view are padded
  with -1024 HU (CT) or 0 SUV (PET).
- **Why:** this is what the encoders saw in pretraining, and it's the paper's
  classification protocol. We reorient because KiTS stores a permuted direction matrix.
- **Open:** whether the paper computes z-score statistics on the whole scan or on the crop.

## D4. Repeated stratified 40/10/50 splits, 5 folds (active, 2026-11-12)
- **Choice:** 5 independent 40/10/50 splits (seeds 12345 + i), used as glovita folds.
  Report the mean ± std of test AUROC.
- **Why:** the paper gives the 40/10/50 ratio but not the number of splits. Our test sets
  are small (KiTS Liver: about 5 positives), so a single split is too noisy.

## D5. Default to pretrained runs; from-scratch baselines are optional (active, 2026-11-12)
- **Choice:** `run_benchmark.sh` trains only the pretrained models; `SCRATCH=1` adds the
  random-init baselines.
- **Why:** the goal is to use the pretrained models in our pipeline, not to measure
  pretraining gains.

## D6. Focus on datasets with their own ROI masks (active, 2026-11-12)
- **Choice:** each task provides, per case, `CaseInputs(image, roi_mask, roi_label)`. The
  mask can be on another grid; everything is resampled in physical space. Prefer
  dataset-provided ROIs (`require_roi=True`: cases without a ROI are excluded and listed).
  ROI masks are exported as `processed/<task>/masks/*.b2nd` on the same 192³ grid as the
  image.
- **Why:** our pipeline is ROI-based, and the exported masks make ROI-pooling
  experiments possible.
- **Kept:** KiTS Liver uses a TotalSegmentator ROI and keeps the paper's fallback to the
  image centre. It stays as the paper-reproduction task.

## D7. HECKTOR DICOM is converted with okapy (active, 2026-11-12)
- **Choice:** okapy's conversion step (CT in HU, PET in SUVbw, one mask per SEG segment
  on the CT grid), run once per patient and cached under
  `data/interim/hecktor_euvip2024_hpv/converted/`. okapy is an editable dependency on
  `../okapy`.
- **Why:** it's our tested SUV and SEG conversion, the same as the rest of our pipeline.
  Bugs get fixed in okapy, not worked around here.

## D8. HECKTOR HPV baseline: PET only, crop centred on GTVp (active, 2026-11-12)
- **Choice:** task `hecktor_hpv_pet_gtvp`: PET SUVbw, single channel, 192³ crop centred
  on the GTVp bounding box. Labels: `Outcome` 1 = HPV positive (confirmed).
- **Why:** the simplest baseline with a single-channel input, matching how the models
  were pretrained.
- **Next (open):** CT, CT+PET channels, adding GTVn, ROI pooling.

## D9. Centre confounding in HECKTOR: centre × label splits and a centre-only baseline (active, 2026-11-12)
- **Fact:** HPV prevalence by centre: MDA 30/31, CHUM 7/8, HGJ 7/17, CHUS 2/8,
  CHUV 1/4, CHUP 5/34. Predicting from the centre alone gives AUROC 0.84.
- **Choice:** splits stratify on centre × label (strata smaller than 4 merge into their
  label group). `nnfb-predict group-baseline` reports the centre-only result
  (`group_prior`) next to the models.
- **Why:** to keep test sets comparable and to make the shortcut visible. Model AUROCs
  are read against 0.84.
- **Open:** leave-one-centre-out evaluation, which would remove the shortcut entirely.

## D10. Heavy jobs run on the host (active, 2026-11-12)
- **Choice:** full downloads, `prepare` over a whole dataset, training and full inference
  are run by the user on the host. In the container we only run tests and smoke tests on
  a few cases.
- **Why:** the container has a 64 MB `/dev/shm` (no DataLoader workers) and these jobs
  take hours.

## D11. No Makefile; uv CLIs plus one runner script (active, 2026-11-12)
- **Choice:** `nnfb-data`, `nnfb-train` and `nnfb-predict` CLIs, plus
  `scripts/run_benchmark.sh`. Skipping work that's already done lives in Python (and in
  the script's `metrics.json` check).
- **Why:** the Make targets were only aliases with no file dependencies. Training
  arguments change per run, so plain commands are clearer.

## D12. MRNet: one task per target x plane, assumed geometry, FOV as ROI (active, 2026-11-12)
- **Fact:** the public MRNet release (Stanford AIMI / Redivis, placed by hand on the NAS)
  has 1250 exams (1130 train + 120 valid; the 120-exam test set is withheld). Each exam
  has three separate uint8 `.npy` series, sagittal T2 FS, coronal T1 and axial PD, at
  256x256 with 17-61 slices. They carry **no spacing or orientation** and **no ROI masks**.
  Labels: abnormal 1008, ACL tear 262, meniscus tear 449 (not exclusive).
- **Choice:**
  - Nine tasks `mrnet_<abnormal|acl|meniscus>_<sagittal|coronal|axial>`. Each is a
    single-channel series, matching how the models were pretrained.
  - Assumed geometry (ASSUMPTION): in-plane spacing 0.625 mm, slice spacing 4 mm, and an
    axis mapping read off rendered slices. After reorientation every plane is
    (S-I, A-P, L-R).
  - The ROI is the series' field of view: the crop is centred on the volume, and the whole
    series fits in 192³ at 1 mm. The exported mask marks data vs padding. Pad value 0.
  - All 1250 public exams are pooled and split 40/10/50 with our usual 5 seeds, stratified
    on the label. The official train/valid split is not used.
- **Why:** this follows the paper's 1 mm / 192³ / 40/10/50 protocol as closely as the data
  allows. The paper lists MRNet only as a development dataset ("pathology") with no details.
- **Rejected:**
  - Stacking the three planes as channels: they are different series with no shared
    geometry, and the models take one channel.
  - Keeping the official split: its 120-exam valid set is too small for a 50% test share.
- **Open:**
  - The true spacing. MRNet's S1 Table is not online. A wrong slice spacing stretches the
    anatomy along one axis.
  - Patient leakage: 1130 train exams come from 1088 patients, and patient IDs are not
    released, so a patient can appear in both train and test.
  - MRNet has no ROI masks, so it does not fit the ROI focus of D6. It is mainly useful
    as a large, well-known MR check.

## D13. Frozen-feature linear probes next to glovita fine-tuning (active, 2026-11-12)
- **Choice:** `nnfb-features extract` loads the encoder with glovita's
  `nnFoundationEncoder`, so the weights and architecture are the same as in fine-tuning.
  It runs one bf16 forward pass per prepared crop and pools the last feature map:
  - CNN: 320 channels on a 6³ grid (32 mm cells);
  - ViT: 1056 channels on a 24³ token grid (8 mm cells).
  Poolings:
  - `mean`: equals glovita's pooled feature;
  - `roi_mean`: cells weighted by their ROI fraction;
  - `roi_margin_mean`: ROI dilated by 10 mm.
  `nnfb-features probe` then fits, per fold, a standardised L2 logistic regression with
  balanced class weights on train. C is chosen from 1e-4 to 1e2 by val AUROC; val is
  not refitted into the model, as with glovita's checkpoint selection. Results go to
  `reports/results/<task>/<model>_frozen_<pooling>_logreg/`.
- **Why:** we want to use the pretrained models as frozen feature extractors in our ROI
  pipeline. This is the cheapest way to compare poolings (minutes, not GPU-hours), and the
  results sit in the same summary as the fine-tuned runs and `group_prior`.
- **Rejected:**
  - Per-channel max pooling (whole crop or ROI). It takes each coordinate from a
    different location, so the pooled vector is not an embedding the encoder produces.
    This is arbitrary for the ViT, whose channels have no privileged basis. Only weighted
    means of real embeddings are kept. If we want to emphasise focal locations, use
    score-level max (MIL) or pick real cells instead.
  - glovita's `precomputed` encoder plus its head (an SGD-trained linear head on the
    same features). Fitting a logistic regression directly is simpler and makes tuning
    the regularisation easy.
- **Open:** the CNN's last grid is coarse (32 mm), so a small GTVp covers only 1-8
  cells. Pooling earlier stages (16 mm, 8 mm) is the obvious next step for the CNN.

## D14. Per-modality embeddings, late fusion (active, 2026-11-12)
- **Choice:** each modality (HECKTOR PET and CT, MRNet's three series) is its own
  single-channel task, with the same case IDs and splits. The frozen encoder gives
  one embedding per modality, and modalities are combined late:
  - feature level: concatenation (M2) or mean (M3) of the embeddings, then one probe;
  - score level: one probe per modality, stacked on val (M4);
  - later, learned attention across modalities (M5).
- **Why:** it matches how the models were pretrained (one channel). It also works when
  modalities share no grid (MRNet planes), lets each modality be cropped and pooled
  separately, and in our pipeline it lets us compute and cache embeddings once per modality.
- **Rejected:** early fusion as channels with the stem weights repeated / n (glovita's
  multi-channel option). With a frozen encoder it only encodes the channel average,
  and it needs a shared grid.
- **Requirement:** the tasks being fused must have identical `labels.json` and
  `splits.json`. If one modality excludes a case (for example a missing series), the
  splits differ and fusion must use the intersection with fresh splits.


## D15. LLD-MMRI: one task per MR phase, MedSAM2 lesion mask as ROI, benign vs malignant (active, 2026-11-12)
- **Fact:** the MedSAM2 release of LLD-MMRI 2023 (`wanglab/LLD-MMRI-MedSAM2`, revision
  `b7e8da56`) has 498 patients with one lesion each and eight series per patient (C-pre,
  C+A, C+V, C+Delay, T2WI, DWI, InPhase, OutPhase). Each series has its own grid (e.g. T2WI
  0.74 x 0.74 x 8 mm, arterial 0.74 x 0.74 x 2.6 mm) and its own 3D lesion mask on that grid.
  Intensities are uint8 0-255, rescaled by the MedSAM2 authors. 7 lesion classes; the
  annotation JSON groups them into benign (232) and malignant (266).
- **Choice:**
  - Eight tasks `lld_mmri_malignant_<phase>` (pre, arterial, venous, delay, t2, dwi,
    inphase, outphase). Each is single-channel, so per-modality late fusion (D14) applies.
  - ROI = the dataset's lesion mask for that series (`require_roi=True`, D6). Pad value 0.
  - Target: malignant (HCC, ICC, metastasis) vs benign (ASSUMPTION; the paper gives no
    target, and the challenge task is 7-class).
- **Why:** it ships 3D ROI masks, and it is one of the paper's test datasets.
- **Rejected:** the 7-class challenge target. Our pipeline and the paper report binary AUROC.
  A one-vs-rest task (e.g. HCC) can be added with another label function.
- **Note:** the case ID suffix (`MR177097_0`) is the lesion category (checked on 498/498),
  so case IDs carry the label. Nothing feeds them to a model, but never use them as features.
- **Open:** whether the 0-255 rescaling (per series, by MedSAM2) hurts compared with the
  original challenge intensities. Patients come from one hospital as far as we know, so
  there is no `group` column.

## D16. STOIC 2021: lungs from TotalSegmentator, targets severe and COVID (active, 2026-11-12)
- **Choice:** tasks `stoic_severe` (death or intubation at one month, the challenge's
  primary target) and `stoic_covid` (RT-PCR), from `metadata/reference.csv`
  (`probSevere`, `probCOVID`, 0/1). The crop is centred on the union of the five
  TotalSegmentator lung lobes (`require_roi=False`, image-centre fallback, as for KiTS).
  `CaseInputs.roi_label` now accepts a tuple of labels for such unions.
  `roi.segment_rois` converts non-NIfTI input (STOIC's .mha) to a temporary NIfTI.
- **Why:** paper test dataset; it has no ROI masks, so we follow the paper's
  TotalSegmentator approach. The lungs are wider than 192 mm (about 240 mm on the
  smoke-test case), so the crop covers their centre only.
- **Open:** the centre is not released, so splits stratify on the label only. Which
  target the paper used is not stated (ASSUMPTION: severe first).

## D17. RSNA Spine: read DICOM from the Kaggle zip, cervical spine from TotalSegmentator (active, 2026-11-12)
- **Choice:** task `rsna_spine_fracture`, label `patient_overall` from `train.csv`
  (ASSUMPTION; the paper lists the dataset for development only). The ~190 GB archive is
  not unpacked. Each study's `train_images/<uid>/*.dcm` is read from the zip into a
  temporary folder, read as one series with SimpleITK/GDCM (largest series if there are
  several), and cached as int16 NIfTI in HU under `data/interim/rsna_spine/nifti/`.
  The member list is cached in `zip_index.json`. ROI = union of TotalSegmentator
  vertebrae C1-C7, with the image-centre fallback.
- **Why:** unpacking would double ~190 GB on the NAS. The NIfTI cache is needed anyway
  because TotalSegmentator reads NIfTI. okapy is kept for PET/CT with SEG; a plain CT series
  needs only SimpleITK.
- **Rejected:** the 87 ground-truth `segmentations/`. They cover too few studies.
- **Open:** not tested on the real archive yet (the download was still running). Check
  slice order and HU on a few studies before the full prepare.

## D18. TotalSegmentator fast model runs on a 3 mm input we resample ourselves (active, 2026-11-12)
Applies to every TotalSegmentator ROI: KiTS liver, STOIC lungs, RSNA C1-C7.
- **Fact:** the fast model works at 3 mm. Given a full-resolution scan, TotalSegmentator
  resamples it to 3 mm, predicts, then resamples the mask back to the original grid. On a
  512 x 512 x 632 STOIC scan the two resamplings took ~40 s of a ~50 s call, and writing the
  temporary gzipped NIfTI input took another ~9 s. The prediction itself takes a few seconds.
- **Choice:** `roi.segment_rois` resamples the scan to 3 mm isotropic with SimpleITK
  (linear, same origin and direction, pad -1024) and writes it as an uncompressed temporary
  NIfTI. TotalSegmentator then runs on that. **The mask is kept on the 3 mm grid**; it is not
  resampled back. Preprocessing resamples masks in physical space, so any grid works.
  The call takes ~18 s per case (model load included) instead of ~50 s.
- **The masks are not identical to TotalSegmentator's own output, although both predict at
  3 mm:**
  1. *Different 3 mm input.* TotalSegmentator resamples with nnU-Net's code (spline
     interpolation, array stretched so the first and last voxels align with the original
     ones). We use SimpleITK (linear, 3 mm steps from the original origin). The spacing is
     the same, but the grid points sit at slightly different positions with slightly
     different values. The network is not perfectly shift-invariant, so some boundary
     voxels change label.
  2. *Different output grid.* TotalSegmentator resamples its 3 mm prediction back to the
     native grid (e.g. 0.66 mm), which gives a smoother boundary. Ours stays at 3 mm. The
     crop centre is the centre of the mask's bounding box, so it moves in 3 mm steps: up to
     ~1.5 mm of difference from this alone.
- **Measured** on STOIC case 1072 (lungs, union of the 5 lobes): centre 1.9 mm apart,
  volume 2444 ml vs 2433 ml (0.5%). Negligible for centring a 192 mm crop.
- **Consequences:**
  - The exported `masks/*.b2nd` are blocky at 3 mm. That is fine for organ-scale ROIs
    (lungs, liver, spine) in ROI-pooling experiments; the CNN's last grid is 32 mm anyway.
  - These masks are not a substitute for TotalSegmentator's native output, e.g. for organ
    volumes or small structures. For those, use `fast=False` (full model, native resampling).
  - Masks made before this change (native grid) stay valid and are reused; mixing both
    kinds in one task only moves crop centres by ~2 mm.
- **Rejected:** leaving TotalSegmentator's internal resampling (2.5x slower, about 28 h for
  STOIC's 2000 cases) and only dropping the gzip (saves ~9 s of ~50 s).
- **Open:** the model is reloaded for every case (most of the remaining ~18 s). Keeping it
  loaded, or segmenting several cases in parallel, would be the next speed-up.

## D19. RSNA Spine: read the unpacked folders, clamp CT below -1024 HU (active, 2026-11-13)
Supersedes the zip-reading part of D17; the rest of D17 stands.
- **Fact:** the archive was unpacked on the NAS (`train.csv`, `train_images/<uid>/*.dcm`,
  2019 studies) and the zip removed. On 3 studies checked: slices are sorted correctly (C1
  above C7 after conversion), spacing 0.25-0.44 mm in-plane and 0.4-0.8 mm between slices.
  About 22% of every scan lies outside the reconstruction circle and is stored as -2048,
  -3024 or -3669, depending on the scanner.
- **Choice:**
  - Read each study's folder directly. The zip index and temporary extraction are gone.
  - At conversion, clamp values below -1024 HU to -1024 (air, and `CT_PAD_VALUE`, the value
    we already use outside the field of view).
- **Why:** the padding is not tissue but enters the whole-scan z-score (D3). On one study the
  mean goes from -46 to -688 HU and the std from 389 to 1273. Because the padding value
  depends on the scanner, normalisation would too, which a model can pick up as a
  confound. After clamping: mean -257, std 530. STOIC (minimum -1024) and KiTS already
  look like this.
- **Rejected:** a mask-based z-score that leaves out padding. It departs from the
  `adaptation_plan.json` normalisation (no mask), and clamping gives the same effect with
  values the model saw in pretraining.
- **Open:** values above 3071 HU (metal, e.g. 10880 in one study) are left as they are.

## D20. Evaluate fine-tuned runs at both the best-val and the last checkpoint (active, 2026-11-13)
- **Fact:** glovita keeps `best.pt` (highest val AUROC over the 100 epochs) and `last.pt`.
  The val split is used only for that selection, never for training. On HECKTOR it holds
  10 cases (about 5 positives). Its AUROC moves in large steps and many epochs tie, so the
  "best" epoch is close to a random pick and optimistic on val.
- **Choice:** `nnfb-predict predict` evaluates `best` (default, results in `<run>/`) or
  `last` (results in `<run>_last/`). `run_benchmark.sh` runs both after each training
  run; inference is cheap next to training. Design table rows: T1 (best) and T1-last.
- **Why:** compares the paper's selection by val with no selection (the fixed schedule
  ends at LR 0), on small and large datasets.
- **Rejected:** overwriting one result folder with whichever checkpoint was evaluated last.
  The `checkpoint` field in metrics.json was the only trace, and `summarize` mixed them.
- **Open:** if `last` is as good as `best` on small datasets, the val cases could go back
  into train there. That ties in with repeated k-fold for small datasets (not decided).

## D21. Radiomics baseline: pyradiomics in the ROI, same probe and splits (active, 2026-11-13)
- **Choice:** `nnfb-radiomics extract <task> --features original|filtered`, then `probe`.
  - Input: the QC crops of `prepare` (`data/interim/<task>/{images,masks}/`), 1 mm
    isotropic, physical units, and the same ROI mask as the embedding poolings (F2).
    No resampling in pyradiomics.
  - Discretisation per modality: CT bin width 25 HU, PET 0.5 SUV, MR 32 bins (MR has no
    physical unit, and MRNet and LLD-MMRI are rescaled to 0-255 per series).
  - All feature classes (shape, first order, GLCM, GLRLM, GLSZM, GLDM, NGTDM).
    `original` = original image (107 features); `filtered` = plus LoG sigma 1 and 3 mm
    and the 8 wavelet decompositions (1037).
  - Classifier: the frozen-feature probe of D13 (median imputation, standardisation, L2
    logistic regression, balanced classes, C by val AUROC), same splits.
    Runs `radiomics_<features>_logreg`; design rows R1 and R2.
  - Features are cached per case as JSON under `data/processed/<task>/radiomics/<set>/`.
    Empty ROIs give no features and are imputed; inf is treated as missing.
- **Why:** a handcrafted reference tells whether the foundation-model embeddings beat what
  the ROI-based pipeline can already do. Sharing crop, mask, probe and splits makes the
  rows directly comparable.
- **Rejected:**
  - Extracting from the native-resolution images: per-dataset resampling choices, and the
    ROI would no longer match the embedding poolings.
  - Feature selection or a random forest: a different classifier would mix up the
    feature comparison with a model comparison. Can be added as another row.
- **Dependency:** pyradiomics is installed from GitHub tag `v3.1.0` (commit `6a761c4`), because
  the PyPI 3.1.0 sdist lacks `cmatrices.h` and does not build on Python 3.12. It works with
  NumPy 2.5.4. With NumPy 2, five shape features (Elongation, Flatness, the axis lengths)
  come back complex with a zero imaginary part. We keep the real part and raise if the
  imaginary part is not zero.
- **Open:**
  - Crops cut ROIs larger than 192 mm (STOIC lungs), so shape features there describe the
    truncated ROI.
  - TotalSegmentator masks are 3 mm (D18), which coarsens shape features.
  - The bin widths are common defaults, not tuned.

## D22. Radiomics redundancy: correlation filter and elastic net, fit per fold (active, 2026-11-13)
Extends D21; R1/R2 (no selection, L2) keep their meaning.
- **Choice:** two options on `nnfb-radiomics probe`, combinable:
  - `--corr-threshold t`: on each fold's train cases only, drop constant features, then,
    as in caret's `findCorrelation`, repeatedly take the most correlated pair with
    |Spearman rho| > t and drop the feature with the larger mean |rho| to the remaining
    ones. NaN get the column median before ranking. t = 0.9 for the table rows.
  - `--classifier enet`: elastic-net logistic regression (saga), with L1 share
    {0.1, 0.5, 0.9} and C chosen together on val. `fit_probe` now takes `l1_ratios`;
    `(0.0,)` is the old L2 probe.
  - Run names `radiomics_<set>[_corr<t>]_<logreg|enet>`. metrics.json records the
    number of features used, C and the L1 share. Design rows R3-R5 (filtered set).
- **Why:** radiomics features are strongly redundant: the same statistic on neighbouring
  filters, or near-duplicate texture features. 1037 correlated features against ~41
  training cases is where L2 alone suffers. Spearman, because many features are skewed
  or related monotonically but not linearly.
- **Results (HECKTOR):** filtered set: R2 0.786 → R3 0.814 (about 305 features kept) and
  R4 0.817. On the 107 original features both are worse than plain L2 (0.68-0.71 vs 0.744).
- **Rejected:**
  - Selecting features on all cases before splitting: the selection would see the test
    labels' cases. It is unsupervised, so it doesn't leak labels, but it still leaks
    test-set statistics.
  - ICC / test-retest filtering: we have no repeat scans.
  - Supervised univariate filters (e.g. top-k by AUROC): easy to overfit on ~41 cases.
- **Open:** the thresholds (0.9) and the L1 grid are common defaults, not tuned. Val
  selection on 10 cases is noisy (C ranges from 1e-4 to 10 across folds). Repeated k-fold
  for small datasets would help all probe rows.

## D23. Results in `outputs/`, result tables generated from them (active, 2026-11-17)
Supersedes the results location in D13 (`reports/results/`) and the versioning of
`metrics.json` / `predictions.csv` in git.
- **Choice:**
  - Machine outputs go to `outputs/` (git-ignored): `outputs/results/<task>/<run>/fold<k>/`
    and `outputs/logs/<task>/`. `reports/` holds only what people read (figures).
  - Result tables in `docs/progress.md` are generated blocks
    (`<!-- generated: design -->`, `<!-- generated: runs <task> -->`), rebuilt by
    `uv run nnfb-predict table --update docs/progress.md`. The design catalogue keeps
    the hand-written descriptions; the results grid next to it is generated.
- **Why:** tables typed from memory drift from the results (e.g. the R3 std, 0.018 vs
  0.017). Per-fold predictions are outputs to regenerate, not documents to version;
  `reports/` should be readable by people.
- **Rejected:** keeping results in git for history: the generated tables in the docs
  carry the numbers that matter, and git history of the docs keeps them.
- **Open:** results now exist on one disk only; back up `outputs/` if it matters.

## D24. Within-centre AUROC as the confound-free metric for multi-centre tasks (active, 2026-11-18)
Extends D9 and replaces its open item (leave-one-centre-out) for HECKTOR.
- **Choice:** for tasks with a `group` column, also report a within-group AUROC: the
  fraction of correctly ranked (positive, negative) pairs, counting only pairs from the
  same group, pooled over groups (a stratified Mann-Whitney statistic; ties count 1/2).
  It is computed per fold from the existing `predictions.csv` and the groups in
  `labels.csv`, so it applies to every past and future run without rerunning. Shown by
  `nnfb-predict table --within`, the `design-within` generated grid, and a second column in
  the `runs <task>` tables.
- **Why:** a score that depends on the group alone (`group_prior`) gets exactly 0.5, so
  the centre shortcut is removed from the metric itself. Plain AUROC stays the main metric
  (paper protocol); the within-centre one says how much of it is not the centre.
- **Rejected:**
  - AUROC per centre, averaged: on HECKTOR, MDA (30/31 positive), CHUM (7/8) and CHUV (1/4)
    have one case of a class or less in a test split, so most per-centre AUROCs are
    undefined or rest on 1-2 cases. Pair pooling uses whatever pairs exist.
  - Leave-one-centre-out (E1) on HECKTOR: same problem, held-out MDA would be 30 positives
    against 1 negative. Kept as a design for better-balanced multi-centre data.
  - LEACE concept erasure as the metric fix: it changes the features, not the evaluation,
    needs shrinkage at 41 train cases, and also removes real signal correlated with the
    centre. Parked in the backlog for centre-free features.
- **Open:** on HECKTOR a test split holds only 66-81 within-centre pairs, more than half
  from CHUP, so the metric is noisy (std 0.04-0.16).


## D25. Drop the filtered radiomics set and rows R2-R5 (active, 2026-11-18)
Supersedes the R2-R5 rows of D21 and D22; R1 (107 original-image features, L2 probe) stays.
- **Choice:** `scripts/run_probes.sh` extracts the `original` set only and fits the plain
  L2 probe (R1). R2-R5 leave the design grids (status "dropped"). The code
  (`--features filtered`, `--corr-threshold`, `--classifier enet`) stays, and past runs and
  cached features are kept; their runs still show in the per-task runs tables.
- **Why:** the backlog trigger was met. On STOIC severe, STOIC covid and all 8 LLD-MMRI
  phases, R2-R5 are at or below R1. HECKTOR was the only task where they beat R1 in plain
  AUROC (0.79-0.82 vs 0.74), but within-centre (D24) R1 is best (0.66 vs 0.58-0.63), so
  the gain was centre signal. The filtered set costs about 10x the extraction time.
- **Also dropped:** correlation filter and elastic net on the original set: never above
  R1 (HECKTOR 0.68-0.71 vs 0.744).
- **Rejected:** keeping the filtered set for the remaining tasks (RSNA, KiTS) "for
  completeness": hours of extraction for a baseline that has not helped anywhere.

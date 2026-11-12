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

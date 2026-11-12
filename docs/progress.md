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
2. Run the first baseline: `FOLDS=0 scripts/run_benchmark.sh hecktor_hpv_pet_gtvp`.
3. Then work on design choices: CT+PET channels, adding GTVn, ROI pooling of features
   (frozen-feature probes), and leave-one-centre-out evaluation.
4. KiTS Liver full run (background, paper reproduction).
5. MRNet: prepare and run `mrnet_abnormal_sagittal` first (closest to the paper's
   "pathology"). Each task needs about 12 GB in `data/processed`. See D12.

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


# nnFoundation classification protocol (arXiv:2609.26924)

Notes from the v1 HTML paper, read 2026-11-12.

## Models
- Two models: nnFoundation CNN and nnFoundationViT (transformer).
- Pretrained on 2.1M CT/MR/PET volumes from 125 datasets (Human Radiome Project).
- Pretraining code: https://github.com/MIC-DKFZ/nnssl.
- Weights: https://huggingface.co/MIC-DKFZ/nnFoundationCNN (ResEnc-L, 102M) and
  https://huggingface.co/MIC-DKFZ/nnFoundationViT (PrimusX, 674M).
- Classification fine-tuning code: https://github.com/MIC-DKFZ/image_classification
  (package `glovita`, docs/nnfoundation.md).
- The checkpoints' `adaptation_plan.json` says: input is single-channel, z-score normalised
  (`ZScoreNormalization`, no mask), not resampled by the model (pretraining data was at
  ~1 mm), read with SimpleITK with no transpose, and the recommended patch size is 192^3.

## Classification fine-tuning
- Inputs resampled to **1 mm isotropic**, **192x192x192 crop around the target region**.
  Target regions come from TotalSegmentator (organs) or HD-BET (brain).
- Encoder -> adaptive average pooling -> MLP head. Full fine-tuning.
- AdamW, mixed precision, default nnU-Net augmentations.
- 100 epochs x 50 steps, effective batch size 32 (gradient accumulation).
  Linear warm-up over the first 10% of epochs, then cosine decay.
  The reduced-compute setting uses 20 epochs.
- Peak LR tuned per model on an internal CT-RATE validation split (Table 1):
  **nnFoundationCNN 3e-4, nnFoundationViT 1e-4** (others up to 1e-1 for CURIA).
- Split: **40/10/50 train/val/test**. CT-RATE uses its official test split.
- Metric: **AUROC**. Results are summarised by mean rank across datasets.

## Classification datasets (Table 9)
| Dataset          | Cases | Modality | Anatomy | Role |
|------------------|------:|----------|---------|------|
| RSNA Spine       | 2019  | CT  | spine   | development |
| MRNet            | 1370  | MR  | knee    | development |
| CT-RATE          | 25000 | CT  | chest   | test |
| RSNA Aneurysm    | 4344  | CTA | brain   | test |
| RSNA Hemorrhage  | 18938 | CT  | brain   | test |
| STOIC 2021       | 2000  | CT  | chest   | test |
| ODELIA           | 1022  | MR  | breast  | test |
| LLD-MMRI2023     | 498   | MR  | abdomen | test |
| KiTS Liver       | 489   | CT  | abdomen | test |
| KiTS Pancreas    | 489   | CT  | abdomen | test |

Table 9 lists the ODELIA anatomy as "chest". The text's "8 classification
datasets" matches the 8 test rows; RSNA Spine and MRNet were used only for development.

## Our implementation and deviations
- Trainer: glovita at a pinned commit. Settings follow its nnFoundation example:
  AdamW, weight decay 0.01, `init_std` 0.01 head, best checkpoint by val AUROC, batch size 1
  with gradient accumulation up to 32. Warm-up is 10% of epochs.
- **Fixed-length epochs:** glovita's `train_steps_per_epoch` only truncates epochs. With
  ~200 training cases, 100 "epochs" would be ~600 optimizer steps instead of the paper's
  5000. `modeling/glovita_entry.py` repeats the training cases so each epoch holds
  50 x 32 samples.
- Precision: bf16. The paper says "mixed precision" and glovita's example uses fp16.
- Normalisation: z-score with the mean/std of the whole original scan, as nnU-Net/nnssl
  preprocessing does before patch sampling. The paper does not say whether the stats come
  from the scan or the crop.
- Orientation: crops are resampled into an identity-direction grid. nnU-Net does not
  reorient, but KiTS23 stores a permuted direction matrix, and most pretraining CTs are
  near-identity.
- Splits: 5 repeated stratified 40/10/50 splits (seeds 12345+i), used as glovita folds.
  The paper does not say how many it used.
- Metric: AUROC computed with sklearn from the softmax P(class 1) in glovita's
  `predictions.pt`.

## Open questions
- **KiTS Liver / Pancreas labels are not defined in the paper.** For liver we
  currently use the KiTS23 clinical comorbidities `mild_liver_disease OR
  moderate_to_severe_liver_disease` (10/489 positives, so the 50% test split has
  about 5 positives). KiTS23 metadata has no pancreas field, which suggests the
  authors derived labels from another source (reports? lesion annotations?). Ask the
  authors or check the nnssl downstream repos.
- Do the authors use one split or several seeds? The paper says only "40/10/50".

## Our own tasks (not in the paper)
### `hecktor_hpv_pet_gtvp`
- Data: curated HECKTOR subset for HPV status (EUVIP 2024), DICOM CT + PET + SEG
  (segment 1 GTVp, 2 GTVn, defined on CT). Converted with okapy: PET to SUVbw.
- Baseline: PET only, 192^3 at 1 mm centred on the GTVp bounding box, z-scored like
  pretraining. Cases without GTVp are excluded. Combining CT/PET/GTVn is future work.
- Labels: HPV status, unknown excluded. Label file name and columns are ASSUMED to follow
  HECKTOR 2022 conventions (`PatientID`, `HPV status (0=-, 1=+)`) until confirmed.
- Folder names carry a curation index (`CHUM-029-86`); the label key is the DICOM
  PatientID (`CHUM-029`).

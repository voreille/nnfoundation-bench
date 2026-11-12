# nnFoundation classification protocol (arXiv:2609.26924)

Notes from the v1 HTML paper, read 2026-11-12.

## Models
- Two models: nnFoundation CNN and nnFoundationViT (transformer).
- Pretrained on 2.1M CT/MR/PET volumes from 125 datasets (Human Radiome Project).
- Code: https://github.com/MIC-DKFZ/nnssl. Released "integrated into nnU-Net and
  nnDetection". The paper gives no direct link to the weights.

## Classification fine-tuning
- Inputs resampled to **1 mm isotropic**, **192x192x192 crop around the target region**.
  Target regions come from TotalSegmentator (organs) or HD-BET (brain).
- Encoder -> adaptive average pooling -> MLP head. Full fine-tuning.
- AdamW, mixed precision, default nnU-Net augmentations.
- 100 epochs x 50 steps, effective batch size 32 (gradient accumulation).
  Linear warm-up over the first 10% of epochs, then cosine decay.
  The reduced-compute setting uses 20 epochs.
- Peak LR tuned per model on an internal CT-RATE validation split
  (1e-4 for CTFM / nnFoundationViT, up to 1e-1 for CURIA).
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

## Open questions
- **KiTS Liver / Pancreas labels are not defined in the paper.** For liver we
  currently use the KiTS23 clinical comorbidities `mild_liver_disease OR
  moderate_to_severe_liver_disease` (10/489 positives, so the 50% test split has
  about 5 positives). KiTS23 metadata has no pancreas field, which suggests the
  authors derived labels from another source (reports? lesion annotations?). Ask the
  authors or check the nnssl downstream repos.
- Do the authors use one split or several seeds? The paper says only "40/10/50".

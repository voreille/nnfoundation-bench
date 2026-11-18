"""UPENN-GBM (TCIA), pre-operative glioblastoma MR with overall survival.

Images: the TCIA NIfTI release on the NAS, linked as ``data/raw/UPENN_GBM/``
(``NIfTI-files/``). 611 pre-operative studies ``UPENN-GBM-<n>_11`` (plus 60 follow-ups
``_21``), each with ``<study>_{T1,T1GD,T2,FLAIR}.nii.gz`` in ``images_structural/<study>/``:
BraTS 2021 layout, 240 x 240 x 155 at 1 mm, co-registered, skull-stripped (background 0).
Tumour masks (labels 1 necrosis, 2 edema, 4 enhancing): ``images_segm/<study>_segm.nii.gz``
(expert-corrected, 147 studies) and ``automated_segm/<study>_automated_approx_segm.nii.gz``
(automatic, every pre-operative study).

Outcome: ``data/raw/GBM-UPENN-OS/`` (QuantImage2 export, placed by hand):
``outcome_QI_UPENN_os.csv`` (``PatientID``, ``Time`` in days, ``Event``) for 452 patients, all
deceased (no censoring), and ``clinical-variables_UPENN 1.csv`` (age, IDH, MGMT, resection;
one row per study, the pre-operative one has ``Time_since_baseline_preop == "0"``).
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from nnfoundation_bench.config import RAW_DATA_DIR
from nnfoundation_bench.datasets.base import CaseInputs, ClassificationTask

SOURCE = "upenn_gbm"
IMAGE_DIR = RAW_DATA_DIR / "UPENN_GBM" / "NIfTI-files"  # symlink to the NAS copy
OUTCOME_DIR = RAW_DATA_DIR / "GBM-UPENN-OS"
OUTCOME_PATH = OUTCOME_DIR / "outcome_QI_UPENN_os.csv"
CLINICAL_PATH = OUTCOME_DIR / "clinical-variables_UPENN 1.csv"
NUM_PATIENTS = 452

# File suffix -> (task suffix, description). One single-channel task per sequence (D14).
SEQUENCES = {
    "T1": ("t1", "T1 native"),
    "T1GD": ("t1gd", "T1 post-contrast"),
    "T2": ("t2", "T2-weighted"),
    "FLAIR": ("flair", "T2-FLAIR"),
}
WHOLE_TUMOUR = (1, 2, 4)  # necrosis + edema + enhancing (BraTS 2021 labels)
MR_PAD_VALUE = 0.0  # skull-stripped background
ONE_YEAR_DAYS = 365


def study_id(patient_id: str) -> str:
    """``UPENN_UPENN-GBM-00002`` (outcome file) -> pre-operative study ``UPENN-GBM-00002_11``."""
    return patient_id.removeprefix("UPENN_") + "_11"


def image_path(case_id: str, sequence: str) -> Path:
    return IMAGE_DIR / "images_structural" / case_id / f"{case_id}_{sequence}.nii.gz"


def mask_paths(case_id: str) -> dict[str, Path]:
    return {
        "expert": IMAGE_DIR / "images_segm" / f"{case_id}_segm.nii.gz",
        "automatic": IMAGE_DIR / "automated_segm" / f"{case_id}_automated_approx_segm.nii.gz",
    }


def mask_source(case_id: str) -> str | None:
    """``expert`` if an expert-corrected mask exists, else ``automatic`` (None if neither)."""
    return next((k for k, p in mask_paths(case_id).items() if p.exists()), None)


def download() -> None:
    """Nothing to download: images (TCIA) and outcome files must be placed by hand."""
    for path in (IMAGE_DIR, OUTCOME_PATH, CLINICAL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing (UPENN-GBM is placed by hand)")
    cases = [study_id(p) for p in pd.read_csv(OUTCOME_PATH)["PatientID"]]
    missing = [
        f"{c}_{seq}" for c in cases for seq in SEQUENCES if not image_path(c, seq).exists()
    ] + [f"{c} mask" for c in cases if mask_source(c) is None]
    if len(cases) != NUM_PATIENTS or missing:
        raise RuntimeError(f"UPENN-GBM: {len(cases)} patients, missing: {missing[:5]}")
    sources = pd.Series([mask_source(c) for c in cases]).value_counts().to_dict()
    logger.info(f"UPENN-GBM: {len(cases)} patients, all 4 sequences, masks {sources}")


def survival_labels(
    outcome: pd.DataFrame, clinical: pd.DataFrame, mask_sources: dict[str, str]
) -> pd.DataFrame:
    """One row per patient of the outcome file (its pre-operative study).

    ``label`` = survived at least one year (``time >= 365`` days); ASSUMPTION, our binary
    view of the survival task so the AUROC probes apply. ``time`` (days) and ``event``
    (1 = death) are kept for the Cox probes (C-index, D30). Every patient in the file died,
    so there is no censoring; patients alive at last follow-up are not in the file.
    ``age`` comes from the pre-operative clinical row. ``mask_source`` says which ROI mask
    the case uses. Single centre: no ``group``.
    """
    preop = clinical[clinical["Time_since_baseline_preop"].astype(str) == "0"]
    age = preop.drop_duplicates("PatientID").set_index("PatientID")["age"]
    labels = pd.DataFrame(
        {
            "case_id": outcome["PatientID"].map(study_id),
            "label": (outcome["Time"] >= ONE_YEAR_DAYS).astype(int),
            "time": outcome["Time"].astype(float),
            "event": outcome["Event"].astype(int),
            "age": pd.to_numeric(outcome["PatientID"].map(age), errors="coerce"),
        }
    )
    labels["mask_source"] = labels["case_id"].map(mask_sources)
    if missing := labels.loc[labels["mask_source"].isna(), "case_id"].tolist():
        raise RuntimeError(f"No tumour mask for {missing[:5]}")
    return labels.sort_values("case_id").reset_index(drop=True)


def load_labels() -> pd.DataFrame:
    outcome = pd.read_csv(OUTCOME_PATH)
    clinical = pd.read_csv(CLINICAL_PATH, index_col=0)
    sources = {study_id(p): mask_source(study_id(p)) for p in outcome["PatientID"]}
    return survival_labels(outcome, clinical, sources)


def make_task(sequence: str) -> ClassificationTask:
    suffix, description = SEQUENCES[sequence]
    return ClassificationTask(
        name=f"upenn_os_{suffix}",
        source=SOURCE,
        description=f"UPENN-GBM pre-operative MR, {description} ({sequence}), crop around the "
        "whole tumour, target: overall survival >= 1 year (time/event kept for C-index)",
        target="os_1year",
        roi="whole tumour (expert-corrected mask, else automatic; labels 1+2+4)",
        modality="MR",
        download=download,
        load_labels=load_labels,
        case_inputs=lambda case_id, device: CaseInputs(
            image_path(case_id, sequence),
            mask_paths(case_id)[mask_source(case_id)],
            WHOLE_TUMOUR,
        ),
        pad_value=MR_PAD_VALUE,
    )


UPENN_OS_TASKS = tuple(make_task(sequence) for sequence in SEQUENCES)

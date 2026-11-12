"""STOIC 2021 COVID-19 chest CT (Revel et al., Radiology 2021), public training set.

Placed by hand from the AWS open-data bucket in ``data/raw/STOIC-2021/`` (symlink to the
NAS copy): ``data/mha/<PatientID>.mha`` (int16 HU) and ``metadata/reference.csv`` with
``PatientID,probCOVID,probSevere`` (0/1, despite the names). 2000 patients from
several French hospitals; the centre is not released, so splits stratify on the label.

There are no ROI masks. The crop is centred on the lungs (union of the five
TotalSegmentator lobes), cached under ``data/interim/stoic2021/totalseg/``. The lungs
are wider than 192 mm, so the crop covers their centre only (paper protocol).
"""

from functools import cache
from pathlib import Path

import pandas as pd
from loguru import logger

from nnfoundation_bench.config import CT_PAD_VALUE, INTERIM_DATA_DIR, RAW_DATA_DIR
from nnfoundation_bench.datasets.base import CaseInputs, ClassificationTask
from nnfoundation_bench.roi import segment_rois, totalseg_class_ids

SOURCE = "stoic2021"
RAW_DIR = RAW_DATA_DIR / "STOIC-2021"  # symlink to the NAS copy
REFERENCE_PATH = RAW_DIR / "metadata" / "reference.csv"
TOTALSEG_DIR = INTERIM_DATA_DIR / SOURCE / "totalseg"
NUM_CASES = 2000

LUNG_LOBES = (
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
)
# Task target -> reference.csv column.
TARGETS = {
    # Death or intubation within one month: the challenge's primary target.
    "severe": "probSevere",
    # RT-PCR positive.
    "covid": "probCOVID",
}


def image_path(case_id: str) -> Path:
    return RAW_DIR / "data" / "mha" / f"{case_id}.mha"


def download() -> None:
    """Nothing to download here (AWS bucket, fetched on the host). Check the copy is complete."""
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"{REFERENCE_PATH} is missing: the STOIC download is unfinished or incomplete."
        )
    cases = load_reference()["PatientID"].astype(str)
    missing = [case for case in cases if not image_path(case).exists()]
    if len(cases) != NUM_CASES or missing:
        raise RuntimeError(f"STOIC: {len(cases)} labelled, {len(missing)} scans missing")
    logger.info(f"STOIC: {len(cases)} scans OK")


@cache
def load_reference() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_PATH)


def target_labels(reference: pd.DataFrame, target: str) -> pd.DataFrame:
    """``case_id``/``label`` for one target (single label column, no centre)."""
    return pd.DataFrame(
        {
            "case_id": reference["PatientID"].astype(str),
            "label": reference[TARGETS[target]].astype(int),
        }
    )


def totalseg_inputs(case_id: str, device: str) -> CaseInputs:
    mask = TOTALSEG_DIR / f"{case_id}.nii.gz"
    if not mask.exists():
        segment_rois(image_path(case_id), mask, list(LUNG_LOBES), device)
    return CaseInputs(
        image_path(case_id), mask, tuple(totalseg_class_ids(list(LUNG_LOBES)).values())
    )


def make_task(target: str) -> ClassificationTask:
    return ClassificationTask(
        name=f"stoic_{target}",
        source=SOURCE,
        description=f"STOIC 2021 chest CT, crop around the lungs, target: {target}",
        target=target,
        roi="lungs (TotalSegmentator, union of the 5 lobes)",
        modality="CT",
        download=download,
        load_labels=lambda: target_labels(load_reference(), target),
        case_inputs=totalseg_inputs,
        pad_value=CT_PAD_VALUE,
        # TotalSegmentator ROI, as for KiTS: fall back to the image centre if it fails.
        require_roi=False,
    )


STOIC_TASKS = tuple(make_task(target) for target in TARGETS)

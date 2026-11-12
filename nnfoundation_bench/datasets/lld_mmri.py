"""LLD-MMRI 2023 liver lesions (Lou et al.), MedSAM2 release with 3D lesion masks.

Source: Hugging Face dataset ``wanglab/LLD-MMRI-MedSAM2`` (pinned below), mirrored on the
NAS and linked as ``data/raw/LLD-MMRI2023/``. 498 patients, one lesion each, eight MR
series per patient (``images/<case>_<phase>_0000.nii.gz``) with a MedSAM2 lesion mask per
series (``labels/<case>_<phase>.nii.gz``, same grid, value 1). Series of one patient have
different grids (e.g. T2WI 8 mm slices, arterial 2.6 mm), so each phase is resampled with
its own mask. Intensities were rescaled to uint8 0-255 by the MedSAM2 authors.

``LLD_MMRI_Annotation.json`` holds the lesion category per patient (7 classes) and the
benign/malignant grouping. Case IDs are file prefixes (``MR-391135_1``, mostly without the
hyphen: ``MR177097_0``); the JSON is keyed by the part before ``_``. The suffix is the
lesion category (checked on all 498), so case IDs carry the label: never feed them to a model.
"""

import json
from functools import cache
from pathlib import Path

import pandas as pd
from loguru import logger

from nnfoundation_bench.config import RAW_DATA_DIR
from nnfoundation_bench.datasets.base import CaseInputs, ClassificationTask

SOURCE = "lld_mmri2023"
RAW_DIR = RAW_DATA_DIR / "LLD-MMRI2023"  # symlink to the NAS copy
HF_REPO = "wanglab/LLD-MMRI-MedSAM2"
HF_REVISION = "b7e8da56b267587689d8440e8298205f3fc4914e"
ANNOTATION_PATH = RAW_DIR / "LLD_MMRI_Annotation.json"
NUM_PATIENTS = 498

# File phase name -> (task suffix, description).
PHASES = {
    "C-pre": ("pre", "T1 non-contrast"),
    "C+A": ("arterial", "T1 arterial phase"),
    "C+V": ("venous", "T1 portal venous phase"),
    "C+Delay": ("delay", "T1 delayed phase"),
    "T2WI": ("t2", "T2-weighted"),
    "DWI": ("dwi", "diffusion-weighted"),
    "InPhase": ("inphase", "T1 in-phase"),
    "OutPhase": ("outphase", "T1 out-of-phase"),
}

MR_PAD_VALUE = 0.0  # black background of the 0-255 rescaled series


def image_path(case_id: str, phase: str) -> Path:
    return RAW_DIR / "images" / f"{case_id}_{phase}_0000.nii.gz"


def mask_path(case_id: str, phase: str) -> Path:
    return RAW_DIR / "labels" / f"{case_id}_{phase}.nii.gz"


def case_ids() -> list[str]:
    """``MR177097_0_C+A_0000.nii.gz`` -> ``MR177097_0``."""
    names = (p.name.split("_") for p in (RAW_DIR / "images").glob("MR*_0000.nii.gz"))
    return sorted({f"{parts[0]}_{parts[1]}" for parts in names})


def download() -> None:
    """Fetch the pinned HF snapshot (~18 GB) if needed, then check every series is there."""
    expected = NUM_PATIENTS * len(PHASES)
    if not (RAW_DIR / "images").is_dir() or len(list((RAW_DIR / "images").iterdir())) < expected:
        from huggingface_hub import snapshot_download

        logger.info(f"Downloading {HF_REPO}@{HF_REVISION[:8]} to {RAW_DIR}")
        snapshot_download(
            repo_id=HF_REPO, repo_type="dataset", revision=HF_REVISION, local_dir=RAW_DIR
        )
    cases = case_ids()
    missing = [
        path.name
        for case in cases
        for phase in PHASES
        for path in (image_path(case, phase), mask_path(case, phase))
        if not path.exists()
    ]
    if len(cases) != NUM_PATIENTS or missing:
        raise RuntimeError(f"LLD-MMRI: {len(cases)} cases, missing files: {missing[:5]}")
    logger.info(f"LLD-MMRI: {len(cases)} cases x {len(PHASES)} phases OK")


@cache
def load_annotation() -> dict:
    with open(ANNOTATION_PATH) as f:
        return json.load(f)


def malignancy_labels(annotation: dict, cases: list[str]) -> pd.DataFrame:
    """1 if the patient's lesion is malignant (HCC, ICC, metastasis), 0 if benign.

    ASSUMPTION: the paper lists LLD-MMRI2023 without its target. The challenge task is
    7-class; we use the dataset's own benign/malignant grouping (``Category_info``) to get
    a binary AUROC. Categories are read per series and must agree within a patient.
    """
    malignant = set(annotation["Category_info"]["Malignant"])
    benign = set(annotation["Category_info"]["Benign"])
    info = annotation["Annotation_info"]
    rows = []
    for case in cases:
        categories = {
            lesion["category"]
            for series in info[case.split("_")[0]]
            for lesion in series["annotation"]["lesion"].values()
        }
        if len(categories) != 1 or not categories <= malignant | benign:
            raise RuntimeError(f"{case}: lesion categories {sorted(categories)}")
        rows.append({"case_id": case, "label": int(categories.pop() in malignant)})
    return pd.DataFrame(rows, columns=["case_id", "label"])


def make_task(phase: str) -> ClassificationTask:
    suffix, series = PHASES[phase]
    return ClassificationTask(
        name=f"lld_mmri_malignant_{suffix}",
        source=SOURCE,
        description=f"LLD-MMRI liver MR, {series} ({phase}), crop around the lesion, "
        "target: malignant",
        target="malignant",
        roi="liver lesion (MedSAM2 mask shipped with the dataset)",
        modality="MR",
        download=download,
        load_labels=lambda: malignancy_labels(load_annotation(), case_ids()),
        case_inputs=lambda case_id, device: CaseInputs(
            image_path(case_id, phase), mask_path(case_id, phase)
        ),
        pad_value=MR_PAD_VALUE,
    )


LLD_MMRI_TASKS = tuple(make_task(phase) for phase in PHASES)

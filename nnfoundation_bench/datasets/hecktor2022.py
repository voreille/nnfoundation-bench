"""HECKTOR 2022 head-and-neck PET/CT with recurrence-free survival (challenge task 2).

Raw data: ``data/raw/HECKTOR2022/`` holds three symlinks to the NAS copy of the challenge
release (placed by hand, registration-gated):

- ``hecktor2022_training/hecktor2022/``: ``imagesTr/<id>__{CT,PT}.nii.gz`` and
  ``labelsTr/<id>.nii.gz`` for 524 patients (7 centres), plus
  ``hecktor2022_clinical_info_training.csv`` and ``hecktor2022_endpoint_training.csv``.
- ``hecktor2022_testing/``: ``imagesTs/`` (359 patients, CHB, MDA, USZ) and
  ``hecktor2022_clinical_info_testing.csv``.
- ``hecktor2022_testing_gt/``: ``labelsTs/`` and ``hecktor2022_endpoint_testing.csv``.

NIfTI already: CT in HU, PET in SUV on its own (coarser) grid, masks on the CT grid with
1 = GTVp, 2 = GTVn. The endpoint files give ``Relapse`` (1 = recurrence) and ``RFS`` (days)
for 488 + 339 patients. We pool the challenge's training and test sets (D34).
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from nnfoundation_bench.config import CT_PAD_VALUE, PET_PAD_VALUE, RAW_DATA_DIR
from nnfoundation_bench.datasets.base import CaseInputs, ClassificationTask

SOURCE = "hecktor2022"
RAW_DIR = RAW_DATA_DIR / "HECKTOR2022"  # symlinks to the NAS copy
COHORTS = {
    # cohort -> (images dir, labels dir, clinical csv, endpoint csv)
    "train": (
        RAW_DIR / "hecktor2022_training" / "hecktor2022" / "imagesTr",
        RAW_DIR / "hecktor2022_training" / "hecktor2022" / "labelsTr",
        RAW_DIR
        / "hecktor2022_training"
        / "hecktor2022"
        / "hecktor2022_clinical_info_training.csv",
        RAW_DIR / "hecktor2022_training" / "hecktor2022" / "hecktor2022_endpoint_training.csv",
    ),
    "test": (
        RAW_DIR / "hecktor2022_testing" / "imagesTs",
        RAW_DIR / "hecktor2022_testing_gt" / "labelsTs",
        RAW_DIR / "hecktor2022_testing" / "hecktor2022_clinical_info_testing.csv",
        RAW_DIR / "hecktor2022_testing_gt" / "hecktor2022_endpoint_testing.csv",
    ),
}
NUM_WITH_ENDPOINT = 488 + 339
GTVP = 1
HPV_COLUMN = "HPV status (0=-, 1=+)"
HORIZON_DAYS = 730  # 2 years
MODALITIES = {"PT": ("pet", "PET (SUV)", PET_PAD_VALUE), "CT": ("ct", "CT (HU)", CT_PAD_VALUE)}


def cohort_of(case_id: str) -> str:
    """Challenge cohort holding the case (``train`` or ``test``)."""
    for cohort, (images, *_) in COHORTS.items():
        if (images / f"{case_id}__CT.nii.gz").exists():
            return cohort
    raise FileNotFoundError(f"{case_id}: no image in HECKTOR 2022")


def image_path(case_id: str, modality: str) -> Path:
    return COHORTS[cohort_of(case_id)][0] / f"{case_id}__{modality}.nii.gz"


def mask_path(case_id: str) -> Path:
    return COHORTS[cohort_of(case_id)][1] / f"{case_id}.nii.gz"


def read_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Endpoint and clinical tables of both cohorts, with a ``cohort`` column."""
    endpoint = pd.concat(
        [pd.read_csv(e).assign(cohort=c) for c, (_, _, _, e) in COHORTS.items()],
        ignore_index=True,
    )
    clinical = pd.concat([pd.read_csv(t) for _, _, t, _ in COHORTS.values()], ignore_index=True)
    return endpoint, clinical


def download() -> None:
    """Nothing to download: the challenge release must be linked in data/raw by hand."""
    for paths in COHORTS.values():
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing (HECKTOR 2022 is placed by hand)")
    endpoint, _ = read_tables()
    missing = [
        f"{c} {m}"
        for c, cohort in zip(endpoint["PatientID"], endpoint["cohort"], strict=True)
        for m in ("CT", "PT", "mask")
        if not (
            COHORTS[cohort][1] / f"{c}.nii.gz"
            if m == "mask"
            else COHORTS[cohort][0] / f"{c}__{m}.nii.gz"
        ).exists()
    ]
    if len(endpoint) != NUM_WITH_ENDPOINT or missing:
        raise RuntimeError(f"HECKTOR 2022: {len(endpoint)} endpoints, missing: {missing[:5]}")
    logger.info(f"HECKTOR 2022: {len(endpoint)} patients with RFS, CT, PET and mask present")


def rfs_labels(
    endpoint: pd.DataFrame, clinical: pd.DataFrame, horizon: int = HORIZON_DAYS
) -> pd.DataFrame:
    """One row per patient with a recurrence-free survival endpoint.

    ``label`` = recurrence within ``horizon`` days (default 2 years): 1 if ``Relapse`` and
    ``RFS < horizon``, 0 if ``RFS >= horizon``. ASSUMPTION, our binary view so the AUROC
    probes apply; patients censored before the horizon without recurrence have no 2-year
    status and are dropped (also from the Cox probes, so both use one cohort). ``time``
    (days) and ``event`` (1 = recurrence) feed the Cox probes (D32). ``age`` and ``hpv``
    (NaN when unknown) are the clinical covariates. ``group`` = centre, ``stratum`` =
    centre x label; ``cohort`` = the challenge's train/test set (kept for reference only).
    """
    clin = clinical.drop_duplicates("PatientID").set_index("PatientID")
    time, event = endpoint["RFS"].astype(float), endpoint["Relapse"].astype(int)
    known = (event == 1) & (time < horizon) | (time >= horizon)
    if dropped := int((~known).sum()):
        logger.info(f"{dropped} patients censored before {horizon} days are excluded")
    e = endpoint[known]
    labels = pd.DataFrame(
        {
            "case_id": e["PatientID"].astype(str),
            "label": ((e["Relapse"] == 1) & (e["RFS"] < horizon)).astype(int),
            "time": e["RFS"].astype(float),
            "event": e["Relapse"].astype(int),
            "age": pd.to_numeric(e["PatientID"].map(clin["Age"]), errors="coerce"),
            "hpv": pd.to_numeric(e["PatientID"].map(clin[HPV_COLUMN]), errors="coerce"),
            "cohort": e["cohort"],
        }
    )
    labels["group"] = labels["case_id"].str.split("-").str[0]  # centre
    labels["stratum"] = labels["group"] + "_" + labels["label"].astype(str)
    return labels.sort_values("case_id").reset_index(drop=True)


def load_labels() -> pd.DataFrame:
    return rfs_labels(*read_tables())


def make_task(modality: str) -> ClassificationTask:
    suffix, description, pad_value = MODALITIES[modality]
    return ClassificationTask(
        name=f"hecktor22_rfs_{suffix}",
        source=SOURCE,
        description=f"HECKTOR 2022 {description}, crop around GTVp, target: recurrence within "
        "2 years (time/event kept for C-index)",
        target="recurrence_2y",
        roi="GTVp",
        modality=modality,
        download=download,
        load_labels=load_labels,
        case_inputs=lambda case_id, device: CaseInputs(
            image_path(case_id, modality), mask_path(case_id), GTVP
        ),
        pad_value=pad_value,
    )


HECKTOR22_RFS_TASKS = tuple(make_task(modality) for modality in MODALITIES)

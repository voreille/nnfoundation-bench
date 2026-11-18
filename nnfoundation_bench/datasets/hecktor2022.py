"""HECKTOR 2022 head-and-neck PET/CT (MICCAI challenge), training and test sets pooled.

Placed by hand in ``data/raw/HECKTOR2022/`` (links to the NAS copy), as released by the
organisers: NIfTI, already converted.

- ``hecktor2022_training/hecktor2022/``: ``imagesTr/<case>__{CT,PT}.nii.gz``,
  ``labelsTr/<case>.nii.gz``, ``hecktor2022_endpoint_training.csv`` (524 patients, 7
  centres; 488 with an endpoint).
- ``hecktor2022_testing/imagesTs/`` and ``hecktor2022_testing_gt/`` (``labelsTs/``,
  ``hecktor2022_endpoint_testing.csv``): 359 patients, 3 centres; 339 with an endpoint.
  The test labels were released after the challenge.

CT is in HU, PET in SUV (checked: tumour SUVmax 10-19), both float32 with an identity
direction, on different grids in the same physical space. Masks are on the CT grid
(48 training masks have one extra row or slice, same origin and spacing): 1 = GTVp,
2 = GTVn. Every case has at least one of them, and their union fits in 192 mm in every
case (largest extent 179 mm), so one crop centred on the union holds all tumours, and the
PET and CT crops of a patient share that centre.

The endpoint is recurrence-free survival (challenge Task 2): ``RFS`` in days and
``Relapse`` (1 = event, 0 = censored). Patients without an endpoint are left out.
"""

from functools import cache
from pathlib import Path

import pandas as pd
from loguru import logger

from nnfoundation_bench.config import CT_PAD_VALUE, PET_PAD_VALUE, RAW_DATA_DIR
from nnfoundation_bench.datasets.base import CaseInputs, ClassificationTask

SOURCE = "hecktor2022"
RAW_DIR = RAW_DATA_DIR / "HECKTOR2022"  # links to the NAS copy
TRAIN_DIR = RAW_DIR / "hecktor2022_training" / "hecktor2022"
TEST_DIR = RAW_DIR / "hecktor2022_testing"
TEST_GT_DIR = RAW_DIR / "hecktor2022_testing_gt"

# Official split -> (images, masks, endpoint CSV).
COHORTS = {
    "train": (
        TRAIN_DIR / "imagesTr",
        TRAIN_DIR / "labelsTr",
        TRAIN_DIR / "hecktor2022_endpoint_training.csv",
    ),
    "test": (
        TEST_DIR / "imagesTs",
        TEST_GT_DIR / "labelsTs",
        TEST_GT_DIR / "hecktor2022_endpoint_testing.csv",
    ),
}
NUM_WITH_ENDPOINT = {"train": 488, "test": 339}
GTVP, GTVN = 1, 2


@cache
def load_endpoints() -> pd.DataFrame:
    """``PatientID, Relapse, RFS, official_split`` for both cohorts."""
    return pd.concat(
        [pd.read_csv(csv).assign(official_split=split) for split, (_, _, csv) in COHORTS.items()],
        ignore_index=True,
    )


def endpoint_labels(endpoints: pd.DataFrame) -> pd.DataFrame:
    """One row per patient with an endpoint.

    ``label`` = ``event`` = ``Relapse`` (used to stratify the splits), ``time`` = ``RFS``
    in days. ``group`` is the centre (the ID prefix), ``stratum`` centre x event.
    """
    if endpoints["PatientID"].duplicated().any():
        raise RuntimeError("HECKTOR 2022: a patient is listed twice across cohorts")
    event = endpoints["Relapse"].astype(int)
    group = endpoints["PatientID"].str.split("-").str[0]
    return pd.DataFrame(
        {
            "case_id": endpoints["PatientID"].astype(str),
            "label": event,
            "time": endpoints["RFS"].astype(float),
            "event": event,
            "group": group,
            "stratum": group + "_" + event.astype(str),
            "official_split": endpoints["official_split"],
        }
    )


@cache
def case_cohorts() -> dict[str, str]:
    endpoints = load_endpoints()
    return dict(zip(endpoints["PatientID"], endpoints["official_split"], strict=True))


def image_path(case_id: str, modality: str) -> Path:
    return COHORTS[case_cohorts()[case_id]][0] / f"{case_id}__{modality}.nii.gz"


def mask_path(case_id: str) -> Path:
    return COHORTS[case_cohorts()[case_id]][1] / f"{case_id}.nii.gz"


def download() -> None:
    """Nothing to download (registration-gated, placed by hand). Check the copy is complete."""
    endpoints = load_endpoints()
    counts = endpoints["official_split"].value_counts().to_dict()
    if counts != NUM_WITH_ENDPOINT:
        raise RuntimeError(
            f"HECKTOR 2022: endpoints per cohort {counts}, expected {NUM_WITH_ENDPOINT}"
        )
    missing = [
        path.name
        for case in endpoints["PatientID"]
        for path in (image_path(case, "CT"), image_path(case, "PT"), mask_path(case))
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(f"HECKTOR 2022: {len(missing)} files missing, e.g. {missing[:3]}")
    logger.info(f"HECKTOR 2022: {len(endpoints)} patients with an endpoint, all files present")


def make_task(modality: str) -> ClassificationTask:
    ct = modality == "CT"
    return ClassificationTask(
        name=f"hecktor2022_rfs_{'ct' if ct else 'pet'}",
        source=SOURCE,
        description=f"HECKTOR 2022 train + test, {'CT' if ct else 'PET (SUV)'}, crop around "
        "GTVp + GTVn, target: recurrence-free survival",
        target="relapse",
        roi="GTVp (1) and GTVn (2), dataset masks; crop centred on their union",
        modality="CT" if ct else "PT",
        download=download,
        load_labels=lambda: endpoint_labels(load_endpoints()),
        case_inputs=lambda case_id, device: CaseInputs(
            image_path(case_id, modality), mask_path(case_id), (GTVP, GTVN)
        ),
        pad_value=CT_PAD_VALUE if ct else PET_PAD_VALUE,
        keep_roi_labels=True,
        roi_names=((GTVP, "gtvp"), (GTVN, "gtvn")),
        # CT padding goes down to -16989 HU; clamp to air as for RSNA (D19).
        intensity_min=CT_PAD_VALUE if ct else None,
    )


HECKTOR2022_TASKS = (make_task("CT"), make_task("PT"))

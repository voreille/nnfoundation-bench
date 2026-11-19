"""BraTS 2024 post-treatment glioma (BraTS-GLI / BraTS-PTG), glioma type per patient.

Source: the BraTS 2024 adult glioma challenge on Synapse (registration-gated), placed by
hand on the NAS and linked as ``data/raw/BraTS_Glioma_synapse/``. Three folders:
``training_data1_v2`` (1350 scans), ``training_data_additional`` (271) and
``validation_data`` (188, **no segmentation**). Each scan folder
``BraTS-GLI-<patient>-<timepoint>/`` holds ``<scan>-{t1c,t1n,t2f,t2w,seg}.nii.gz``, all on
one grid: 182 x 218 x 182 at 1 mm, co-registered and skull-stripped (background 0). The
seg is float32 in one training folder and float64 in the other. Seg labels: 1 non-enhancing
tumour core, 2 surrounding FLAIR hyperintensity (edema), 3 enhancing tumour, 4 resection
cavity. The scans are post-treatment; patients have 1-10 timepoints (``-100``, ``-101``...).

``BraTS-PTG supplementary demographic information and metadata.xlsx`` (sheet
``Individual Data``) has one row per scan, including the validation scans, with the
``Site`` (5 centres) and the ``Glioma type``.
"""

from functools import cache
from pathlib import Path

import pandas as pd
from loguru import logger

from nnfoundation_bench.config import RAW_DATA_DIR
from nnfoundation_bench.datasets.base import CaseInputs, ClassificationTask

SOURCE = "brats2024_gli"
RAW_DIR = RAW_DATA_DIR / "BraTS_Glioma_synapse"  # symlink to the NAS copy
METADATA_PATH = RAW_DIR / "BraTS-PTG supplementary demographic information and metadata.xlsx"
METADATA_SHEET = "Individual Data"
SCAN_FOLDERS = ("training_data1_v2", "training_data_additional", "validation_data")
NUM_SCANS = 1809

# File suffix -> description. One single-channel task per sequence (D14).
SEQUENCES = {
    "t1c": "T1 post-contrast",
    "t1n": "T1 native",
    "t2f": "T2-FLAIR",
    "t2w": "T2-weighted",
}
WHOLE_TUMOUR = (1, 2, 3)  # NETC + SNFH + ET; the resection cavity (4) is left out
MR_PAD_VALUE = 0.0  # skull-stripped background

# Glioma type (lower case, stripped) -> label. Other types are excluded.
GLIOMA_LABELS = {"glioblastoma": 1, "astrocytoma": 0, "oligodendroglioma": 0}


@cache
def scan_dirs() -> dict[str, Path]:
    """Scan ID -> folder, for scans that have a segmentation (the ROI)."""
    return {
        seg.name.removesuffix("-seg.nii.gz"): seg.parent
        for folder in SCAN_FOLDERS
        for seg in (RAW_DIR / folder).glob("BraTS-GLI-*/BraTS-GLI-*-seg.nii.gz")
    }


def image_path(case_id: str, sequence: str) -> Path:
    return scan_dirs()[case_id] / f"{case_id}-{sequence}.nii.gz"


def mask_path(case_id: str) -> Path:
    return scan_dirs()[case_id] / f"{case_id}-seg.nii.gz"


def split_scan_id(scan_id: str) -> tuple[str, int]:
    """``BraTS-GLI-00005-101`` -> (``BraTS-GLI-00005``, 101)."""
    patient, timepoint = scan_id.rsplit("-", 1)
    return patient, int(timepoint)


def download() -> None:
    """Nothing to download: the Synapse release must be placed in data/raw by hand."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"{METADATA_PATH} is missing. BraTS is registration-gated (Synapse); unpack the "
            f"2024 GLI training, additional training and validation data into {RAW_DIR}."
        )
    scans = [d for folder in SCAN_FOLDERS for d in (RAW_DIR / folder).glob("BraTS-GLI-*")]
    missing = [
        f"{d.name}-{seq}"
        for d in scans
        for seq in SEQUENCES
        if not (d / f"{d.name}-{seq}.nii.gz").exists()
    ]
    if len(scans) != NUM_SCANS or missing:
        raise RuntimeError(f"BraTS: {len(scans)} scans, missing files: {missing[:5]}")
    logger.info(f"BraTS: {len(scans)} scans, {len(scan_dirs())} with a segmentation")


@cache
def load_metadata() -> pd.DataFrame:
    table = pd.read_excel(METADATA_PATH, sheet_name=METADATA_SHEET)
    table.columns = table.columns.str.strip()  # e.g. "Glioma type " has a trailing space
    return table


def glioma_labels(metadata: pd.DataFrame, cases: list[str]) -> pd.DataFrame:
    """Glioblastoma (1) vs astrocytoma or oligodendroglioma (0), one scan per patient.

    ASSUMPTION: the paper does not use BraTS for classification; this target is ours.
    Under WHO 2021 it is close to IDH-wildtype vs IDH-mutant. "Glioma NOS" (type not
    determined, almost all from UCSF) and "Other" are excluded. Spelling is normalised
    ("glioma NOS" / "Glioma NOS").

    ``cases`` are the scans with a segmentation. Each patient keeps its earliest one, so
    no patient is in two splits and the shared split code applies unchanged.
    ``group`` = site, ``stratum`` = site x label: GBM share differs a lot by site
    (Indiana: GBM only; UCSF: 80 of 297). ``age`` is a confounder reference, not a feature.
    """
    meta = metadata.set_index("BraTS Subject ID")
    if missing := sorted(set(cases) - set(meta.index)):
        raise RuntimeError(f"{len(missing)} scans not in the metadata, e.g. {missing[:3]}")
    earliest = {}
    for case in sorted(cases, key=split_scan_id):
        earliest.setdefault(split_scan_id(case)[0], case)
    chosen = sorted(earliest.values())
    types = meta.loc[chosen, "Glioma type"].str.strip().str.lower()
    labels = pd.DataFrame(
        {
            "case_id": chosen,
            "label": types.map(GLIOMA_LABELS).to_numpy(),
            "group": meta.loc[chosen, "Site"].to_numpy(),
            # Age at the chosen scan (NaN if unknown), for the age-only reference (D29).
            "age": pd.to_numeric(meta.loc[chosen, "Patient's Age"]).to_numpy(),
        }
    )
    if dropped := types[labels["label"].isna().to_numpy()].value_counts().to_dict():
        logger.info(f"Excluded glioma types (patients): {dropped}")
    labels = labels.dropna(subset=["label"]).reset_index(drop=True)
    labels["label"] = labels["label"].astype(int)
    return labels.assign(stratum=labels["group"] + "_" + labels["label"].astype(str))


def make_task(sequence: str) -> ClassificationTask:
    return ClassificationTask(
        name=f"brats_gbm_{sequence}",
        source=SOURCE,
        description=f"BraTS 2024 post-treatment glioma MR, {SEQUENCES[sequence]} "
        f"({sequence}), crop around the whole tumour, target: glioblastoma",
        target="glioblastoma",
        roi="whole tumour (dataset seg labels 1+2+3)",
        modality="MR",
        download=download,
        load_labels=lambda: glioma_labels(load_metadata(), list(scan_dirs())),
        case_inputs=lambda case_id, device: CaseInputs(
            image_path(case_id, sequence), mask_path(case_id), WHOLE_TUMOUR
        ),
        pad_value=MR_PAD_VALUE,
    )


BRATS_TASKS = tuple(make_task(sequence) for sequence in SEQUENCES)

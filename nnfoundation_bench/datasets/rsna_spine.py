"""RSNA 2022 Cervical Spine Fracture Detection (Kaggle), training set.

The Kaggle archive was unpacked by hand in ``data/raw/RSNA-Spine/`` (symlink to the NAS
copy). Used:

- ``train.csv``: ``StudyInstanceUID,patient_overall,C1..C7`` (2019 studies);
- ``train_images/<StudyInstanceUID>/<n>.dcm``: one axial CT series per study.

Each study is converted to an int16 NIfTI in HU with SimpleITK (GDCM sorts the slices and
decodes the JPEG-lossless files) and cached under ``data/interim/rsna_spine/nifti/``.
There are no ROI masks for most studies (``segmentations/`` covers 87), so the crop is
centred on the cervical spine (union of the TotalSegmentator vertebrae C1-C7), cached
under ``data/interim/rsna_spine/totalseg/``.
"""

from functools import cache
from pathlib import Path

import pandas as pd
import SimpleITK as sitk
from loguru import logger

from nnfoundation_bench.config import CT_PAD_VALUE, INTERIM_DATA_DIR, RAW_DATA_DIR
from nnfoundation_bench.datasets.base import CaseInputs, ClassificationTask
from nnfoundation_bench.roi import segment_rois, totalseg_class_ids

SOURCE = "rsna_spine"
RAW_DIR = RAW_DATA_DIR / "RSNA-Spine"  # symlink to the NAS copy
TRAIN_CSV = RAW_DIR / "train.csv"
IMAGES_DIR = RAW_DIR / "train_images"
INTERIM_DIR = INTERIM_DATA_DIR / SOURCE
NIFTI_DIR = INTERIM_DIR / "nifti"
TOTALSEG_DIR = INTERIM_DIR / "totalseg"
NUM_STUDIES = 2019

VERTEBRAE = tuple(f"vertebrae_C{i}" for i in range(1, 8))


def download() -> None:
    """Nothing to download here (Kaggle, fetched and unpacked on the host). Check the copy.

    Listing 2019 folders on the NAS takes ~5 min; a marker under data/interim skips it later.
    """
    marker = INTERIM_DIR / "raw_verified"
    if marker.exists():
        return
    if not TRAIN_CSV.exists() or not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"{RAW_DIR}: train.csv or train_images/ missing (unpacked?)")
    labels = load_train_csv()
    missing = [
        uid
        for uid in labels["StudyInstanceUID"]
        if not (IMAGES_DIR / uid).is_dir() or not any((IMAGES_DIR / uid).glob("*.dcm"))
    ]
    if len(labels) != NUM_STUDIES or missing:
        raise RuntimeError(f"RSNA Spine: {len(labels)} labelled, {len(missing)} without images")
    logger.info(f"RSNA Spine: {len(labels)} studies OK")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


@cache
def load_train_csv() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV)


def fracture_labels(table: pd.DataFrame) -> pd.DataFrame:
    """1 if the study has a fracture at any cervical level (``patient_overall``).

    ASSUMPTION: the paper lists RSNA Spine (2019 CT, development) without its target;
    ``patient_overall`` is the competition's study-level label.
    """
    return pd.DataFrame(
        {
            "case_id": table["StudyInstanceUID"].astype(str),
            "label": table["patient_overall"].astype(int),
        }
    )


def read_dicom_series(folder: Path) -> sitk.Image:
    """Read the largest DICOM series in ``folder`` (slices sorted by GDCM)."""
    reader = sitk.ImageSeriesReader()
    series = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder))
    if not series:
        raise RuntimeError(f"No DICOM series in {folder}")
    files = max(
        (sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), s) for s in series), key=len
    )
    reader.SetFileNames(files)
    return reader.Execute()


def nifti_path(case_id: str) -> Path:
    """Convert one study to NIfTI in HU, clamped below at air (cached).

    About 22% of each scan lies outside the reconstruction circle and is stored as
    -2048, -3024 or -3669 depending on the scanner. Left in, it drags the whole-scan
    z-score (e.g. mean -46 -> -688 HU, std 389 -> 1273) in a scanner-dependent way, so it
    is set to ``CT_PAD_VALUE`` (-1024), the value used outside the field of view (D19).
    """
    out = NIFTI_DIR / f"{case_id}.nii.gz"
    if not out.exists():
        image = sitk.Cast(read_dicom_series(IMAGES_DIR / case_id), sitk.sitkInt16)
        image = sitk.Clamp(image, lowerBound=CT_PAD_VALUE, upperBound=32767)
        NIFTI_DIR.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".partial.{out.name}")
        sitk.WriteImage(image, tmp, True)
        tmp.rename(out)
    return out


def totalseg_inputs(case_id: str, device: str) -> CaseInputs:
    image = nifti_path(case_id)
    mask = TOTALSEG_DIR / f"{case_id}.nii.gz"
    if not mask.exists():
        segment_rois(image, mask, list(VERTEBRAE), device)
    return CaseInputs(image, mask, tuple(totalseg_class_ids(list(VERTEBRAE)).values()))


RSNA_SPINE_FRACTURE = ClassificationTask(
    name="rsna_spine_fracture",
    source=SOURCE,
    description="RSNA 2022 cervical spine CT, crop around C1-C7, target: any fracture",
    target="fracture",
    roi="cervical spine C1-C7 (TotalSegmentator)",
    modality="CT",
    download=download,
    load_labels=lambda: fracture_labels(load_train_csv()),
    case_inputs=totalseg_inputs,
    pad_value=CT_PAD_VALUE,
    # TotalSegmentator ROI, as for KiTS: fall back to the image centre if it fails.
    require_roi=False,
)

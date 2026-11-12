"""RSNA 2022 Cervical Spine Fracture Detection (Kaggle), training set.

The Kaggle competition archive is placed in ``data/raw/RSNA-Spine/`` (symlink to the NAS
copy) and is **not unpacked**: it is ~190 GB of DICOM. Members used:

- ``train.csv``: ``StudyInstanceUID,patient_overall,C1..C7`` (2019 studies);
- ``train_images/<StudyInstanceUID>/<n>.dcm``: one axial CT series per study.

Each study is read straight from the zip into a temporary folder, converted to an int16
NIfTI in HU with SimpleITK (GDCM sorts the slices and decodes the JPEG-lossless files) and
cached under ``data/interim/rsna_spine/nifti/``. There are no ROI masks for most studies
(``segmentations/`` covers 87), so the crop is centred on the cervical spine (union of the
TotalSegmentator vertebrae C1-C7), cached under ``data/interim/rsna_spine/totalseg/``.
"""

import io
import json
import shutil
import zipfile
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
ZIP_PATH = RAW_DIR / "rsna-2022-cervical-spine-fracture-detection.zip"
INTERIM_DIR = INTERIM_DATA_DIR / SOURCE
NIFTI_DIR = INTERIM_DIR / "nifti"
TOTALSEG_DIR = INTERIM_DIR / "totalseg"
INDEX_PATH = INTERIM_DIR / "zip_index.json"
NUM_STUDIES = 2019

VERTEBRAE = tuple(f"vertebrae_C{i}" for i in range(1, 8))


def download() -> None:
    """Nothing to download here (Kaggle, fetched on the host). Check the archive is complete."""
    partial = ZIP_PATH.with_name(ZIP_PATH.name + ".kaggle-partial")
    if partial.exists() or not ZIP_PATH.exists():
        raise FileNotFoundError(f"{ZIP_PATH}: the Kaggle download is unfinished.")
    if not zipfile.is_zipfile(ZIP_PATH):
        raise RuntimeError(f"{ZIP_PATH} is not a readable zip (truncated download?)")
    studies = zip_index()
    labels = load_train_csv()
    missing = sorted(set(labels["StudyInstanceUID"]) - set(studies))
    if len(labels) != NUM_STUDIES or missing:
        raise RuntimeError(f"RSNA Spine: {len(labels)} labelled, {len(missing)} without images")
    logger.info(f"RSNA Spine: {len(labels)} studies OK")


def build_zip_index(archive: zipfile.ZipFile) -> dict[str, list[str]]:
    """StudyInstanceUID -> its ``train_images/<uid>/*.dcm`` members."""
    index: dict[str, list[str]] = {}
    for name in archive.namelist():
        parts = name.split("/")
        if len(parts) == 3 and parts[0] == "train_images" and parts[2].endswith(".dcm"):
            index.setdefault(parts[1], []).append(name)
    return index


@cache
def zip_index() -> dict[str, list[str]]:
    """Cached on disk: listing the ~700k members of the archive takes a while."""
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text())
    with zipfile.ZipFile(ZIP_PATH) as archive:
        index = build_zip_index(archive)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index))
    return index


@cache
def load_train_csv() -> pd.DataFrame:
    with zipfile.ZipFile(ZIP_PATH) as archive:
        return pd.read_csv(io.BytesIO(archive.read("train.csv")))


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
    """Convert one study from the zip to NIfTI (cached)."""
    out = NIFTI_DIR / f"{case_id}.nii.gz"
    if out.exists():
        return out
    tmp_dir = INTERIM_DIR / f".partial.{case_id}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    try:
        with zipfile.ZipFile(ZIP_PATH) as archive:
            for member in zip_index()[case_id]:
                (tmp_dir / Path(member).name).write_bytes(archive.read(member))
        image = sitk.Cast(read_dicom_series(tmp_dir), sitk.sitkInt16)
        NIFTI_DIR.mkdir(parents=True, exist_ok=True)
        tmp_out = out.with_name(f".partial.{out.name}")
        sitk.WriteImage(image, tmp_out, True)
        tmp_out.rename(out)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
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

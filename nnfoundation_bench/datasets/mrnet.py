"""MRNet knee MRI (Bien et al., PLoS Med 2018), public release from Stanford AIMI (Redivis).

The data is registration-gated and is placed by hand in ``data/raw/MRNet/`` (symlink to
the NAS copy). Each exam has three series, stored separately as uint8 ``.npy`` arrays of
shape (slices, 256, 256): sagittal T2 (fat-sat), coronal T1 and axial PD. The arrays were
resized to 256x256 and intensity-standardised by the MRNet authors. They carry **no
geometry** (spacing, origin, direction), so we attach an assumed geometry (see
``PLANE_GEOMETRY``) and cache the result as NIfTI under ``data/interim/mrnet/nifti/``.

Public release: 1130 train (``0000``-``1129``) + 120 valid (``1130``-``1249``) exams. The
120-exam MRNet test set is held back by Stanford. We pool the 1250 public exams and apply
our 40/10/50 protocol. Labels per exam: ``abnormal``, ``acl`` (tear) and ``meniscus``
(tear), not mutually exclusive.

There are no ROI masks: the knee fills the field of view, and the whole series fits in a
192^3 crop at 1 mm. The ROI is therefore the series' field of view, so the exported
mask marks real data vs padding.
"""

import base64
import hashlib
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from loguru import logger

from nnfoundation_bench.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from nnfoundation_bench.datasets.base import CaseInputs, ClassificationTask

SOURCE = "mrnet"
RAW_DIR = RAW_DATA_DIR / "MRNet"  # symlink to the NAS copy
NIFTI_DIR = INTERIM_DATA_DIR / SOURCE / "nifti"
SPLITS = ("train", "valid")
PLANES = ("sagittal", "coronal", "axial")
TARGETS = ("abnormal", "acl", "meniscus")
NUM_EXAMS = {"train": 1130, "valid": 120}

# Intensities are standardised to 0-255 with a black (0) background.
MR_PAD_VALUE = 0.0

# ASSUMPTION: MRNet ships no geometry and its acquisition table (S1 Table) is not public.
# In-plane: the knee spans ~200/256 px in the axial series, i.e. ~0.6 mm/px for a typical
# 160 mm knee FOV -> 0.625 mm. Through-plane: ~30 slices cover the knee -> ~4 mm.
# Axis mapping (array (slice, row, col) -> patient LPS) was read off rendered slices: rows
# run superior -> inferior (sagittal, coronal) or anterior -> posterior (axial), sagittal
# columns run anterior -> posterior, axial slices go superior -> inferior. Left/right signs
# are unknown (left and right knees are mixed anyway). After ``crop_around`` reorients to
# identity, every plane ends up as (z, y, x) = (S-I, A-P, L-R) like the pretraining data.
IN_PLANE_SPACING_MM = 0.625
SLICE_SPACING_MM = 4.0
# Direction cosines (row-major 3x3) for SimpleITK axes (x=col, y=row, z=slice).
# Columns of the matrix are the physical (LPS) directions of the x, y and z axes.
PLANE_DIRECTION = {
    # col -> +P, row -> -S, slice -> +L
    "sagittal": (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
    # col -> +L, row -> -S, slice -> +P
    "coronal": (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0),
    # col -> +L, row -> +P, slice -> -S
    "axial": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
}


def exam_split(case_id: str) -> str:
    return "train" if int(case_id) < NUM_EXAMS["train"] else "valid"


def npy_path(case_id: str, plane: str) -> Path:
    return RAW_DIR / exam_split(case_id) / plane / f"{case_id}.npy"


def _md5_b64(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode()


def verify(check_md5: bool = True) -> None:
    """Check every file listed in the Redivis index CSVs (size, and md5 if asked)."""
    for split in SPLITS:
        index = pd.read_csv(RAW_DIR / f"redivis-index-{split}.csv")
        bad = []
        for row in index.itertuples():
            path = RAW_DIR / split / row.file_name
            size_ok = path.exists() and path.stat().st_size == row.size
            if not size_ok or (check_md5 and _md5_b64(path) != row.md5_hash):
                bad.append(row.file_name)
        if bad:
            raise RuntimeError(f"MRNet {split}: {len(bad)} missing or corrupt files: {bad[:5]}")
        if len(index) != NUM_EXAMS[split] * len(PLANES):
            raise RuntimeError(f"MRNet {split}: index lists {len(index)} files")
        logger.info(f"MRNet {split}: {len(index)} files OK")


def download() -> None:
    """Nothing to download: MRNet is registration-gated. Verify the hand-placed copy.

    The md5 check reads the whole 7 GB once; a marker under data/interim skips it later.
    """
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(
            f"{RAW_DIR} is missing. MRNet is registration-gated (Stanford AIMI / Redivis); "
            "place the train/ and valid/ folders and the redivis-*.csv files there."
        )
    marker = INTERIM_DATA_DIR / SOURCE / "raw_verified"
    verify(check_md5=not marker.exists())
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


def read_label_csv(path: Path) -> pd.DataFrame:
    """Read a Redivis label export: ``case_id`` (4-digit str), ``label`` (int).

    The Redivis export has no header row and prefixes values with ``_``. Read with a
    header, the first exam ends up as the column names (``_0000,_1``), so read with
    ``header=None``.
    """
    table = pd.read_csv(path, header=None, names=["case_id", "label"], dtype=str)
    table = table.apply(lambda col: col.str.strip().str.lstrip("_"))
    return table.assign(label=table["label"].astype(int))


@cache
def load_all_labels() -> pd.DataFrame:
    """One row per public exam, one column per target."""
    tables = []
    for split in SPLITS:
        per_target = [
            read_label_csv(RAW_DIR / f"redivis-{split}-{target}.csv")
            .set_index("case_id")["label"]
            .rename(target)
            for target in TARGETS
        ]
        tables.append(pd.concat(per_target, axis=1, join="outer"))
    labels = pd.concat(tables).sort_index()
    if labels.isna().any().any() or not labels.index.is_unique:
        raise RuntimeError("MRNet label files disagree on the exam list")
    return labels.astype(int).rename_axis("case_id").reset_index()


def target_labels(labels: pd.DataFrame, target: str) -> pd.DataFrame:
    """``case_id``/``label`` for one target (single centre: no ``group``/``stratum``)."""
    return labels[["case_id", target]].rename(columns={target: "label"})


def npy_to_image(array: np.ndarray, plane: str) -> sitk.Image:
    """Wrap a (slice, row, col) array in the assumed MRNet geometry, centred on the origin."""
    image = sitk.GetImageFromArray(array.astype(np.float32))
    image.SetSpacing((IN_PLANE_SPACING_MM, IN_PLANE_SPACING_MM, SLICE_SPACING_MM))
    image.SetDirection(PLANE_DIRECTION[plane])
    center = np.array(image.TransformContinuousIndexToPhysicalPoint(
        ((np.array(image.GetSize()) - 1) / 2).tolist()
    ))  # fmt: skip
    image.SetOrigin((np.array(image.GetOrigin()) - center).tolist())
    return image


def nifti_inputs(case_id: str, plane: str) -> CaseInputs:
    """Convert one series to NIfTI plus a field-of-view mask (cached)."""
    out_dir = NIFTI_DIR / plane
    image_path = out_dir / f"{case_id}.nii.gz"
    fov_path = out_dir / f"{case_id}_fov.nii.gz"
    if not (image_path.exists() and fov_path.exists()):
        out_dir.mkdir(parents=True, exist_ok=True)
        image = npy_to_image(np.load(npy_path(case_id, plane), allow_pickle=False), plane)
        fov = sitk.Image(image.GetSize(), sitk.sitkUInt8) + 1
        fov.CopyInformation(image)
        for img, path in ((fov, fov_path), (image, image_path)):
            tmp = path.with_name(f".partial.{path.name}")
            sitk.WriteImage(img, tmp, True)
            tmp.rename(path)
    return CaseInputs(image_path, fov_path)


SERIES = {"sagittal": "T2 FS", "coronal": "T1", "axial": "PD FS"}


def make_task(target: str, plane: str) -> ClassificationTask:
    return ClassificationTask(
        name=f"mrnet_{target}_{plane}",
        source=SOURCE,
        description=f"MRNet knee MR, {plane} {SERIES[plane]} series, target: {target}",
        target=target,
        roi="field of view (no ROI masks; whole series)",
        modality="MR",
        download=download,
        load_labels=lambda: target_labels(load_all_labels(), target),
        case_inputs=lambda case_id, device: nifti_inputs(case_id, plane),
        pad_value=MR_PAD_VALUE,
    )


MRNET_TASKS = tuple(make_task(target, plane) for target in TARGETS for plane in PLANES)

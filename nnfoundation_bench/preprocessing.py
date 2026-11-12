"""Shared preprocessing: ROI-centred 1 mm / 192^3 crops exported for glovita.

Output follows glovita's ``generic_3d_dataset`` format (see
https://github.com/MIC-DKFZ/image_classification/blob/master/docs/nnfoundation.md).
"""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from nnfoundation_bench.config import (
    CROP_SIZE,
    CT_PAD_VALUE,
    DEFAULT_SEED,
    INTERIM_DATA_DIR,
    N_SPLITS,
    PROCESSED_DATA_DIR,
    SPLIT_FRACTIONS,
    TARGET_SPACING_MM,
)
from nnfoundation_bench.datasets.base import ClassificationTask
from nnfoundation_bench.roi import segment_rois, totalseg_class_ids

Point = tuple[float, float, float]
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def roi_center(mask: sitk.Image, label: int) -> Point | None:
    """Physical centre of the bounding box of ``label`` in ``mask`` (None if absent)."""
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.Cast(mask, sitk.sitkUInt8))
    if label not in stats.GetLabels():
        return None
    bbox = stats.GetBoundingBox(label)  # (x, y, z, size_x, size_y, size_z) in voxels
    start, size = np.array(bbox[:3]), np.array(bbox[3:])
    return mask.TransformContinuousIndexToPhysicalPoint((start + (size - 1) / 2).tolist())


def image_center(image: sitk.Image) -> Point:
    return image.TransformContinuousIndexToPhysicalPoint(
        ((np.array(image.GetSize()) - 1) / 2).tolist()
    )


def crop_around(
    image: sitk.Image,
    center: Point,
    size: Sequence[int] = CROP_SIZE,
    spacing: Sequence[float] = TARGET_SPACING_MM,
    interpolator: int = sitk.sitkLinear,
    pad_value: float = CT_PAD_VALUE,
    direction: Sequence[float] = IDENTITY,
) -> sitk.Image:
    """Resample ``image`` onto a ``size`` grid of ``spacing`` centred on ``center``.

    Cropping, resampling and reorientation happen in one step in physical space.
    The default identity ``direction`` gives every case the same (LPS) axis order,
    so arrays are (z, y, x) = (axial, coronal, sagittal) regardless of how the source
    stored its axes. Regions outside the field of view get ``pad_value``.
    """
    direction_matrix = np.array(direction).reshape(3, 3)
    half_extent = (np.array(size) - 1) / 2 * np.array(spacing)
    origin = np.array(center) - direction_matrix @ half_extent
    reference = sitk.Image([int(s) for s in size], image.GetPixelID())
    reference.SetSpacing([float(s) for s in spacing])
    reference.SetDirection([float(d) for d in direction])
    reference.SetOrigin(origin.tolist())
    return sitk.Resample(
        image, reference, sitk.Transform(), interpolator, pad_value, image.GetPixelID()
    )


def zscore(array: np.ndarray, mean: float, std: float) -> np.ndarray:
    """nnU-Net ``ZScoreNormalization`` (no mask), as used for nnFoundation pretraining."""
    return ((array.astype(np.float32) - mean) / max(std, 1e-8)).astype(np.float32)


def write_b2nd(array: np.ndarray, path: Path) -> None:
    """Write a channel-first float32 volume in glovita's blosc2 layout."""
    from glovita.datasets.blosc2io import Blosc2IO

    blocks, chunks = Blosc2IO.comp_blosc2_params(array.shape, array.shape[1:])
    tmp = path.with_name(f".partial.{path.name}")
    tmp.unlink(missing_ok=True)
    Blosc2IO.save(array, str(tmp), chunks=chunks, blocks=blocks)
    tmp.rename(path)


def stratified_split(
    labels: pd.DataFrame,
    fractions: dict[str, float] = SPLIT_FRACTIONS,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[str]]:
    """Label-stratified train/val/test split of ``labels.case_id``."""
    cases, y = labels["case_id"].to_numpy(), labels["label"].to_numpy()
    train_val, test, y_train_val, _ = train_test_split(
        cases, y, test_size=fractions["test"], stratify=y, random_state=seed
    )
    val_share = fractions["val"] / (fractions["train"] + fractions["val"])
    train, val = train_test_split(
        train_val, test_size=val_share, stratify=y_train_val, random_state=seed
    )
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def repeated_splits(
    labels: pd.DataFrame, n_splits: int = N_SPLITS, seed: int = DEFAULT_SEED
) -> list[dict[str, list[str]]]:
    """``n_splits`` independent 40/10/50 splits (seeds ``seed``, ``seed+1``, ...).

    glovita treats each entry as a fold (``--data.fold``).
    """
    return [stratified_split(labels, seed=seed + i) for i in range(n_splits)]


def _to_int16(image: sitk.Image) -> sitk.Image:
    array = sitk.GetArrayFromImage(image)
    out = sitk.GetImageFromArray(np.clip(np.round(array), -32768, 32767).astype(np.int16))
    out.CopyInformation(image)
    return out


def _process_case(
    image_path: Path, roi_path: Path, roi_label: int, b2nd_out: Path, qc_dir: Path, case_id: str
) -> bool:
    """Crop one case; returns whether the ROI was found."""
    image = sitk.Cast(sitk.ReadImage(image_path), sitk.sitkFloat32)
    array = sitk.GetArrayViewFromImage(image)
    mean, std = float(array.mean()), float(array.std())

    mask = sitk.ReadImage(roi_path) == roi_label
    center = roi_center(mask, 1)
    if center is None:
        logger.warning(f"{case_id}: ROI not found, cropping around image centre")
        center = image_center(image)

    crop = crop_around(image, center)
    write_b2nd(zscore(sitk.GetArrayFromImage(crop), mean, std)[None], b2nd_out)

    mask_crop = crop_around(
        sitk.Cast(mask, sitk.sitkUInt8), center, interpolator=sitk.sitkNearestNeighbor, pad_value=0
    )
    sitk.WriteImage(_to_int16(crop), qc_dir / "images" / f"{case_id}.nii.gz", True)
    sitk.WriteImage(mask_crop, qc_dir / "masks" / f"{case_id}.nii.gz", True)
    return center is not None and bool(sitk.GetArrayViewFromImage(mask_crop).any())


def prepare_task(
    task: ClassificationTask,
    device: str = "gpu",
    fast_roi: bool = True,
    seed: int = DEFAULT_SEED,
    n_splits: int = N_SPLITS,
    overwrite: bool = False,
) -> None:
    """Build ``data/processed/<task>/`` (glovita ``generic_3d_dataset`` layout).

    Processed::

        images/<case_id>.b2nd     float32 (1, z, y, x), 192^3 @ 1 mm, ROI-centred, z-scored
        labels.json               {case_id: class index}
        splits.json               [{"train", "val", "test"}, ...] one entry per fold
        dataset.json              num_classes/subtask for glovita + preprocessing record
        labels.csv                case_id,label,roi_found

    Interim (QC, viewable in ITK-SNAP)::

        <task>/images/<case_id>.nii.gz   same crop in int16 HU
        <task>/masks/<case_id>.nii.gz    ROI mask in the same grid
    """
    out_dir = PROCESSED_DATA_DIR / task.name
    qc_dir = INTERIM_DATA_DIR / task.name
    roi_dir = INTERIM_DATA_DIR / task.source / "totalseg"
    for d in (out_dir / "images", qc_dir / "images", qc_dir / "masks"):
        d.mkdir(parents=True, exist_ok=True)

    labels = task.load_labels()
    roi_label = totalseg_class_ids([task.roi])[task.roi]
    roi_found = {}

    for case_id in tqdm(labels["case_id"], desc=task.name):
        b2nd_out = out_dir / "images" / f"{case_id}.b2nd"
        mask_qc = qc_dir / "masks" / f"{case_id}.nii.gz"
        if b2nd_out.exists() and mask_qc.exists() and not overwrite:
            roi_found[case_id] = bool(sitk.GetArrayViewFromImage(sitk.ReadImage(mask_qc)).any())
            continue
        roi_path = roi_dir / f"{case_id}.nii.gz"
        if not roi_path.exists():
            segment_rois(task.image_path(case_id), roi_path, [task.roi], device, fast_roi)
        roi_found[case_id] = _process_case(
            task.image_path(case_id), roi_path, roi_label, b2nd_out, qc_dir, case_id
        )

    labels = labels.assign(roi_found=labels["case_id"].map(roi_found))
    labels.to_csv(out_dir / "labels.csv", index=False)
    with open(out_dir / "labels.json", "w") as f:
        json.dump(labels.set_index("case_id")["label"].astype(int).to_dict(), f, indent=1)

    splits = repeated_splits(labels, n_splits=n_splits, seed=seed)
    with open(out_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=1)

    summary = {
        "num_classes": 2,
        "subtask": "multiclass",
        "name": task.name,
        "source": task.source,
        "description": task.description,
        "modality": task.modality,
        "target": task.target,
        "roi": task.roi,
        "spacing_mm": list(TARGET_SPACING_MM),
        "crop_size": list(CROP_SIZE),
        "axis_order": "z,y,x (LPS identity direction)",
        "normalization": "z-score with whole-scan mean/std (nnU-Net ZScoreNormalization)",
        "pad_value_hu": CT_PAD_VALUE,
        "split_seeds": [seed + i for i in range(n_splits)],
        "num_cases": len(labels),
        "num_positive": int(labels["label"].sum()),
        "num_roi_missing": int((~labels["roi_found"]).sum()),
        "positives_per_fold": [
            {k: int(labels.set_index("case_id").loc[v, "label"].sum()) for k, v in s.items()}
            for s in splits
        ],
    }
    with open(out_dir / "dataset.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.success(f"{task.name}: {summary['num_cases']} cases -> {out_dir}")
    logger.info(f"Positives per fold: {summary['positives_per_fold']}")

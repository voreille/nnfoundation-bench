"""Shared preprocessing: ROI-centred 1 mm / 192^3 crops and stratified splits."""

import json
from collections.abc import Sequence

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
    PROCESSED_DATA_DIR,
    SPLIT_FRACTIONS,
    TARGET_SPACING_MM,
)
from nnfoundation_bench.datasets.base import ClassificationTask
from nnfoundation_bench.roi import segment_rois, totalseg_class_ids

Point = tuple[float, float, float]


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
) -> sitk.Image:
    """Resample ``image`` onto a ``size`` grid of ``spacing`` centred on ``center``.

    Cropping and resampling happen in one step in physical space, keeping the
    image's direction cosines; regions outside the field of view get ``pad_value``.
    """
    direction = np.array(image.GetDirection()).reshape(3, 3)
    half_extent = (np.array(size) - 1) / 2 * np.array(spacing)
    origin = np.array(center) - direction @ half_extent
    reference = sitk.Image([int(s) for s in size], image.GetPixelID())
    reference.SetSpacing([float(s) for s in spacing])
    reference.SetDirection(image.GetDirection())
    reference.SetOrigin(origin.tolist())
    return sitk.Resample(
        image, reference, sitk.Transform(), interpolator, pad_value, image.GetPixelID()
    )


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


def _to_int16(image: sitk.Image) -> sitk.Image:
    array = sitk.GetArrayFromImage(image)
    out = sitk.GetImageFromArray(np.clip(np.round(array), -32768, 32767).astype(np.int16))
    out.CopyInformation(image)
    return out


def prepare_task(
    task: ClassificationTask,
    device: str = "gpu",
    fast_roi: bool = True,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> None:
    """Build ``data/processed/<task>/`` from the raw source.

    Layout::

        images/<case_id>.nii.gz   int16 HU, 192^3 @ 1 mm, centred on the ROI
        masks/<case_id>.nii.gz    ROI mask in the same grid (QC only)
        labels.csv                case_id,label,roi_found
        splits.json               {"seed", "train", "val", "test"}
        dataset.json              task description and preprocessing parameters
    """
    out_dir = PROCESSED_DATA_DIR / task.name
    roi_dir = INTERIM_DATA_DIR / task.source / "totalseg"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    labels = task.load_labels()
    roi_label = totalseg_class_ids([task.roi])[task.roi]
    roi_found = {}

    for case_id in tqdm(labels["case_id"], desc=task.name):
        image_out = out_dir / "images" / f"{case_id}.nii.gz"
        mask_out = out_dir / "masks" / f"{case_id}.nii.gz"
        roi_path = roi_dir / f"{case_id}.nii.gz"
        if image_out.exists() and mask_out.exists() and not overwrite:
            roi_found[case_id] = bool(sitk.GetArrayViewFromImage(sitk.ReadImage(mask_out)).any())
            continue
        if not roi_path.exists():
            segment_rois(task.image_path(case_id), roi_path, [task.roi], device, fast_roi)

        image = sitk.ReadImage(task.image_path(case_id))
        mask = sitk.ReadImage(roi_path) == roi_label
        center = roi_center(mask, 1)
        roi_found[case_id] = center is not None
        if center is None:
            logger.warning(f"{case_id}: no {task.roi} found, cropping around image centre")
            center = image_center(image)

        sitk.WriteImage(_to_int16(crop_around(image, center)), image_out, useCompression=True)
        mask_crop = crop_around(
            sitk.Cast(mask, sitk.sitkUInt8),
            center,
            interpolator=sitk.sitkNearestNeighbor,
            pad_value=0,
        )
        sitk.WriteImage(mask_crop, mask_out, useCompression=True)

    labels = labels.assign(roi_found=labels["case_id"].map(roi_found))
    labels.to_csv(out_dir / "labels.csv", index=False)

    splits = stratified_split(labels, seed=seed)
    with open(out_dir / "splits.json", "w") as f:
        json.dump({"seed": seed, **splits}, f, indent=2)

    split_of = {case: name for name, cases in splits.items() for case in cases}
    counts = (
        labels.assign(split=labels["case_id"].map(split_of))
        .groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
    )
    summary = {
        "name": task.name,
        "source": task.source,
        "description": task.description,
        "modality": task.modality,
        "target": task.target,
        "roi": task.roi,
        "spacing_mm": list(TARGET_SPACING_MM),
        "crop_size": list(CROP_SIZE),
        "pad_value": CT_PAD_VALUE,
        "num_cases": len(labels),
        "num_positive": int(labels["label"].sum()),
        "num_roi_missing": int((~labels["roi_found"]).sum()),
        "split_counts": {
            s: {int(k): int(v) for k, v in row.items()} for s, row in counts.iterrows()
        },
    }
    with open(out_dir / "dataset.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.success(f"{task.name}: {summary['num_cases']} cases -> {out_dir}")
    logger.info(f"Split counts (label 0/1): {summary['split_counts']}")

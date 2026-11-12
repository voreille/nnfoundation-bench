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
from nnfoundation_bench.datasets.base import CaseInputs, ClassificationTask

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


def _strata(labels: np.ndarray, strata: np.ndarray, min_count: int) -> np.ndarray:
    """Use ``strata`` where each group has >= ``min_count`` cases, else fall back to the label."""
    values, counts = np.unique(strata, return_counts=True)
    rare = set(values[counts < min_count])
    out = np.array([str(lab) if s in rare else s for lab, s in zip(labels, strata, strict=True)])
    # sklearn needs >= 2 cases per group; leftovers join the largest group.
    values, counts = np.unique(out, return_counts=True)
    largest = values[np.argmax(counts)]
    return np.where(np.isin(out, values[counts < 2]), largest, out)


def stratified_split(
    labels: pd.DataFrame,
    fractions: dict[str, float] = SPLIT_FRACTIONS,
    seed: int = DEFAULT_SEED,
    min_stratum: int = 4,
) -> dict[str, list[str]]:
    """Stratified train/val/test split of ``labels.case_id``.

    Stratifies on ``labels.stratum`` when present (e.g. centre x label), otherwise on
    the label. Strata smaller than ``min_stratum`` are merged into their label's group.
    """
    cases, y = labels["case_id"].to_numpy(), labels["label"].to_numpy()
    strata = labels["stratum"].astype(str).to_numpy() if "stratum" in labels else y.astype(str)
    train_val, test, y_tv, _, s_tv, _ = train_test_split(
        cases, y, strata, test_size=fractions["test"], random_state=seed,
        stratify=_strata(y, strata, min_stratum),
    )  # fmt: skip
    val_share = fractions["val"] / (fractions["train"] + fractions["val"])
    train, val = train_test_split(
        train_val,
        test_size=val_share,
        random_state=seed,
        stratify=_strata(y_tv, s_tv, min_stratum),
    )
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def repeated_splits(
    labels: pd.DataFrame, n_splits: int = N_SPLITS, seed: int = DEFAULT_SEED
) -> list[dict[str, list[str]]]:
    """``n_splits`` independent 40/10/50 splits (seeds ``seed``, ``seed+1``, ...).

    glovita treats each entry as a fold (``--data.fold``).
    """
    return [stratified_split(labels, seed=seed + i) for i in range(n_splits)]


def _process_case(inputs: CaseInputs, pad_value: float, out_dir: Path, qc_dir: Path, case_id: str):
    """Crop one case around its ROI; returns whether the ROI is present in the crop."""
    image = sitk.Cast(sitk.ReadImage(inputs.image), sitk.sitkFloat32)
    array = sitk.GetArrayViewFromImage(image)
    mean, std = float(array.mean()), float(array.std())

    if inputs.roi_mask is None:
        mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        mask.CopyInformation(image)
    else:
        mask = sitk.Cast(sitk.ReadImage(inputs.roi_mask) == inputs.roi_label, sitk.sitkUInt8)
    center = roi_center(mask, 1)
    if center is None:
        center = image_center(image)

    crop = crop_around(image, center, pad_value=pad_value)
    mask_crop = crop_around(mask, center, interpolator=sitk.sitkNearestNeighbor, pad_value=0)
    mask_array = sitk.GetArrayFromImage(mask_crop)

    write_b2nd(
        zscore(sitk.GetArrayFromImage(crop), mean, std)[None],
        out_dir / "images" / f"{case_id}.b2nd",
    )
    write_b2nd(mask_array[None], out_dir / "masks" / f"{case_id}.b2nd")
    sitk.WriteImage(crop, qc_dir / "images" / f"{case_id}.nii.gz", True)
    sitk.WriteImage(mask_crop, qc_dir / "masks" / f"{case_id}.nii.gz", True)
    return bool(mask_array.any())


def _read_mask_b2nd(path: Path) -> np.ndarray:
    import blosc2

    return blosc2.open(str(path), mode="r")[...]


def prepare_task(
    task: ClassificationTask,
    device: str = "gpu",
    seed: int = DEFAULT_SEED,
    n_splits: int = N_SPLITS,
    overwrite: bool = False,
) -> None:
    """Build ``data/processed/<task>/`` (glovita ``generic_3d_dataset`` layout).

    Processed::

        images/<case_id>.b2nd     float32 (1, z, y, x), 192^3 @ 1 mm, ROI-centred, z-scored
        masks/<case_id>.b2nd      uint8 (1, z, y, x) ROI mask on the same grid (ROI pooling)
        labels.json               {case_id: class index}
        splits.json               [{"train", "val", "test"}, ...] one entry per fold
        dataset.json              num_classes/subtask for glovita + preprocessing record
        labels.csv                case_id,label,roi_found for every labelled case

    Interim (QC, viewable in ITK-SNAP)::

        <task>/images/<case_id>.nii.gz   same crop in physical units (HU / SUV), float32
        <task>/masks/<case_id>.nii.gz    ROI mask on the same grid

    Cases whose processing fails, and (if ``task.require_roi``) cases without ROI, are
    left out of labels.json and splits.json and listed in dataset.json.
    """
    out_dir = PROCESSED_DATA_DIR / task.name
    qc_dir = INTERIM_DATA_DIR / task.name
    for d in (out_dir / "images", out_dir / "masks", qc_dir / "images", qc_dir / "masks"):
        d.mkdir(parents=True, exist_ok=True)

    labels = task.load_labels()
    roi_found, failed = {}, {}

    for case_id in tqdm(labels["case_id"], desc=task.name):
        image_out = out_dir / "images" / f"{case_id}.b2nd"
        mask_out = out_dir / "masks" / f"{case_id}.b2nd"
        if image_out.exists() and mask_out.exists() and not overwrite:
            roi_found[case_id] = bool(_read_mask_b2nd(mask_out).any())
            continue
        try:
            inputs = task.case_inputs(case_id, device)
            roi_found[case_id] = _process_case(inputs, task.pad_value, out_dir, qc_dir, case_id)
        except Exception as e:  # noqa: BLE001 - one bad case must not stop the run
            logger.exception(f"{case_id}: preprocessing failed")
            failed[case_id] = f"{type(e).__name__}: {e}"
            continue
        if not roi_found[case_id]:
            logger.warning(f"{case_id}: no {task.roi} in the crop")

    labels = labels.assign(roi_found=labels["case_id"].map(roi_found))
    labels.to_csv(out_dir / "labels.csv", index=False)
    usable = labels["case_id"].isin(roi_found) & (
        labels["roi_found"].fillna(False).astype(bool) | (not task.require_roi)
    )
    no_roi = sorted(set(roi_found) - set(labels.loc[usable, "case_id"]))
    labels = labels[usable].reset_index(drop=True)

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
        "pad_value": task.pad_value,
        "split_seeds": [seed + i for i in range(n_splits)],
        "num_cases": len(labels),
        "num_positive": int(labels["label"].sum()),
        "excluded_no_roi": no_roi if task.require_roi else [],
        "crops_without_roi": [] if task.require_roi else no_roi,
        "failed": failed,
        "positives_per_fold": [
            {k: int(labels.set_index("case_id").loc[v, "label"].sum()) for k, v in s.items()}
            for s in splits
        ],
    }
    with open(out_dir / "dataset.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.success(f"{task.name}: {summary['num_cases']} cases -> {out_dir}")
    if failed or summary["excluded_no_roi"]:
        logger.warning(
            f"Excluded: {len(failed)} failed, {len(summary['excluded_no_roi'])} without ROI"
        )
    logger.info(f"Positives per fold: {summary['positives_per_fold']}")

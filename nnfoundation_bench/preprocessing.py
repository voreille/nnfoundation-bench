"""Shared preprocessing: ROI-centred 1 mm / 192^3 crops exported for glovita.

Output follows glovita's ``generic_3d_dataset`` format (see
https://github.com/MIC-DKFZ/image_classification/blob/master/docs/nnfoundation.md).
Organ tasks (D38) use a scan layout instead: whole scans at 1 mm plus a label map
(``prepare_scan_task``), cropped per organ at extract time.
"""

import json
import os
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
    MIN_ORGAN_MM3,
    N_SPLITS,
    PROCESSED_DATA_DIR,
    SPLIT_FRACTIONS,
    TARGET_SPACING_MM,
)
from nnfoundation_bench.datasets.base import CaseInputs, ClassificationTask
from nnfoundation_bench.datasets.totalsegmentator import (
    most_common_structures,
    organ_samples,
    structure_classes,
)

Point = tuple[float, float, float]
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def read_image(path: Path) -> sitk.Image:
    """``sitk.ReadImage``, also for NIfTI files whose direction is not quite orthonormal.

    ITK refuses such headers ("ITK only supports orthonormal direction cosines"); e.g.
    TotalSegmentator v2.0.1 sforms are off by ~2e-4 (float32 rounding), in the image and
    some masks of most scans. Those are read with nibabel and given the nearest orthonormal
    direction (polar decomposition), which moves voxels by well under 0.1 mm.
    """
    try:
        return sitk.ReadImage(str(path))
    except RuntimeError as e:
        if "orthonormal" not in str(e):
            raise
    import nibabel as nib

    nifti = nib.load(str(path))
    affine = nifti.affine
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    u, _, vt = np.linalg.svd(affine[:3, :3] / spacing)
    ras_to_lps = np.diag([-1.0, -1.0, 1.0])
    image = sitk.GetImageFromArray(np.asanyarray(nifti.dataobj).transpose(2, 1, 0))
    image.SetSpacing(spacing.tolist())
    image.SetDirection((ras_to_lps @ u @ vt).flatten().tolist())
    image.SetOrigin((ras_to_lps @ affine[:3, 3]).tolist())
    return image


def roi_center(mask: sitk.Image, label: int) -> Point | None:
    """Physical centre of the bounding box of ``label`` in ``mask`` (None if absent)."""
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.Cast(mask, sitk.sitkUInt8))
    if label not in stats.GetLabels():
        return None
    bbox = stats.GetBoundingBox(label)  # (x, y, z, size_x, size_y, size_z) in voxels
    start, size = np.array(bbox[:3]), np.array(bbox[3:])
    return mask.TransformContinuousIndexToPhysicalPoint((start + (size - 1) / 2).tolist())


def binary_roi(
    mask: sitk.Image, labels: int | Sequence[int], keep_labels: bool = False
) -> sitk.Image:
    """uint8 mask of the voxels whose value is in ``labels`` (one label or a union).

    The mask is 0/1, or with ``keep_labels`` the original values of ``labels`` (others 0).
    """
    labels = [labels] if isinstance(labels, int) else list(labels)
    array = sitk.GetArrayViewFromImage(mask)
    inside = np.isin(array, labels)
    roi = sitk.GetImageFromArray(
        np.where(inside, array, 0).astype(np.uint8) if keep_labels else inside.astype(np.uint8)
    )
    roi.CopyInformation(mask)
    return roi


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


def resample_whole(
    image: sitk.Image,
    spacing: Sequence[float] = TARGET_SPACING_MM,
    interpolator: int = sitk.sitkLinear,
    pad_value: float = CT_PAD_VALUE,
) -> sitk.Image:
    """The whole extent of ``image`` at ``spacing``, LPS identity direction (``crop_around``
    on the physical bounding box of its 8 corners). A rotated field of view gets
    ``pad_value`` in the corners of that box."""
    center, size = _whole_grid(image, spacing)
    return crop_around(image, center, size, spacing, interpolator, pad_value)


def _whole_grid(image: sitk.Image, spacing: Sequence[float]) -> tuple[Point, list[int]]:
    """(centre, size) of the identity-direction grid covering ``image``."""
    last = np.array(image.GetSize()) - 1
    corners = np.array(
        [
            image.TransformContinuousIndexToPhysicalPoint((last * np.array(c)).tolist())
            for c in np.ndindex(2, 2, 2)
        ]
    )
    lo, hi = corners.min(0), corners.max(0)
    size = [int(np.ceil(e / s - 1e-6)) + 1 for e, s in zip(hi - lo, spacing, strict=True)]
    return tuple(((lo + hi) / 2).tolist()), size


def merge_label_map(seg_dir: Path, classes: tuple[str, ...]) -> tuple[sitk.Image, int]:
    """uint8 label map of ``seg_dir/<class>.nii.gz``: id = index in ``classes`` + 1, 0 =
    background. A voxel in several masks goes to the first class; returns the map and the
    number of such overlapping voxels."""
    if len(classes) > 255:
        raise ValueError(f"{len(classes)} classes do not fit in uint8")
    out, reference, overlap = None, None, 0
    for label_id, name in enumerate(classes, start=1):
        mask = read_image(seg_dir / f"{name}.nii.gz")
        view = sitk.GetArrayViewFromImage(mask)
        if out is None:
            out, reference = np.zeros(view.shape, np.uint8), mask
        elif view.shape != out.shape:
            raise ValueError(f"{name}: mask shape {view.shape} differs from {out.shape}")
        flat = out.reshape(-1)  # a view: organs are small, so work on their voxels only
        idx = np.flatnonzero(view)
        current = flat[idx]
        overlap += int(np.count_nonzero(current))
        flat[idx[current == 0]] = label_id
    label_map = sitk.GetImageFromArray(out)
    label_map.CopyInformation(reference)
    return label_map, overlap


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


def official_split(
    labels: pd.DataFrame, val_fraction: float = 0.2, seed: int = DEFAULT_SEED, min_stratum: int = 4
) -> dict[str, list[str]]:
    """The dataset's own train/test split (``official_split`` column), with ``val_fraction``
    of its train cases held out as val (stratified like the repeated splits)."""
    train_val = labels[labels["official_split"] == "train"]
    test = labels.loc[labels["official_split"] == "test", "case_id"]
    y = train_val["label"].to_numpy()
    strata = train_val["stratum"].astype(str).to_numpy() if "stratum" in labels else y.astype(str)
    train, val = train_test_split(
        train_val["case_id"].to_numpy(),
        test_size=val_fraction,
        random_state=seed,
        stratify=_strata(y, strata, min_stratum),
    )
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def _scan_groups(samples: pd.DataFrame) -> pd.Series:
    """scan_id -> sorted list of its case_ids."""
    return samples.groupby("scan_id")["case_id"].agg(sorted)


def _scans_per_split(split: dict[str, list[str]]) -> dict[str, int]:
    """Number of scans in each part of a split of ``<scan>__<organ>`` case IDs."""
    return {k: len({c.split("__")[0] for c in v}) for k, v in split.items()}


def _expand(groups: pd.Series, scans) -> list[str]:
    return sorted(c for scan in scans for c in groups[scan])


def scan_splits(
    samples: pd.DataFrame, fractions: dict[str, float] = SPLIT_FRACTIONS, seed: int = DEFAULT_SEED
) -> dict[str, list[str]]:
    """train/val/test split of the unique ``scan_id`` of ``samples`` (unstratified: a scan
    has no single label), expanded to their ``case_id``: a scan's samples share a split.

    Scan counts are the rounded fractions, with at least one train scan and (from 3 scans
    on) one val scan, so a handful of scans (a smoke test) still splits.
    """
    groups = _scan_groups(samples)
    scans = np.random.default_rng(seed).permutation(groups.index.to_numpy())
    n = len(scans)
    n_val = int(np.floor(n * fractions["val"] + 0.5))
    n_val = max(n_val, 1) if n >= 3 else 0  # the probe picks C on val
    n_test = min(int(np.floor(n * fractions["test"] + 0.5)), max(n - n_val - 1, 0))
    test, val, train = np.split(scans, [n_test, n_test + n_val])
    return {k: _expand(groups, v) for k, v in (("train", train), ("val", val), ("test", test))}


def official_scan_split(
    samples: pd.DataFrame, val_fraction: float = 0.2, seed: int = DEFAULT_SEED
) -> dict[str, list[str]]:
    """The dataset's own split per scan (``official_split``: train/val/test). Without a val,
    ``val_fraction`` of the train scans is held out as val."""
    groups = _scan_groups(samples)
    official = samples.groupby("scan_id")["official_split"].first()
    unexpected = sorted(set(official.dropna()) - {"train", "val", "test"})
    if official.isna().any() or unexpected:
        raise ValueError(
            f"official_split must be train/val/test for every scan: {official.isna().sum()} "
            f"missing, unexpected values {unexpected}"
        )
    if not (official == "train").any():
        raise ValueError("official_split has no train scan")
    scans = {k: sorted(official.index[official == k]) for k in ("train", "val", "test")}
    if not scans["val"]:
        scans["train"], scans["val"] = train_test_split(
            np.array(scans["train"]), test_size=val_fraction, random_state=seed
        )
    return {k: _expand(groups, v) for k, v in scans.items()}


def repeated_splits(
    labels: pd.DataFrame, n_splits: int = N_SPLITS, seed: int = DEFAULT_SEED
) -> list[dict[str, list[str]]]:
    """``n_splits`` independent 40/10/50 splits (seeds ``seed``, ``seed+1``, ...).

    glovita treats each entry as a fold (``--data.fold``).
    """
    return [stratified_split(labels, seed=seed + i) for i in range(n_splits)]


def _link(link: Path, target: Path) -> None:
    """Relative symlink ``link -> target`` (kept if it already points there)."""
    rel = Path(os.path.relpath(target, link.parent))
    if link.is_symlink() and Path(os.readlink(link)) == rel:
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"{link} exists and is not a link to {target}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(rel, target_is_directory=True)


def prepare_split_variant(task: ClassificationTask, seed: int = DEFAULT_SEED) -> None:
    """Build ``data/processed/<task>/`` for a task that reuses the prepared task
    ``task.split_of`` and only changes its splits (D37).

    Links the parent's ``images``, ``masks``, ``features`` and ``radiomics`` (so frozen
    features and radiomics are computed once) and its QC crops in data/interim. Writes
    the parent's labels.json, its labels.csv plus the ``official_split`` column, and a
    ``splits.json`` with one fold: the dataset's official train/test split, 20% of train
    held out as val (``official_split``).
    """
    parent_dir = PROCESSED_DATA_DIR / task.split_of
    if not (parent_dir / "labels.json").exists():
        raise FileNotFoundError(f"{task.name}: prepare {task.split_of} first")
    out_dir = PROCESSED_DATA_DIR / task.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("images", "masks", "features", "radiomics"):
        (parent_dir / name).mkdir(exist_ok=True)  # features may not be extracted yet
        _link(out_dir / name, parent_dir / name)
    _link(INTERIM_DATA_DIR / task.name, INTERIM_DATA_DIR / task.split_of)
    if task.organ_task:
        _official_scan_variant(task, parent_dir, out_dir, seed)
        return

    # From the raw labels: a parent prepared before the column existed lacks it.
    official = task.load_labels().set_index("case_id")["official_split"]
    labels = pd.read_csv(parent_dir / "labels.csv", dtype={"case_id": str})
    labels["official_split"] = labels["case_id"].map(official)
    labels.to_csv(out_dir / "labels.csv", index=False)
    usable = json.loads((parent_dir / "labels.json").read_text())
    (out_dir / "labels.json").write_text(json.dumps(usable, indent=1))
    split = official_split(labels[labels["case_id"].isin(usable)], seed=seed)
    (out_dir / "splits.json").write_text(json.dumps([split], indent=1))

    summary = json.loads((parent_dir / "dataset.json").read_text())
    summary |= {
        "name": task.name,
        "description": task.description,
        "split_of": task.split_of,
        "splits": "one fold: official train (80% train, 20% val) / official test",
        "split_seeds": [seed],
        "positives_per_fold": [{k: sum(usable[c] for c in v) for k, v in split.items()}],
        "cases_per_fold": [{k: len(v) for k, v in split.items()}],
    }
    (out_dir / "dataset.json").write_text(json.dumps(summary, indent=2))
    logger.success(f"{task.name}: official split of {task.split_of} -> {out_dir}")
    logger.info(
        f"Cases: {summary['cases_per_fold'][0]}, positives: {summary['positives_per_fold'][0]}"
    )


def _official_scan_variant(
    task: ClassificationTask, parent_dir: Path, out_dir: Path, seed: int
) -> None:
    """Organ-task part of ``prepare_split_variant``: one fold from the scans' official split
    (labels.csv ``official_split``), grouped by scan."""
    for name in ("labels.csv", "labels.json", "scans.csv", "organs.csv"):
        (out_dir / name).write_text((parent_dir / name).read_text())
    labels = pd.read_csv(parent_dir / "labels.csv", dtype={"case_id": str, "scan_id": str})
    split = official_scan_split(labels, seed=seed)
    (out_dir / "splits.json").write_text(json.dumps([split], indent=1))
    has_val = (labels["official_split"] == "val").any()
    summary = json.loads((parent_dir / "dataset.json").read_text())
    summary |= {
        "name": task.name,
        "description": task.description,
        "split_of": task.split_of,
        "splits": "one fold: official train / val / test by scan"
        if has_val
        else "one fold: official train (80% train, 20% val, by scan) / official test",
        "split_seeds": [seed],
        "cases_per_fold": [{k: len(v) for k, v in split.items()}],
        "scans_per_fold": [_scans_per_split(split)],
    }
    (out_dir / "dataset.json").write_text(json.dumps(summary, indent=2))
    logger.success(f"{task.name}: official split of {task.split_of} -> {out_dir}")
    logger.info(f"Scans: {summary['scans_per_fold'][0]}, samples: {summary['cases_per_fold'][0]}")


def _process_case(
    inputs: CaseInputs,
    task: ClassificationTask,
    out_dir: Path,
    qc_dir: Path,
    case_id: str,
):
    """Crop one case around its ROI; returns whether the ROI is present in the crop."""
    image = sitk.Cast(read_image(inputs.image), sitk.sitkFloat32)
    if task.intensity_min is not None:
        image = sitk.Maximum(image, float(task.intensity_min))
    array = sitk.GetArrayViewFromImage(image)
    mean, std = float(array.mean()), float(array.std())

    if inputs.roi_mask is None:
        mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        mask.CopyInformation(image)
    else:
        mask = binary_roi(read_image(inputs.roi_mask), inputs.roi_label, task.keep_roi_labels)
    center = roi_center(sitk.Cast(mask > 0, sitk.sitkUInt8), 1)  # union of the ROI labels
    if center is None:
        center = image_center(image)

    crop = crop_around(image, center, pad_value=task.pad_value)
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
    left out of labels.json and splits.json and listed in dataset.json. A task with
    ``split_of`` reuses that prepared task instead (``prepare_split_variant``).
    """
    if task.split_of:
        prepare_split_variant(task, seed=seed)
        return
    if task.organ_task:
        prepare_scan_task(task, seed=seed, n_splits=n_splits, overwrite=overwrite)
        return
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
            roi_found[case_id] = _process_case(inputs, task, out_dir, qc_dir, case_id)
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
    if "official_split" in labels:
        with open(out_dir / "splits_official.json", "w") as f:
            json.dump([official_split(labels, seed=seed)], f, indent=1)

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
        "intensity_min": task.intensity_min,
        "mask_values": "ROI labels kept" if task.keep_roi_labels else "binary (union)",
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


def _process_scan(
    task: ClassificationTask,
    scan_id: str,
    structures: tuple[str, ...],
    out_dir: Path,
    qc_dir: Path,
) -> dict:
    """Resample one scan and its merged label map (``structures``, already checked against
    the scan's files) to 1 mm; returns its scans.csv row and organs.csv rows."""
    image_path, seg_dir = task.scan_paths(scan_id)
    image = sitk.Cast(read_image(image_path), sitk.sitkFloat32)
    if task.intensity_min is not None:
        image = sitk.Maximum(image, float(task.intensity_min))
    array = sitk.GetArrayViewFromImage(image)
    mean, std = float(array.mean()), float(array.std())
    label_map, overlap = merge_label_map(seg_dir, structures)
    sitk.WriteImage(label_map, qc_dir / "labelmaps" / f"{scan_id}.nii.gz", True)

    center, size = _whole_grid(image, TARGET_SPACING_MM)
    resampled = crop_around(image, center, size, pad_value=task.pad_value)
    mask = sitk.GetArrayFromImage(
        crop_around(label_map, center, size, interpolator=sitk.sitkNearestNeighbor, pad_value=0)
    )
    image_z = zscore(sitk.GetArrayFromImage(resampled), mean, std)
    write_b2nd(image_z[None], out_dir / "images" / f"{scan_id}.b2nd")
    write_b2nd(mask[None], out_dir / "masks" / f"{scan_id}.b2nd")

    voxels = np.bincount(mask.ravel(), minlength=len(structures) + 1)
    scan = {
        "scan_id": scan_id,
        "shape": str(tuple(int(n) for n in mask.shape)),
        "mean": mean,
        "std": std,
        "pad_z": (task.pad_value - mean) / max(std, 1e-8),
    }
    organs = [
        {"scan_id": scan_id, "organ": name, "label_id": i, "voxels": int(voxels[i]),
         "overlap_voxels": overlap}
        for i, name in enumerate(structures, start=1)
    ]  # fmt: skip
    return {"scan": scan, "organs": organs}


def prepare_scan_task(
    task: ClassificationTask,
    seed: int = DEFAULT_SEED,
    n_splits: int = N_SPLITS,
    overwrite: bool = False,
) -> None:
    """Build ``data/processed/<task>/`` of an organ task (scan layout, D38).

    Processed::

        images/<scan_id>.b2nd   float32 (1, z, y, x), whole scan, 1 mm, LPS, z-scored
        masks/<scan_id>.b2nd    uint8 (1, z, y, x) label map, same grid
        scans.csv               scan_id, shape, mean, std, pad_z + the scan's metadata
        organs.csv              scan_id, organ, label_id, voxels (1 mm), overlap_voxels
                                (the scan's total, native grid, on every row)
        labels.csv/json         one (scan, organ) sample per row: case_id = <scan>__<organ>
        splits.json             repeated 40/10/50 splits drawn on scans
        dataset.json            layout, classes, excluded classes, counts, failures

    Interim: ``<task>/labelmaps/<scan_id>.nii.gz`` (merged label map on the native grid,
    for QC next to the raw image) and ``<task>/scans/<scan_id>.json`` (per-scan record;
    rerunning skips scans that have one). The label-map ids come from the sorted structure
    names that most scans have; a scan with another structure set fails. Failed scans are
    left out and listed in dataset.json (their stale outputs are removed).
    """
    out_dir = PROCESSED_DATA_DIR / task.name
    qc_dir = INTERIM_DATA_DIR / task.name
    for d in (out_dir / "images", out_dir / "masks", qc_dir / "labelmaps", qc_dir / "scans"):
        d.mkdir(parents=True, exist_ok=True)

    meta = task.load_scans()
    meta["scan_id"] = meta["scan_id"].astype(str)
    scan_ids = sorted(meta["scan_id"])
    found = {s: structure_classes(task.scan_paths(s)[1]) for s in scan_ids}
    structures = most_common_structures(found.values())
    classes = task.organ_classes()
    seeds = [seed + i for i in range(n_splits)]
    records, failed = {}, {}
    for scan_id in tqdm(scan_ids, desc=task.name):
        record_path = qc_dir / "scans" / f"{scan_id}.json"
        outputs = (
            record_path,
            out_dir / "images" / f"{scan_id}.b2nd",
            out_dir / "masks" / f"{scan_id}.b2nd",
        )
        if all(p.exists() for p in outputs) and not overwrite:
            records[scan_id] = json.loads(record_path.read_text())
            continue
        try:
            if found[scan_id] != structures:
                missing = sorted(set(structures) - set(found[scan_id]))
                extra = sorted(set(found[scan_id]) - set(structures))
                raise ValueError(f"different structure set: missing {missing}, extra {extra}")
            records[scan_id] = _process_scan(task, scan_id, structures, out_dir, qc_dir)
        except Exception as e:  # noqa: BLE001 - one bad scan must not stop the run
            logger.exception(f"{scan_id}: preprocessing failed")
            failed[scan_id] = f"{type(e).__name__}: {e}"
            for stale in outputs:  # a later rerun must not take them as finished
                stale.unlink(missing_ok=True)
            continue
        record_path.write_text(json.dumps(records[scan_id], indent=1))

    if not records:
        raise RuntimeError(
            f"{task.name}: no scan could be prepared (structure set of most scans: "
            f"{len(structures)} structures). Failures: {failed}"
        )
    scans = pd.DataFrame([r["scan"] for r in records.values()])
    scans = scans.merge(meta, on="scan_id", how="left", suffixes=("", "_meta"))
    scans.to_csv(out_dir / "scans.csv", index=False)
    organs = pd.DataFrame([o for r in records.values() for o in r["organs"]])
    organs.to_csv(out_dir / "organs.csv", index=False)
    if "split" in scans:
        organs["official_split"] = organs["scan_id"].map(scans.set_index("scan_id")["split"])
    samples = organ_samples(organs, classes, MIN_ORGAN_MM3)
    samples.to_csv(out_dir / "labels.csv", index=False)
    with open(out_dir / "labels.json", "w") as f:
        json.dump(samples.set_index("case_id")["label"].astype(int).to_dict(), f, indent=1)
    splits = [scan_splits(samples, seed=s) for s in seeds]
    with open(out_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=1)

    kept = samples.drop_duplicates("label").sort_values("label")["organ"].tolist()
    counts = samples["organ"].value_counts()
    summary = {
        "layout": "scan",
        "num_classes": len(kept),
        "subtask": "multiclass",
        "name": task.name,
        "source": task.source,
        "description": task.description,
        "modality": task.modality,
        "target": task.target,
        "roi": task.roi,
        "spacing_mm": list(TARGET_SPACING_MM),
        "crop_size": list(CROP_SIZE),
        "crop": "192^3 around each organ's bounding-box centre, at extract time",
        "axis_order": "z,y,x (LPS identity direction)",
        "normalization": "z-score with whole-scan mean/std (nnU-Net ZScoreNormalization)",
        "pad_value": task.pad_value,
        "intensity_min": task.intensity_min,
        "min_organ_mm3": MIN_ORGAN_MM3,
        "classes": {str(i): name for i, name in enumerate(kept)},
        "excluded_classes": [c for c in classes if c not in kept],
        "class_counts": {name: int(counts[name]) for name in kept},
        "label_map_ids": {str(i): name for i, name in enumerate(structures, start=1)},
        "split_seeds": seeds,
        "num_scans": len(records),
        "num_cases": len(samples),
        "failed": failed,
        "cases_per_fold": [{k: len(v) for k, v in s.items()} for s in splits],
        "scans_per_fold": [_scans_per_split(s) for s in splits],
    }
    with open(out_dir / "dataset.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.success(
        f"{task.name}: {len(records)} scans, {len(samples)} samples, {len(kept)} classes "
        f"-> {out_dir}"
    )
    if failed or summary["excluded_classes"]:
        logger.warning(
            f"{len(failed)} scans failed; classes without samples: {summary['excluded_classes']}"
        )

"""TotalSegmentator CT (v2.0.1) and MRI (v2.0.0): multi-class organ recognition (D38).

Curia-2 tasks A1/A2: given a 192^3 crop centred on one organ, predict which organ it is.
Both datasets ship expert masks for every structure (``segmentations/<structure>.nii.gz``),
so a sample is one (scan, organ) pair: ``case_id = <scan>__<organ>``, label = the organ's
index in the task's class set. ``prepare`` stores whole scans and a merged label map
(``preprocessing.prepare_scan_task``); ``nnfb-features extract`` crops each organ in memory.

Class sets:
- CT follows Curia-2's TotalSegmentator v1 setup as far as v2.0.1 allows: the 97 v1
  classes present in v2 under the same name, plus v2's ``heart`` (v1's five heart
  structures and ``pulmonary_artery`` merged into it): 98 classes. v1's ``face`` is not
  v2's ``skull`` and is left out (ASSUMPTION: Curia-2 does not list its classes).
- MRI: all 50 structures.

Raw data (registration-free, Zenodo; placed by hand, ``download`` only checks it)::

    data/raw/totalsegmentator_ct/Totalsegmentator_dataset_v201/{meta.csv,sXXXX/ct.nii.gz,
        sXXXX/segmentations/*.nii.gz}
    data/raw/totalsegmentator_mri/TotalsegmentatorMRI_dataset_v200/{meta.csv,
        sXXXX/mri.nii.gz, sXXXX/segmentations/*.nii.gz}
"""

from collections import Counter
from functools import cache, partial
from pathlib import Path

import pandas as pd
from loguru import logger

from nnfoundation_bench.config import CT_PAD_VALUE, PROCESSED_DATA_DIR, RAW_DATA_DIR
from nnfoundation_bench.datasets.base import ClassificationTask

MR_PAD_VALUE = 0.0  # background outside the MR field of view


# source -> (folder in data/raw/<source>/, image file, subjects, Zenodo record, zip, md5)
DATASETS = {
    "totalsegmentator_ct": (
        "Totalsegmentator_dataset_v201", "ct.nii.gz", 1228,
        "10047292", "Totalsegmentator_dataset_v201.zip", "fe250e5718e0a3b5df4c4ea9d58a62fe",
    ),
    "totalsegmentator_mri": (
        "TotalsegmentatorMRI_dataset_v200", "mri.nii.gz", 616,
        "14710732", "TotalsegmentatorMRI_dataset_v200.zip", "54638f4cb883ce3b34225195358c398f",
    ),
}  # fmt: skip


def dataset_dir(source: str) -> Path:
    return RAW_DATA_DIR / source / DATASETS[source][0]


def _subjects(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("s*") if p.is_dir())


def download(source: str) -> None:
    """No network: checks that the unzipped dataset is in place, else says where to get it."""
    _, _, n_subjects, record, zip_name, md5 = DATASETS[source]
    root = dataset_dir(source)
    found = len(_subjects(root)) if root.is_dir() else 0
    if found == n_subjects and (root / "meta.csv").exists():
        logger.info(f"{source}: {found} subjects in {root}")
        return
    raise FileNotFoundError(
        f"{source}: expected {n_subjects} subjects and meta.csv in {root}, found {found}. "
        f"Download {zip_name} from https://zenodo.org/records/{record} (md5 {md5}), "
        f"check it with md5sum and unzip it so that {root}/sXXXX/ exists."
    )


def read_meta(path: Path) -> pd.DataFrame:
    """meta.csv (``;``-separated, UTF-8 BOM) with ``image_id`` renamed to ``scan_id``."""
    meta = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype={"image_id": str})
    return meta.rename(columns={"image_id": "scan_id"})


def structure_classes(seg_dir: Path) -> tuple[str, ...]:
    """Sorted structure names of ``seg_dir/<structure>.nii.gz``."""
    return tuple(sorted(p.name.removesuffix(".nii.gz") for p in seg_dir.glob("*.nii.gz")))


def ct_v1_classes(v2_names) -> tuple[str, ...]:
    """CT class set: TotalSegmentator v1 names also present in ``v2_names``, plus ``heart``."""
    from totalsegmentator.map_to_binary import class_map

    v1 = set(class_map["total_v1"].values())
    return tuple(sorted((v1 & set(v2_names)) | {"heart"}))


def organ_samples(organs: pd.DataFrame, classes: tuple[str, ...], min_mm3: float) -> pd.DataFrame:
    """(scan, organ) samples: organs of the class set with ``voxels >= min_mm3`` (1 mm grid).

    ``organs`` has ``scan_id, organ, voxels`` (and optionally ``official_split``). The label
    is the organ's index among the classes that keep at least one sample (0..K-1, class
    order). Output: ``case_id, scan_id, organ, label, official_split``, sorted by case_id.
    """
    kept = organs[organs["organ"].isin(classes) & (organs["voxels"] >= min_mm3)]
    present = [c for c in classes if c in set(kept["organ"])]
    samples = pd.DataFrame(
        {
            "case_id": kept["scan_id"].astype(str) + "__" + kept["organ"],
            "scan_id": kept["scan_id"].astype(str),
            "organ": kept["organ"],
            "label": kept["organ"].map({c: i for i, c in enumerate(present)}).astype(int),
            "official_split": kept.get("official_split"),
        }
    )
    return samples.sort_values("case_id").reset_index(drop=True)


def load_scans(source: str) -> pd.DataFrame:
    """Scans on disk with their meta.csv row (``split`` = official train/val/test)."""
    root = dataset_dir(source)
    meta = read_meta(root / "meta.csv")
    on_disk = {p.name for p in _subjects(root)}
    return meta[meta["scan_id"].isin(on_disk)].sort_values("scan_id").reset_index(drop=True)


def scan_paths(source: str, scan_id: str) -> tuple[Path, Path]:
    scan_dir = dataset_dir(source) / scan_id
    return scan_dir / DATASETS[source][1], scan_dir / "segmentations"


def most_common_structures(structure_sets) -> tuple[str, ...]:
    """The structure set most scans have (the label-map id table of ``prepare``)."""
    return Counter(structure_sets).most_common(1)[0][0]


@cache
def common_structures(source: str) -> tuple[str, ...]:
    """``most_common_structures`` over every subject of SOURCE on disk."""
    subjects = _subjects(dataset_dir(source))
    if not subjects:
        download(source)  # raises with the instructions
    return most_common_structures(structure_classes(s / "segmentations") for s in subjects)


def ct_classes() -> tuple[str, ...]:
    return ct_v1_classes(common_structures("totalsegmentator_ct"))


def mri_classes() -> tuple[str, ...]:
    return common_structures("totalsegmentator_mri")


def load_labels(task: str) -> pd.DataFrame:
    """The (scan, organ) samples written by ``prepare`` (labels.csv)."""
    path = PROCESSED_DATA_DIR / task / "labels.csv"
    if not path.exists():
        raise FileNotFoundError(f"{task}: no {path}; run `nnfb-data prepare {task}` first")
    return pd.read_csv(path, dtype={"case_id": str, "scan_id": str})


def _no_case_inputs(case_id: str, device: str):
    raise NotImplementedError("Organ tasks are prepared per scan (scan_paths), not per case")


def _organ_task(name, source, description, modality, pad_value, classes, **kwargs):
    return ClassificationTask(
        name=name,
        source=source,
        description=description,
        target="organ",
        roi="organ (TotalSegmentator expert masks)",
        modality=modality,
        download=partial(download, source),
        load_labels=partial(load_labels, name),
        case_inputs=_no_case_inputs,
        pad_value=pad_value,
        organ_task=True,
        load_scans=partial(load_scans, source),
        scan_paths=partial(scan_paths, source),
        organ_classes=classes,
        **kwargs,
    )


TOTALSEG_CT_ORGANS = _organ_task(
    "totalseg_ct_organs",
    "totalsegmentator_ct",
    "TotalSegmentator CT v2.0.1: which of 98 organs is at the crop centre (Curia-2 A1)",
    "CT",
    CT_PAD_VALUE,
    ct_classes,
    intensity_min=CT_PAD_VALUE,
)
TOTALSEG_MRI_ORGANS = _organ_task(
    "totalseg_mri_organs",
    "totalsegmentator_mri",
    "TotalSegmentator MRI v2.0.0: which of 50 organs is at the crop centre (Curia-2 A2)",
    "MR",
    MR_PAD_VALUE,
    mri_classes,
)
TOTALSEG_TASKS = (TOTALSEG_CT_ORGANS, TOTALSEG_MRI_ORGANS)

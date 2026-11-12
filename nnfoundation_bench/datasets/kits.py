"""KiTS23 (Heller et al., 2023) and the classification tasks derived from it.

Images come from the official Hugging Face mirror used by the ``kits23`` repo, the
clinical metadata (``kits23.json``) from the ``neheller/kits23`` GitHub repo. Both
are pinned to fixed revisions so the benchmark is reproducible.
"""

import hashlib
import json
from functools import cache
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from nnfoundation_bench.config import CT_PAD_VALUE, INTERIM_DATA_DIR, RAW_DATA_DIR
from nnfoundation_bench.datasets.base import CaseInputs, ClassificationTask
from nnfoundation_bench.roi import segment_rois, totalseg_class_ids

SOURCE = "kits23"
RAW_DIR = RAW_DATA_DIR / SOURCE
NUM_CASES = 489

HF_REPO = "neheller/KiTS-Challenge-Imaging"
HF_REVISION = "65f1f295873a326230153c7e1de0c7dba10f0b29"

GH_COMMIT = "c1088353084c17b8882a11db71429e7c022b7785"
METADATA_URL = f"https://raw.githubusercontent.com/neheller/kits23/{GH_COMMIT}/dataset/kits23.json"
METADATA_SHA256 = "c747c720a1c8ea6476050b73eb4cc298f4030e1be0d43df1540601ffb1e9924b"
METADATA_PATH = RAW_DIR / "kits23.json"
TOTALSEG_DIR = INTERIM_DATA_DIR / SOURCE / "totalseg"


def image_path(case_id: str) -> Path:
    return RAW_DIR / "images" / f"{case_id}.nii.gz"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_metadata() -> None:
    if METADATA_PATH.exists() and _sha256(METADATA_PATH) == METADATA_SHA256:
        logger.info(f"Metadata already present: {METADATA_PATH}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {METADATA_URL}")
    response = requests.get(METADATA_URL, timeout=60)
    response.raise_for_status()
    tmp = METADATA_PATH.with_suffix(".json.partial")
    tmp.write_bytes(response.content)
    if (digest := _sha256(tmp)) != METADATA_SHA256:
        tmp.unlink()
        raise RuntimeError(f"kits23.json checksum mismatch: {digest} != {METADATA_SHA256}")
    tmp.rename(METADATA_PATH)


def download_images(max_workers: int = 8) -> None:
    """Fetch the 489 training CTs (~100 GB). Resumable: re-running skips finished files."""
    from huggingface_hub import snapshot_download

    logger.info(f"Downloading KiTS23 images from {HF_REPO}@{HF_REVISION[:8]} to {RAW_DIR}")
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        revision=HF_REVISION,
        allow_patterns=["images/case_*.nii.gz"],
        local_dir=RAW_DIR,
        max_workers=max_workers,
    )
    found = len(list((RAW_DIR / "images").glob("case_*.nii.gz")))
    if found != NUM_CASES:
        raise RuntimeError(f"Expected {NUM_CASES} KiTS23 images, found {found}")


def download() -> None:
    download_metadata()
    download_images()


@cache
def load_metadata() -> tuple[dict, ...]:
    with open(METADATA_PATH) as f:
        return tuple(json.load(f))


def liver_disease_labels(metadata: tuple[dict, ...] | list[dict]) -> pd.DataFrame:
    """1 if the patient has any Charlson liver-disease comorbidity, else 0.

    ASSUMPTION: the nnFoundation paper lists "KiTS Liver" (489 CT, target "pathology")
    without defining the label. KiTS23's only liver information is the pair of
    Charlson comorbidity flags below, so we use their union (10/489 positives at the
    pinned revision). Revisit once the authors' downstream split files are available.
    """
    rows = [
        {
            "case_id": case["case_id"],
            "label": int(
                case["comorbidities"]["mild_liver_disease"]
                or case["comorbidities"]["moderate_to_severe_liver_disease"]
            ),
        }
        for case in metadata
    ]
    return pd.DataFrame(rows, columns=["case_id", "label"])


def totalseg_inputs(case_id: str, device: str, roi: str) -> CaseInputs:
    """CT plus a cached TotalSegmentator mask of ``roi`` (KiTS has no organ masks)."""
    mask = TOTALSEG_DIR / f"{case_id}.nii.gz"
    if not mask.exists():
        segment_rois(image_path(case_id), mask, [roi], device)
    return CaseInputs(image_path(case_id), mask, totalseg_class_ids([roi])[roi])


KITS_LIVER = ClassificationTask(
    name="kits_liver",
    source=SOURCE,
    description="KiTS23 CT, crop around the liver, target: liver-disease comorbidity",
    target="liver_disease",
    roi="liver (TotalSegmentator)",
    modality="CT",
    download=download,
    load_labels=lambda: liver_disease_labels(load_metadata()),
    case_inputs=lambda case_id, device: totalseg_inputs(case_id, device, "liver"),
    pad_value=CT_PAD_VALUE,
    # Paper protocol: always crop, fall back to the image centre if the liver is missed.
    require_roi=False,
)

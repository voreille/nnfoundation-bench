"""ROI masks from TotalSegmentator, used only to position the classification crop."""

import os
from pathlib import Path

from loguru import logger

from nnfoundation_bench.config import TOTALSEG_HOME_DIR


def totalseg_class_ids(rois: list[str]) -> dict[str, int]:
    """Label value of each ROI in TotalSegmentator's multi-label "total" output."""
    from totalsegmentator.map_to_binary import class_map

    name_to_id = {name: idx for idx, name in class_map["total"].items()}
    unknown = sorted(set(rois) - set(name_to_id))
    if unknown:
        raise ValueError(f"Not TotalSegmentator 'total' classes: {unknown}")
    return {roi: name_to_id[roi] for roi in rois}


def segment_rois(
    image: Path, output: Path, rois: list[str], device: str = "gpu", fast: bool = True
) -> None:
    """Write a multi-label mask of ``rois`` for ``image``.

    ``fast`` uses the 3 mm model: coarse, but more than enough to centre a 192 mm crop.
    """
    os.environ.setdefault("TOTALSEG_HOME_DIR", str(TOTALSEG_HOME_DIR))
    from totalsegmentator.python_api import totalsegmentator

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".partial.{output.name}")
    logger.debug(f"TotalSegmentator {rois} on {image.name}")
    totalsegmentator(
        image,
        tmp,
        ml=True,
        fast=fast,
        roi_subset=None if fast else rois,
        device=device,
        quiet=True,
    )
    tmp.rename(output)

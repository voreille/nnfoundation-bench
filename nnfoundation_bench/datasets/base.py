from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ClassificationTask:
    """A case-level classification task derived from a raw source dataset.

    Several tasks may share one ``source`` (e.g. KiTS Liver / KiTS Pancreas): they
    then share the raw download and the cached TotalSegmentator ROI masks.
    """

    name: str
    source: str
    description: str
    # TotalSegmentator ("total" task) class whose bounding-box centre anchors the crop.
    roi: str
    # Name of the binary target, e.g. "liver_disease"; ``label`` is 1 when present.
    target: str
    download: Callable[[], None]
    # Returns a DataFrame with columns ``case_id`` (str) and ``label`` (int).
    load_labels: Callable[[], pd.DataFrame]
    image_path: Callable[[str], Path]
    modality: str = "CT"

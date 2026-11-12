from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CaseInputs:
    """Per-case inputs for preprocessing, in physical units and any native grid.

    ``image`` is a NIfTI in physical units (CT: HU, PET: SUVbw). ``roi_mask`` may be on a
    different grid (e.g. a CT-defined GTV for a PET image); everything is resampled in
    physical space. ``roi_mask`` is None when the case has no ROI.
    """

    image: Path
    roi_mask: Path | None
    roi_label: int = 1


@dataclass(frozen=True)
class ClassificationTask:
    """A case-level binary classification task with a ROI per case.

    Several tasks may share one ``source`` (e.g. KiTS Liver / KiTS Pancreas): they
    then share the raw data and cached intermediate files (ROI masks, conversions).
    """

    name: str
    source: str
    description: str
    # Name of the binary target, e.g. "hpv_positive"; ``label`` is 1 when present.
    target: str
    # ROI the crop is centred on, e.g. "GTVp" or "liver (TotalSegmentator)".
    roi: str
    modality: str
    download: Callable[[], None]
    # Returns a DataFrame with columns ``case_id`` (str) and ``label`` (int).
    load_labels: Callable[[], pd.DataFrame]
    # (case_id, device) -> inputs. May compute and cache intermediates (ROI masks,
    # DICOM conversion) under data/interim/<source>/.
    case_inputs: Callable[[str, str], CaseInputs]
    # Image value outside the field of view (CT: air in HU, PET: 0 SUV).
    pad_value: float
    # True: cases without ROI are excluded. False: crop around the image centre instead.
    require_roi: bool = True

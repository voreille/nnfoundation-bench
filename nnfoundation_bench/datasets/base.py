from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CaseInputs:
    """Per-case inputs for preprocessing, in physical units and any native grid.

    ``image`` is a NIfTI in physical units (CT: HU, PET: SUVbw). ``roi_mask`` may be on a
    different grid (e.g. a CT-defined GTV for a PET image); everything is resampled in
    physical space. ``roi_mask`` is None when the case has no ROI. ``roi_label`` may be a
    tuple: the ROI is then the union of those labels (e.g. the five lung lobes).
    """

    image: Path
    roi_mask: Path | None
    roi_label: int | tuple[int, ...] = 1


@dataclass(frozen=True)
class ClassificationTask:
    """A case-level binary classification task with a ROI per case.

    Several tasks may share one ``source`` (e.g. KiTS Liver / KiTS Pancreas): they
    then share the raw data and cached intermediate files (ROI masks, conversions).

    An *organ task* (``organ_task=True``, e.g. TotalSegmentator organ recognition, D38) is
    multi-class instead: a sample is one (scan, organ) pair, ``case_id = <scan>__<organ>``
    and the label is the organ's class. ``prepare`` stores whole scans with a label map
    (scan layout) and ``nnfb-features extract`` crops each organ at extract time. It uses
    ``load_scans``, ``scan_paths`` and ``organ_classes`` instead of ``case_inputs``.
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
    # True: the exported mask keeps the values of ``roi_label`` (e.g. 1 = GTVp, 2 = GTVn)
    # so each ROI can be pooled on its own; the crop is still centred on their union.
    # False: the mask is binary (union).
    keep_roi_labels: bool = False
    # Names of the kept ROI labels, e.g. ((1, "gtvp"), (2, "gtvn")): frozen features are
    # then also pooled per ROI (``<name>_mean``) and as their concatenation.
    roi_names: tuple[tuple[int, str], ...] = ()
    # Image values below this are set to it before z-scoring (e.g. CT scanner padding far
    # below air, D19). None: no clamping.
    intensity_min: float | None = None
    # Name of a prepared task whose crops, masks and features this task reuses: it
    # differs only in its splits (``official_variant``). None: a task with its own data.
    split_of: str | None = None
    # Organ task (D38): scan layout, per-organ samples, multi-class labels.
    organ_task: bool = False
    # Organ tasks only. () -> one row per scan: ``scan_id`` plus metadata columns
    # (``split`` = the dataset's official split).
    load_scans: Callable[[], pd.DataFrame] | None = None
    # Organ tasks only. scan_id -> (image, folder of ``<structure>.nii.gz`` masks).
    scan_paths: Callable[[str], tuple[Path, Path]] | None = None
    # Organ tasks only. () -> the class set: structures that become samples.
    organ_classes: Callable[[], tuple[str, ...]] | None = None

    @property
    def is_multiclass(self) -> bool:
        return self.organ_task


def official_variant(task: ClassificationTask) -> ClassificationTask:
    """``<task>_official``: TASK's data with the dataset's own train/test split as its
    single fold (``official_split`` column of the labels; val held out from train). Its
    data/processed folder links TASK's images, masks and features (D37)."""
    return replace(
        task,
        name=f"{task.name}_official",
        description=f"{task.description}; official train/test split",
        split_of=task.name,
    )

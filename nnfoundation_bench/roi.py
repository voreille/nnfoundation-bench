"""ROI masks from TotalSegmentator, used only to position the classification crop."""

import os
import tempfile
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


FAST_SPACING_MM = 3.0  # working resolution of TotalSegmentator's fast model


def resample_isotropic(image, spacing: float, pad_value: float = -1024):
    """Resample a SimpleITK image to isotropic ``spacing`` over the same extent."""
    import SimpleITK as sitk

    extent = zip(image.GetSize(), image.GetSpacing(), strict=True)
    size = [max(1, round(n * s / spacing)) for n, s in extent]
    return sitk.Resample(
        image, size, sitk.Transform(), sitk.sitkLinear, image.GetOrigin(), [spacing] * 3,
        image.GetDirection(), pad_value, image.GetPixelID(),
    )  # fmt: skip


def segment_rois(
    image: Path, output: Path, rois: list[str], device: str = "gpu", fast: bool = True
) -> None:
    """Write a multi-label mask of ``rois`` for ``image``.

    ``fast`` uses the 3 mm model: coarse, but more than enough to centre a 192 mm crop.
    The image is then resampled to 3 mm with SimpleITK first and the mask stays on that
    grid (preprocessing resamples masks in physical space). TotalSegmentator would do the
    same resampling itself, but its full-resolution resampling in and out took ~40 s of a
    ~50 s STOIC case; on a 3 mm input the whole call takes ~8 s, with the same lung centre
    (< 2 mm apart). The input goes through an uncompressed temporary NIfTI next to
    ``output`` (TotalSegmentator reads NIfTI only; gzip alone took ~9 s). The mask is
    close to, not identical to, TotalSegmentator's native output: see D18 in
    docs/decisions.md.
    """
    import SimpleITK as sitk

    os.environ.setdefault("TOTALSEG_HOME_DIR", str(TOTALSEG_HOME_DIR))
    from totalsegmentator.python_api import totalsegmentator

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".partial.{output.name}")
    logger.debug(f"TotalSegmentator {rois} on {image.name}")
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".tmp.") as tmp_dir:
        source = sitk.ReadImage(image)
        if fast:
            source = resample_isotropic(source, FAST_SPACING_MM)
        nifti = Path(tmp_dir) / "image.nii"
        sitk.WriteImage(source, nifti, False)
        totalsegmentator(
            nifti,
            tmp,
            ml=True,
            fast=fast,
            roi_subset=None if fast else rois,
            device=device,
            quiet=True,
        )
    tmp.rename(output)

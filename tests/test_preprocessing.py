import numpy as np
import pandas as pd
import SimpleITK as sitk
from nnfoundation_bench.preprocessing import crop_around, roi_center, stratified_split


def _image(shape_zyx, spacing_xyz, fill=0.0):
    image = sitk.GetImageFromArray(np.full(shape_zyx, fill, dtype=np.float32))
    image.SetSpacing(spacing_xyz)
    image.SetOrigin((10.0, -5.0, 3.0))
    return image


def test_roi_center_matches_bbox_centre():
    mask = sitk.GetImageFromArray(np.zeros((20, 30, 40), dtype=np.uint8))
    mask.SetSpacing((2.0, 1.0, 3.0))
    array = sitk.GetArrayFromImage(mask)
    array[4:9, 10:21, 2:7] = 1  # z, y, x
    mask_with_roi = sitk.GetImageFromArray(array)
    mask_with_roi.CopyInformation(mask)
    assert np.allclose(roi_center(mask_with_roi, 1), (4 * 2.0, 15 * 1.0, 6 * 3.0))
    assert roi_center(mask, 1) is None


def test_crop_has_target_grid_and_is_centred():
    image = _image((50, 60, 70), (0.8, 0.8, 2.5))
    array = sitk.GetArrayFromImage(image)
    array[25, 30, 35] = 1000.0
    marked = sitk.GetImageFromArray(array)
    marked.CopyInformation(image)
    center = marked.TransformIndexToPhysicalPoint((35, 30, 25))

    crop = crop_around(marked, center, size=(32, 32, 32), spacing=(1.0, 1.0, 1.0))
    assert crop.GetSize() == (32, 32, 32)
    assert crop.GetSpacing() == (1.0, 1.0, 1.0)
    hot = np.unravel_index(np.argmax(sitk.GetArrayFromImage(crop)), (32, 32, 32))
    assert all(abs(h - 15.5) <= 1 for h in hot)


def test_crop_pads_outside_field_of_view():
    image = _image((10, 10, 10), (1.0, 1.0, 1.0), fill=50.0)
    corner = image.TransformIndexToPhysicalPoint((0, 0, 0))
    crop = sitk.GetArrayFromImage(crop_around(image, corner, size=(20, 20, 20), pad_value=-1024))
    assert crop.min() == -1024 and crop.max() == 50


def test_stratified_split_fractions_and_disjoint():
    labels = pd.DataFrame(
        {
            "case_id": [f"case_{i:03d}" for i in range(200)],
            "label": [i % 10 == 0 for i in range(200)],
        }
    )
    labels["label"] = labels["label"].astype(int)
    splits = stratified_split(labels, {"train": 0.4, "val": 0.1, "test": 0.5}, seed=0)
    sizes = {k: len(v) for k, v in splits.items()}
    assert sizes == {"train": 80, "val": 20, "test": 100}
    assert set().union(*splits.values()) == set(labels["case_id"])
    positives = set(labels.loc[labels["label"] == 1, "case_id"])
    assert len(positives & set(splits["test"])) == 10
    assert splits == stratified_split(labels, {"train": 0.4, "val": 0.1, "test": 0.5}, seed=0)

import json

import numpy as np
import pandas as pd
import SimpleITK as sitk

from nnfoundation_bench.preprocessing import (
    crop_around,
    repeated_splits,
    roi_center,
    stratified_split,
    write_b2nd,
    zscore,
)


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


def test_crop_reorients_permuted_direction_to_identity():
    # KiTS23 stores volumes with a permuted direction matrix.
    image = _image((40, 50, 60), (1.0, 1.0, 1.0))
    image.SetDirection((0, 0, 1, 0, 1, 0, -1, 0, 0))
    array = sitk.GetArrayFromImage(image)
    array[20, 25, 30] = 1000.0
    marked = sitk.GetImageFromArray(array)
    marked.CopyInformation(image)
    center = marked.TransformIndexToPhysicalPoint((30, 25, 20))

    crop = crop_around(marked, center, size=(16, 16, 16))
    assert crop.GetDirection() == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    hot = np.unravel_index(np.argmax(sitk.GetArrayFromImage(crop)), (16, 16, 16))
    assert all(abs(h - 7.5) <= 1 for h in hot)


def test_zscore():
    out = zscore(np.array([0, 10, 20]), mean=10.0, std=5.0)
    assert out.dtype == np.float32 and np.allclose(out, [-2, 0, 2])


def test_repeated_splits_differ_by_seed():
    labels = pd.DataFrame(
        {"case_id": [f"c{i}" for i in range(50)], "label": [i % 5 == 0 for i in range(50)]}
    )
    labels["label"] = labels["label"].astype(int)
    splits = repeated_splits(labels, n_splits=3, seed=0)
    assert len(splits) == 3 and splits[0]["test"] != splits[1]["test"]


def test_export_is_readable_by_glovita(tmp_path):
    from glovita.datasets.generic_3d_dataset import Generic3DDataset

    (tmp_path / "images").mkdir()
    volume = np.random.default_rng(0).normal(size=(1, 8, 9, 10)).astype(np.float32)
    write_b2nd(volume, tmp_path / "images" / "case_a.b2nd")
    (tmp_path / "labels.json").write_text(json.dumps({"case_a": 1}))
    (tmp_path / "splits.json").write_text(
        json.dumps([{"train": ["case_a"], "val": [], "test": []}])
    )

    dataset = Generic3DDataset(tmp_path, "train", fold="0")
    image, label = dataset[0]
    assert label == 1 and np.allclose(image.numpy(), volume)


def test_split_balances_centres_and_tolerates_tiny_strata():
    # Confounded centres like HECKTOR: A mostly positive, B mostly negative, C tiny.
    rows = [("A", 1)] * 30 + [("A", 0)] * 2 + [("B", 0)] * 28 + [("B", 1)] * 6 + [("C", 0)] * 1
    labels = pd.DataFrame(
        {
            "case_id": [f"c{i}" for i in range(len(rows))],
            "label": [lab for _, lab in rows],
            "stratum": [f"{c}_{lab}" for c, lab in rows],
        }
    )
    for seed in range(5):
        split = stratified_split(labels, seed=seed)
        test = labels[labels["case_id"].isin(split["test"])]
        assert abs((test["stratum"] == "A_1").sum() - 15) <= 1
        assert abs((test["stratum"] == "B_0").sum() - 14) <= 1

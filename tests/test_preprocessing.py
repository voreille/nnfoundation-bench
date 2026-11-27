import json

import numpy as np
import pandas as pd
import pytest
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


def test_resample_isotropic_keeps_extent():
    import SimpleITK as sitk

    from nnfoundation_bench.roi import resample_isotropic

    image = sitk.Image([512, 512, 100], sitk.sitkInt16)
    image.SetSpacing((0.6, 0.6, 2.0))
    image.SetOrigin((-150.0, -150.0, -50.0))
    low = resample_isotropic(image, 3.0)
    assert low.GetSize() == (102, 102, 67)
    assert low.GetSpacing() == (3.0, 3.0, 3.0) and low.GetOrigin() == image.GetOrigin()


def test_binary_roi_can_keep_labels():
    import numpy as np
    import SimpleITK as sitk

    from nnfoundation_bench.preprocessing import binary_roi

    mask = sitk.GetImageFromArray(np.array([[[0, 1, 2, 3]]], dtype=np.uint8))
    union = sitk.GetArrayFromImage(binary_roi(mask, (1, 2)))
    kept = sitk.GetArrayFromImage(binary_roi(mask, (1, 2), keep_labels=True))
    assert union.tolist() == [[[0, 1, 1, 0]]]
    assert kept.tolist() == [[[0, 1, 2, 0]]]


def test_official_split_keeps_test_and_holds_out_val():
    import pandas as pd

    from nnfoundation_bench.preprocessing import official_split

    labels = pd.DataFrame(
        {
            "case_id": [f"c{i}" for i in range(30)],
            "label": [i % 2 for i in range(30)],
            "official_split": ["train"] * 20 + ["test"] * 10,
        }
    )
    split = official_split(labels, val_fraction=0.2)
    assert split["test"] == sorted(f"c{i}" for i in range(20, 30))
    assert len(split["val"]) == 4 and len(split["train"]) == 16
    assert not set(split["train"]) & set(split["val"])


def test_split_variant_links_parent_data_and_keeps_one_official_fold(tmp_path, monkeypatch):
    from dataclasses import replace

    from nnfoundation_bench import preprocessing
    from nnfoundation_bench.datasets import get_task
    from nnfoundation_bench.datasets.base import official_variant

    processed, interim = tmp_path / "processed", tmp_path / "interim"
    monkeypatch.setattr(preprocessing, "PROCESSED_DATA_DIR", processed)
    monkeypatch.setattr(preprocessing, "INTERIM_DATA_DIR", interim)
    cases = [f"A-{i:03d}" for i in range(20)] + [f"B-{i:03d}" for i in range(10)]
    raw = pd.DataFrame(
        {
            "case_id": cases,
            "label": [i % 2 for i in range(30)],
            "official_split": ["train"] * 20 + ["test"] * 10,
        }
    )
    parent = replace(get_task("hecktor22_rfs_pet"), load_labels=lambda: raw)
    parent_dir = processed / parent.name
    for name in ("images", "masks"):
        (parent_dir / name).mkdir(parents=True)
    (interim / parent.name).mkdir(parents=True)
    usable = {c: int(y) for c, y in zip(raw["case_id"][1:], raw["label"][1:], strict=True)}
    (parent_dir / "labels.json").write_text(json.dumps(usable))  # A-000: no ROI
    raw.drop(columns="official_split").to_csv(parent_dir / "labels.csv", index=False)
    (parent_dir / "dataset.json").write_text(json.dumps({"num_classes": 2, "name": "x"}))

    task = official_variant(parent)
    assert task.name == "hecktor22_rfs_pet_official" and task.split_of == parent.name
    for _ in range(2):  # rerunnable
        preprocessing.prepare_task(task)

    out = processed / task.name
    for name in ("images", "masks", "features", "radiomics"):
        assert (out / name).is_symlink() and (out / name).resolve() == (parent_dir / name)
    assert (interim / task.name).resolve() == interim / parent.name
    (split,) = json.loads((out / "splits.json").read_text())
    assert split["test"] == cases[20:]
    assert "A-000" not in split["train"] + split["val"]
    assert len(split["train"]) + len(split["val"]) == 19 and len(split["val"]) == 4
    assert json.loads((out / "labels.json").read_text()) == usable
    assert "official_split" in pd.read_csv(out / "labels.csv").columns
    summary = json.loads((out / "dataset.json").read_text())
    assert summary["name"] == task.name and summary["split_of"] == parent.name


def test_official_variants_are_registered():
    from nnfoundation_bench.datasets import get_task

    for name in ("hecktor22_rfs_pet", "hecktor22_rfs_ct", "hecktor2022_rfs_pet"):
        task = get_task(f"{name}_official")
        assert task.split_of == name and task.roi_names == get_task(name).roi_names


# ------------------------------------------------------------ organ tasks (issue #4)


def _write_mask(path, array, spacing=(1.5, 1.5, 2.0), origin=(1.0, 2.0, 3.0)):
    image = sitk.GetImageFromArray(array.astype(np.uint8))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    sitk.WriteImage(image, path)


def test_merge_label_map_uses_the_id_table_and_counts_overlap(tmp_path):
    from nnfoundation_bench.preprocessing import merge_label_map

    shape = (6, 8, 10)
    a, b, c = (np.zeros(shape, np.uint8) for _ in range(3))
    a[0:2, 0:3, 0:4] = 1
    b[1:3, 2:5, 3:6] = 1  # shares the voxel (z 1, y 2, x 3) with a
    b[0:2, 0:3, 0] = 1  # and 6 more voxels
    c[4:6, 6:8, 8:10] = 1
    overlap = int((a & b).sum())
    for name, array in (("alpha", a), ("beta", b), ("gamma", c)):
        _write_mask(tmp_path / f"{name}.nii.gz", array)

    label_map, n_overlap = merge_label_map(tmp_path, ("alpha", "beta", "gamma"))
    out = sitk.GetArrayFromImage(label_map)
    assert label_map.GetPixelID() == sitk.sitkUInt8
    assert label_map.GetSpacing() == (1.5, 1.5, 2.0) and label_map.GetOrigin() == (1.0, 2.0, 3.0)
    assert n_overlap == overlap == 7
    assert (out[a.astype(bool)] == 1).all()  # overlap goes to the first class
    assert (out[b.astype(bool) & ~a.astype(bool)] == 2).all()
    assert (out[c.astype(bool)] == 3).all()
    assert (out[~(a | b | c).astype(bool)] == 0).all()


def _marked_image(direction):
    array = np.full((16, 24, 20), -100.0, np.float32)  # z, y, x
    array[6:10, 10:14, 8:12] = 700.0
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.5, 1.25, 2.0))
    image.SetOrigin((-10.0, 4.0, 7.0))
    image.SetDirection(direction)
    return image


def _corners(image):
    size = np.array(image.GetSize()) - 1
    return np.array(
        [
            image.TransformContinuousIndexToPhysicalPoint((size * np.array(c)).tolist())
            for c in np.ndindex(2, 2, 2)
        ]
    )


def test_resample_whole_gives_1mm_lps_grid_covering_the_scan():
    from nnfoundation_bench.preprocessing import IDENTITY, resample_whole

    image = _marked_image((0, 0, 1, 0, 1, 0, -1, 0, 0))  # permuted axes, as in KiTS
    out = resample_whole(image, interpolator=sitk.sitkNearestNeighbor, pad_value=-1024)
    assert out.GetDirection() == IDENTITY
    assert out.GetSpacing() == (1.0, 1.0, 1.0)
    lo, hi = _corners(image).min(0), _corners(image).max(0)
    out_lo, out_hi = _corners(out).min(0), _corners(out).max(0)
    assert np.all(out_lo <= lo + 1.0) and np.all(out_hi >= hi - 1.0)
    assert np.all(out_lo >= lo - 1.5) and np.all(out_hi <= hi + 1.5)  # no large margin
    # The marked block is at the same physical place.
    centre = image.TransformContinuousIndexToPhysicalPoint((9.5, 11.5, 7.5))
    assert out.GetPixel(out.TransformPhysicalPointToIndex(centre)) == 700.0
    values = set(np.unique(sitk.GetArrayFromImage(out)).tolist())
    assert values == {-100.0, 700.0}  # axis-aligned: no padding needed


def test_resample_whole_pads_outside_a_rotated_field_of_view():
    from nnfoundation_bench.preprocessing import IDENTITY, resample_whole

    angle = np.deg2rad(30)
    c, s = np.cos(angle), np.sin(angle)
    image = _marked_image((c, -s, 0, s, c, 0, 0, 0, 1))
    for pad in (-1024.0, 0.0):  # CT air, MRI 0
        out = resample_whole(image, interpolator=sitk.sitkLinear, pad_value=pad)
        array = sitk.GetArrayFromImage(out)
        assert out.GetDirection() == IDENTITY
        assert array[0, 0, 0] == pad and array[-1, -1, -1] == pad  # bbox corners
        assert array.max() == pytest.approx(700.0)


def _samples(n_scans=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_scans):
        for organ in ("aorta", "heart", "liver", "spleen")[: rng.integers(1, 5)]:
            rows.append({"case_id": f"s{i:04d}__{organ}", "scan_id": f"s{i:04d}"})
    return pd.DataFrame(rows)


def _scan_sets(split):
    return {k: {c.split("__")[0] for c in v} for k, v in split.items()}


def test_scan_splits_keep_every_scan_in_one_split():
    from nnfoundation_bench.preprocessing import scan_splits

    samples = _samples()
    split = scan_splits(samples, fractions=SPLIT_FRACTIONS_, seed=3)
    assert set(split) == {"train", "val", "test"}
    cases = [c for v in split.values() for c in v]
    assert len(cases) == len(set(cases)) and set(cases) == set(samples["case_id"])
    scans = _scan_sets(split)
    assert not scans["train"] & scans["val"] and not scans["train"] & scans["test"]
    assert not scans["val"] & scans["test"]
    # Fractions are on scans (20): 8 / 2 / 10.
    assert abs(len(scans["train"]) - 8) <= 1 and abs(len(scans["val"]) - 2) <= 1
    assert abs(len(scans["test"]) - 10) <= 1
    for k, v in split.items():  # every sample of a scan in the scan's split
        expected = set(samples.loc[samples["scan_id"].isin(scans[k]), "case_id"])
        assert set(v) == expected


def test_scan_splits_are_deterministic_per_seed():
    from nnfoundation_bench.preprocessing import scan_splits

    samples = _samples()
    a = scan_splits(samples, fractions=SPLIT_FRACTIONS_, seed=1)
    assert a == scan_splits(samples, fractions=SPLIT_FRACTIONS_, seed=1)
    assert a["test"] != scan_splits(samples, fractions=SPLIT_FRACTIONS_, seed=2)["test"]


SPLIT_FRACTIONS_ = {"train": 0.4, "val": 0.1, "test": 0.5}


def test_official_scan_split_uses_its_own_val_when_present():
    from nnfoundation_bench.preprocessing import official_scan_split

    samples = _samples(12)
    official = {f"s{i:04d}": "train" if i < 7 else "val" if i < 9 else "test" for i in range(12)}
    samples["official_split"] = samples["scan_id"].map(official)
    split = official_scan_split(samples, seed=0)
    scans = _scan_sets(split)
    assert scans == {
        "train": {s for s, v in official.items() if v == "train"},
        "val": {s for s, v in official.items() if v == "val"},
        "test": {s for s, v in official.items() if v == "test"},
    }
    assert sorted(c for v in split.values() for c in v) == sorted(samples["case_id"])


def test_official_scan_split_holds_out_train_scans_without_val():
    from nnfoundation_bench.preprocessing import official_scan_split

    samples = _samples(15)
    official = {f"s{i:04d}": "train" if i < 10 else "test" for i in range(15)}
    samples["official_split"] = samples["scan_id"].map(official)
    split = official_scan_split(samples, val_fraction=0.2, seed=0)
    scans = _scan_sets(split)
    assert scans["test"] == {s for s, v in official.items() if v == "test"}
    assert len(scans["val"]) == 2 and len(scans["train"]) == 8  # 20% of 10 train scans
    assert scans["train"] | scans["val"] == {s for s, v in official.items() if v == "train"}
    assert not scans["train"] & scans["val"]
    for k, v in split.items():
        assert set(v) == set(samples.loc[samples["scan_id"].isin(scans[k]), "case_id"])
    assert split == official_scan_split(samples, val_fraction=0.2, seed=0)


def test_prepare_task_for_a_crop_task_is_unchanged(tmp_path, monkeypatch):
    """Regression: binary crop tasks still write 192^3 crops per case."""
    from dataclasses import replace

    from nnfoundation_bench import preprocessing
    from nnfoundation_bench.datasets import get_task
    from nnfoundation_bench.datasets.base import CaseInputs

    monkeypatch.setattr(preprocessing, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(preprocessing, "INTERIM_DATA_DIR", tmp_path / "interim")
    raw = tmp_path / "raw"
    raw.mkdir()
    cases = [f"case_{i:02d}" for i in range(20)]
    for case in cases:
        array = np.full((10, 10, 10), -500.0, np.float32)
        array[3:6, 3:6, 3:6] = 100.0
        mask = (array > 0).astype(np.uint8)
        for name, a in (("img", array), ("mask", mask)):
            image = sitk.GetImageFromArray(a)
            image.SetSpacing((2.0, 2.0, 2.0))
            sitk.WriteImage(image, raw / f"{case}_{name}.nii.gz")
    labels = pd.DataFrame({"case_id": cases, "label": [i % 2 for i in range(20)]})
    task = replace(
        get_task("kits_liver"),
        load_labels=lambda: labels,
        case_inputs=lambda c, d: CaseInputs(raw / f"{c}_img.nii.gz", raw / f"{c}_mask.nii.gz"),
    )
    preprocessing.prepare_task(task, device="cpu", seed=0, n_splits=1)
    out = tmp_path / "processed" / task.name
    image = _read(out / "images" / "case_00.b2nd")
    mask = _read(out / "masks" / "case_00.b2nd")
    assert image.shape == (1, 192, 192, 192) and image.dtype == np.float32
    assert mask.shape == (1, 192, 192, 192) and mask.dtype == np.uint8
    assert json.loads((out / "labels.json").read_text()) == dict(
        zip(cases, labels["label"], strict=True)
    )
    summary = json.loads((out / "dataset.json").read_text())
    assert summary["num_classes"] == 2 and summary["num_positive"] == 10
    assert summary.get("layout", "crop") != "scan"
    assert not (out / "scans.csv").exists() and not (out / "organs.csv").exists()
    (split,) = json.loads((out / "splits.json").read_text())
    assert {k: len(v) for k, v in split.items()} == {"train": 8, "val": 2, "test": 10}


def _read(path):
    import blosc2

    return blosc2.open(str(path), mode="r")[...]


# ------------------------------------------------------------ non-orthonormal NIfTI headers


def _nifti_with_affine(path, array_xyz, affine):
    import nibabel as nib

    image = nib.Nifti1Image(array_xyz, affine)
    image.set_qform(None, code=0)  # TotalSegmentator v2.0.1: sform only
    image.set_sform(affine, code=2)
    nib.save(image, str(path))


def _sloppy_affine():
    """1.5 mm, rotated, with float32-like rounding: directions off orthonormal by ~2e-4."""
    angle = np.deg2rad(10)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotation[0, 1] += 2e-4
    affine = np.eye(4)
    affine[:3, :3] = rotation * 1.5
    affine[:3, 3] = (-100.0, 50.0, 20.0)
    return affine


def test_read_image_accepts_nearly_orthonormal_directions(tmp_path):
    """Regression: 3 of 4 TotalSegmentator CT scans failed with 'ITK only supports
    orthonormal direction cosines'; such files are read with the nearest orthonormal
    direction, voxel values and physical positions unchanged (to 0.1 mm)."""
    from nnfoundation_bench.preprocessing import read_image

    array_xyz = np.arange(6 * 5 * 4, dtype=np.int16).reshape(6, 5, 4)
    affine = _sloppy_affine()
    path = tmp_path / "sloppy.nii.gz"
    _nifti_with_affine(path, array_xyz, affine)
    with pytest.raises(RuntimeError, match="orthonormal"):
        sitk.ReadImage(str(path))  # precondition: plain SimpleITK refuses it

    image = read_image(path)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(image), array_xyz.transpose(2, 1, 0))
    direction = np.array(image.GetDirection()).reshape(3, 3)
    np.testing.assert_allclose(direction.T @ direction, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(image.GetSpacing(), np.linalg.norm(affine[:3, :3], axis=0))
    lps = np.diag([-1.0, -1.0, 1.0])
    for index in ((0, 0, 0), (5, 4, 3), (5, 0, 3)):
        expected = lps @ (affine[:3, :3] @ np.array(index) + affine[:3, 3])
        got = np.array(image.TransformIndexToPhysicalPoint(index))
        np.testing.assert_allclose(got, expected, atol=0.1)


def test_read_image_is_simpleitk_on_valid_files(tmp_path):
    from nnfoundation_bench.preprocessing import read_image

    image = sitk.GetImageFromArray(np.arange(60, dtype=np.float32).reshape(3, 4, 5))
    image.SetSpacing((0.7, 0.8, 2.5))
    image.SetOrigin((1.0, -2.0, 3.0))
    image.SetDirection((0, 0, 1, 0, 1, 0, -1, 0, 0))
    sitk.WriteImage(image, tmp_path / "ok.nii.gz")
    expected, got = sitk.ReadImage(tmp_path / "ok.nii.gz"), read_image(tmp_path / "ok.nii.gz")
    np.testing.assert_array_equal(sitk.GetArrayFromImage(got), sitk.GetArrayFromImage(expected))
    assert got.GetSpacing() == expected.GetSpacing()
    assert got.GetOrigin() == expected.GetOrigin()
    assert got.GetDirection() == expected.GetDirection()

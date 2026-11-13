import json

import numpy as np
import pytest
import SimpleITK as sitk

from nnfoundation_bench.modeling import radiomics_baseline as rb


def test_discretisation_per_modality():
    assert rb.extractor_settings("CT")["binWidth"] == 25.0
    assert rb.extractor_settings("PT")["binWidth"] == 0.5
    assert rb.extractor_settings("MR")["binCount"] == 32
    with pytest.raises(ValueError):
        rb.extractor_settings("US")


def test_to_real_drops_zero_imaginary_part_only():
    assert rb.to_real("f", np.complex128(0.77 + 0j)) == pytest.approx(0.77)
    assert rb.to_real("f", np.array(3.0)) == 3.0
    with pytest.raises(ValueError):
        rb.to_real("f", 1 + 0.5j)


def _sphere(tmp_path, radius=8):
    z, y, x = np.mgrid[:32, :32, :32]
    inside = ((z - 16) ** 2 + (y - 16) ** 2 + (x - 16) ** 2) <= radius**2
    rng = np.random.default_rng(0)
    image = sitk.GetImageFromArray(
        (inside * 40 + rng.normal(0, 5, inside.shape)).astype(np.float32)
    )
    mask = sitk.GetImageFromArray(inside.astype(np.uint8))
    for img, name in ((image, "img.nii.gz"), (mask, "mask.nii.gz")):
        img.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(img, tmp_path / name)
    return tmp_path / "img.nii.gz", tmp_path / "mask.nii.gz", inside.sum()


def test_extract_case_on_a_sphere(tmp_path):
    image, mask, n_voxels = _sphere(tmp_path)
    feats = rb.extract_case(image, mask, rb.extractor_settings("CT"), rb.FEATURE_SETS["original"])
    assert len(feats) > 100 and all(isinstance(v, float) for v in feats.values())
    assert feats["original_shape_VoxelVolume"] == pytest.approx(n_voxels)
    assert feats["original_shape_Elongation"] == pytest.approx(1.0, abs=0.05)
    assert feats["original_firstorder_Mean"] == pytest.approx(40, abs=1)


def test_empty_roi_gives_no_features_and_is_imputed(tmp_path, monkeypatch):
    image, _, _ = _sphere(tmp_path)
    empty = sitk.Image(32, 32, 32, sitk.sitkUInt8)
    sitk.WriteImage(empty, tmp_path / "empty.nii.gz")
    settings = rb.extractor_settings("CT")
    assert rb.extract_case(image, tmp_path / "empty.nii.gz", settings, {"Original": {}}) == {}

    monkeypatch.setattr(rb, "PROCESSED_DATA_DIR", tmp_path)
    folder = rb.radiomics_dir("t", "original")
    folder.mkdir(parents=True)
    (folder / "a.json").write_text(json.dumps({"f1": 1.0, "f2": float("inf")}))
    (folder / "b.json").write_text(json.dumps({}))
    x, names = rb.load_features("t", "original", ["a", "b"])
    assert names == ["f1", "f2"]
    assert x[0, 0] == 1.0 and np.isnan(x[0, 1]) and np.isnan(x[1]).all()


def test_design_table_radiomics_rows_have_one_value():
    import pandas as pd

    from nnfoundation_bench.modeling.predict import design_table

    lines = design_table({"t": pd.Series({"radiomics_original_logreg": "0.70 ± 0.05"})})
    assert "| R1 | 0.70 ± 0.05 |" in lines.splitlines()


def test_run_names_keep_r1_r2_and_encode_options():
    assert rb.run_name("original") == "radiomics_original_logreg"
    assert rb.run_name("filtered", 0.9, "enet") == "radiomics_filtered_corr0.9_enet"
    assert rb.run_name("filtered", None, "enet") == "radiomics_filtered_enet"


def test_correlation_filter_drops_redundant_and_constant_features():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=50), rng.normal(size=50)
    x = np.column_stack([
        a,
        np.exp(a),  # monotonic in a: Spearman rho = 1
        b,
        b + rng.normal(0, 0.01, 50),  # nearly b
        np.full(50, 3.0),  # constant
        rng.normal(size=50),
    ])  # fmt: skip
    x[0, 2] = np.nan  # imputed, not fatal
    keep = rb.correlation_filter(x, 0.9)
    assert len(keep) == 3 and 4 not in keep and 5 in keep
    assert len({0, 1} & set(keep)) == 1 and len({2, 3} & set(keep)) == 1


def test_elastic_net_grid_selects_an_l1_ratio():
    from nnfoundation_bench.modeling.features import fit_probe

    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 30)
    x = rng.normal(size=(60, 20))
    x[:, 0] += 2 * y
    idx = np.arange(60)
    split = {"train": idx[::2], "val": idx[1::4], "test": idx[3::4]}
    _, params, val_auc = fit_probe(x, y, split, l1_ratios=rb.ENET_L1_RATIOS)
    assert params["l1_ratio"] in rb.ENET_L1_RATIOS and val_auc > 0.8

"""Audit tests for the TotalSegmentator organ tasks (issue #4): edge cases of the crops,
prepare reruns, extract resuming, official variants, probes and CLI guards. Synthetic
data only (no real data, no network, no GPU; the encoder is faked)."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from nnfoundation_bench.modeling import features

CT_TASK = "totalseg_ct_organs"


def _check_mapping(image, mask, organ, crop, organ_mask, size, pad):
    """Every crop voxel is image[start + i] (pad outside); returns the start offsets."""
    nz_orig = np.nonzero(mask == organ)
    nz_crop = np.nonzero(organ_mask)
    start = np.array([nz_orig[a].min() - nz_crop[a].min() for a in range(3)])
    idx = np.indices((size,) * 3).reshape(3, -1).T + start
    inside = np.all((idx >= 0) & (idx < np.array(image.shape)), axis=1)
    expected = np.full(len(idx), pad, np.float32)
    expected[inside] = image[tuple(idx[inside].T)]
    np.testing.assert_array_equal(crop.reshape(-1), expected)
    exp_mask = np.zeros(len(idx), bool)
    exp_mask[inside] = mask[tuple(idx[inside].T)] == organ
    np.testing.assert_array_equal(organ_mask.reshape(-1).astype(bool), exp_mask)
    return start


# ---------------------------------------------------------------- organ_crop


def test_organ_crop_centres_on_bbox_not_centre_of_mass():
    from nnfoundation_bench.modeling.features import organ_crop

    shape = (30, 30, 30)
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    mask = np.zeros(shape, np.uint8)
    # L-shaped organ: bbox z 4..23 (centre 13.5), but most voxels at z 4..7.
    mask[4:8, 10:20, 10:20] = 2
    mask[8:24, 10:12, 10:12] = 2
    com_z = np.nonzero(mask == 2)[0].mean()
    assert abs(com_z - 13.5) > 3  # the two centres really differ
    size = 24
    crop, organ_mask, inside = organ_crop(image, mask, 2, size=size, pad=-1.0)
    start = _check_mapping(image, mask, 2, crop, organ_mask, size, -1.0)
    # bbox centre 13.5 sits exactly at the crop centre 11.5: start = 2 (centre of mass
    # ~7 would give start ~ -5).
    assert start[0] == 2
    # y/x bbox 10..19 (centre 14.5): 14.5 - 11.5 = 3.
    assert start[1] == 3 and start[2] == 3
    assert inside == 1.0


@pytest.mark.parametrize(("lo", "hi", "size"), [(5, 8, 8), (5, 9, 8), (5, 8, 7), (5, 9, 7)])
def test_organ_crop_odd_and_even_extents_put_the_bbox_centre_at_the_crop_centre(lo, hi, size):
    from nnfoundation_bench.modeling.features import organ_crop

    shape = (20, 20, 20)
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    mask = np.zeros(shape, np.uint8)
    mask[lo:hi, lo:hi, lo:hi] = 1
    crop, organ_mask, _ = organ_crop(image, mask, 1, size=size, pad=0.0)
    start = _check_mapping(image, mask, 1, crop, organ_mask, size, 0.0)
    bbox_centre = (lo + hi - 1) / 2
    crop_centre = start[0] + (size - 1) / 2
    assert abs(bbox_centre - crop_centre) <= 0.5
    before = lo - start[0]
    after = start[0] + size - hi
    assert abs(before - after) <= 1


def test_organ_crop_scan_smaller_than_crop_and_organ_touching_both_borders():
    from nnfoundation_bench.modeling.features import organ_crop

    shape = (6, 20, 3)  # z and x smaller than the crop
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1.0
    mask = np.zeros(shape, np.uint8)
    mask[0:6, 4:9, 0:3] = 5  # touches both z borders and both x borders
    crop, organ_mask, inside = organ_crop(image, mask, 5, size=10, pad=-7.0)
    assert crop.shape == organ_mask.shape == (10, 10, 10)
    assert inside == 1.0 and int(organ_mask.sum()) == int((mask == 5).sum())
    _check_mapping(image, mask, 5, crop, organ_mask, 10, -7.0)
    assert (crop == -7.0).sum() == 1000 - 6 * 10 * 3


def test_organ_crop_organ_longer_than_crop_on_both_sides():
    from nnfoundation_bench.modeling.features import organ_crop

    shape = (30, 10, 10)
    image = np.ones(shape, np.float32)
    mask = np.zeros(shape, np.uint8)
    mask[:, 4:6, 4:6] = 1  # spans the whole z axis: 30 slices
    _crop, organ_mask, inside = organ_crop(image, mask, 1, size=8, pad=0.0)
    assert inside == pytest.approx(8 / 30)
    assert int(organ_mask.sum()) == 8 * 4


def test_organ_crop_with_find_objects_bbox_and_lazy_volume_match_plain(tmp_path):
    from scipy.ndimage import find_objects

    from nnfoundation_bench.modeling.features import LazyVolume, organ_crop
    from nnfoundation_bench.preprocessing import write_b2nd

    rng = np.random.default_rng(0)
    image = rng.normal(size=(25, 22, 19)).astype(np.float32)
    mask = np.zeros(image.shape, np.uint8)
    mask[1:5, 2:20, 15:19] = 3
    mask[10:12, 0:4, 0:2] = 1
    write_b2nd(image[None], tmp_path / "img.b2nd")
    lazy = LazyVolume(tmp_path / "img.b2nd")
    assert lazy.shape == image.shape
    plain = organ_crop(image, mask, 3, size=12, pad=-2.0)
    boxed = organ_crop(lazy, mask, 3, size=12, pad=-2.0, bbox=find_objects(mask)[2])
    for a, b in zip(plain[:2], boxed[:2], strict=True):
        np.testing.assert_array_equal(a, b)
    assert plain[2] == boxed[2]


def test_organ_crop_absent_organ_raises():
    from nnfoundation_bench.modeling.features import organ_crop

    with pytest.raises(ValueError):
        organ_crop(np.zeros((5, 5, 5), np.float32), np.zeros((5, 5, 5), np.uint8), 1, size=4)


# ---------------------------------------------------------------- merge / resample


def test_merge_label_map_voxel_in_three_masks_goes_to_the_first(tmp_path):
    from nnfoundation_bench.preprocessing import merge_label_map

    for name in ("a", "b", "c"):
        array = np.zeros((4, 4, 4), np.uint8)
        array[1, 1, 1] = 1
        sitk.WriteImage(sitk.GetImageFromArray(array), tmp_path / f"{name}.nii.gz")
    label_map, overlap = merge_label_map(tmp_path, ("b", "c", "a"))
    out = sitk.GetArrayFromImage(label_map)
    assert out[1, 1, 1] == 1  # "b": first in the given class order
    assert overlap >= 1


def test_resample_whole_flipped_anisotropic_keeps_physical_positions():
    from nnfoundation_bench.preprocessing import IDENTITY, resample_whole

    array = np.zeros((6, 30, 40), np.float32)  # z, y, x
    array[2:4, 10:15, 5:9] = 100.0
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.7, 0.8, 5.0))
    image.SetOrigin((12.0, -3.0, 40.0))
    image.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))  # RAS-like in-plane flip
    out = resample_whole(image, interpolator=sitk.sitkNearestNeighbor, pad_value=-1024)
    assert out.GetDirection() == IDENTITY and out.GetSpacing() == (1.0, 1.0, 1.0)
    extent = [(n - 1) * s for n, s in zip(image.GetSize(), image.GetSpacing(), strict=True)]
    for n, e in zip(out.GetSize(), extent, strict=True):
        assert e + 1 <= n <= e + 2
    centre = image.TransformContinuousIndexToPhysicalPoint((6.5, 12.0, 2.5))
    assert out.GetPixel(out.TransformPhysicalPointToIndex(centre)) == 100.0
    corner = image.TransformContinuousIndexToPhysicalPoint((1.0, 1.0, 1.0))
    assert out.GetPixel(out.TransformPhysicalPointToIndex(corner)) == 0.0


# ---------------------------------------------------------------- splits


def _samples(n_scans, organs=("aorta", "liver")):
    return pd.DataFrame(
        [{"case_id": f"s{i:04d}__{o}", "scan_id": f"s{i:04d}"}
         for i in range(n_scans) for o in organs]
    )  # fmt: skip


@pytest.mark.parametrize("n_scans", [2, 3])
def test_scan_splits_with_very_few_scans_do_not_crash(n_scans):
    """The issue plans a prepare smoke test on 2 real scans: the split must not crash."""
    from nnfoundation_bench.preprocessing import scan_splits

    samples = _samples(n_scans)
    split = scan_splits(samples, seed=0)
    cases = [c for v in split.values() for c in v]
    assert sorted(cases) == sorted(samples["case_id"])


def test_scan_splits_with_many_organs_per_scan_split_on_scans():
    """40/10/50 holds for scans, not samples, whatever the number of organs per scan."""
    from nnfoundation_bench.preprocessing import scan_splits

    rows = [
        {"case_id": f"s{i:04d}__o{k}", "scan_id": f"s{i:04d}"}
        for i in range(40)
        for k in range(1 if i < 20 else 20)
    ]
    split = scan_splits(pd.DataFrame(rows), seed=0)
    scans = {k: {c.split("__")[0] for c in v} for k, v in split.items()}
    assert {k: len(v) for k, v in scans.items()} == {"train": 16, "val": 4, "test": 20}


def test_official_scan_split_with_empty_test_and_partial_val():
    from nnfoundation_bench.preprocessing import official_scan_split

    samples = _samples(6)
    official = {"s0000": "train", "s0001": "train", "s0002": "train", "s0003": "val",
                "s0004": "train", "s0005": "train"}  # fmt: skip
    samples["official_split"] = samples["scan_id"].map(official)
    split = official_scan_split(samples, seed=0)
    assert split["test"] == []
    assert {c.split("__")[0] for c in split["val"]} == {"s0003"}
    assert len(split["train"]) == 10


# ---------------------------------------------------------------- multi-class probe


def test_multiclass_metrics_single_class_in_test_is_nan_and_json_like_binary():
    from nnfoundation_bench.modeling.predict import binary_metrics, multiclass_metrics

    labels = np.array([2, 2, 2])
    proba = np.array([[0.1, 0.1, 0.8], [0.5, 0.2, 0.3], [0.2, 0.2, 0.6]])
    m = multiclass_metrics(labels, proba, [0, 1, 2])
    assert np.isnan(m["auroc_macro"])
    assert m["accuracy"] == pytest.approx(2 / 3) and m["balanced_accuracy"] == pytest.approx(2 / 3)
    b = binary_metrics(np.array([1, 1]), np.array([0.2, 0.7]))
    # Same JSON convention as the binary metrics: NaN written as NaN.
    assert "NaN" in json.dumps(m) and ("NaN" in json.dumps(b)) == ("NaN" in json.dumps(m))
    assert np.isnan(json.loads(json.dumps(m))["auroc_macro"])


def test_fit_probe_multiclass_class_missing_from_train():
    from nnfoundation_bench.modeling.features import fit_probe

    rng = np.random.default_rng(3)
    y = np.repeat([0, 1, 2, 3], 20)
    x = rng.normal(size=(80, 6)) * 0.3
    x[np.arange(80), y] += 4.0
    idx = np.arange(80)
    train = idx[(idx % 20 < 10) & (y != 2)]  # class 2 never in train
    val = idx[(idx % 20 >= 10) & (idx % 20 < 13)]
    test = idx[idx % 20 >= 13]
    proba, _c, val_bacc = fit_probe(x, y, {"train": train, "val": val, "test": test},
                                   multiclass=True)  # fmt: skip
    assert proba.shape == (len(test), 4)
    np.testing.assert_allclose(proba[:, 2], 0.0)
    np.testing.assert_allclose(proba.sum(1), 1.0, rtol=1e-6)
    pred = proba.argmax(1)
    ok = y[test] != 2
    assert (pred[ok] == y[test][ok]).all()  # column k is class k despite the gap
    assert np.isfinite(val_bacc)


def _organ_layout(tmp_path, monkeypatch, missing_in_train=None):
    from nnfoundation_bench.modeling import predict, runs, survival

    rng = np.random.default_rng(0)
    organs = ("aorta", "heart", "liver")
    root = tmp_path / "processed" / CT_TASK
    (root / "features" / "cnn").mkdir(parents=True)
    rows = []
    for i in range(20):
        for label, organ in enumerate(organs):
            case = f"s{i:04d}__{organ}"
            rows.append({"case_id": case, "scan_id": f"s{i:04d}", "organ": organ,
                         "label": label})  # fmt: skip
            vec = rng.normal(size=5) * 0.3
            vec[label] += 4.0
            np.savez(root / "features" / "cnn" / f"{case}.npz", mean=vec)
    table = pd.DataFrame(rows).sort_values("case_id")
    table.to_csv(root / "labels.csv", index=False)
    (root / "labels.json").write_text(
        json.dumps(dict(zip(table["case_id"], table["label"].astype(int), strict=True)))
    )

    def cases(lo, hi):
        return sorted(c for c in table["case_id"] if lo <= int(c[1:5]) < hi)

    train = cases(0, 8)
    if missing_in_train:
        train = [c for c in train if not c.endswith(f"__{missing_in_train}")]
    (root / "splits.json").write_text(
        json.dumps([{"train": train, "val": cases(8, 10), "test": cases(10, 20)}])
    )
    for module in (features, predict, survival):
        monkeypatch.setattr(module, "PROCESSED_DATA_DIR", tmp_path / "processed")
    for module in (features, predict, runs):
        monkeypatch.setattr(module, "RESULTS_DIR", tmp_path / "results")
    return root


def test_organ_probe_with_class_missing_from_train_writes_valid_outputs(tmp_path, monkeypatch):
    _organ_layout(tmp_path, monkeypatch, missing_in_train="heart")
    features.probe(CT_TASK, model="cnn", pooling=["mean"])
    out = tmp_path / "results" / CT_TASK / "cnn_frozen_mean_logreg" / "fold0"
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["n_classes"] == 3 and metrics["n"] == 30
    # Heart is never predicted: recall 1, 0, 1.
    assert metrics["balanced_accuracy"] == pytest.approx(2 / 3)
    probs = np.load(out / "probs.npz", allow_pickle=False)
    assert probs["probs"].shape == (30, 3)
    np.testing.assert_allclose(probs["probs"][:, 1], 0.0)
    assert probs["case_id"].dtype.kind == "U" and probs["classes"].dtype.kind in "iu"
    pred = pd.read_csv(out / "predictions.csv")
    assert set(pred["pred"]) == {0, 2}


def test_organ_probe_is_deterministic(tmp_path, monkeypatch):
    _organ_layout(tmp_path, monkeypatch)
    out = tmp_path / "results" / CT_TASK / "cnn_frozen_mean_logreg" / "fold0"
    features.probe(CT_TASK, model="cnn", pooling=["mean"])
    first = (out / "predictions.csv").read_text()
    features.probe(CT_TASK, model="cnn", pooling=["mean"])
    assert (out / "predictions.csv").read_text() == first


def test_binary_fit_probe_matches_a_plain_sklearn_pipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from nnfoundation_bench.modeling.features import fit_probe

    rng = np.random.default_rng(5)
    y = np.repeat([0, 1], 30)
    x = rng.normal(size=(60, 4)) + y[:, None] * 0.8
    order = rng.permutation(60)
    split = {"train": order[:30], "val": order[30:40], "test": order[40:]}
    scores, params, val_auc = fit_probe(x, y, split)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=params["C"], class_weight="balanced", max_iter=5000),
    ).fit(x[split["train"]], y[split["train"]])
    np.testing.assert_allclose(scores, clf.predict_proba(x[split["test"]])[:, 1], rtol=1e-6)
    assert scores.ndim == 1 and isinstance(params, dict) and 0 <= val_auc <= 1


# ---------------------------------------------------------------- end-to-end prepare + extract

SIZE_XYZ = (20, 20, 14)
SPACING = (1.5, 1.5, 2.0)
STRUCTURES = ("aorta", "liver", "spleen")
VALUES = {"aorta": 300.0, "liver": 60.0, "spleen": 45.0}


def _boxes():
    return {
        "aorta": (slice(1, 13), slice(2, 7), slice(2, 7)),  # z, y, x; 1350 mm^3
        "liver": (slice(2, 8), slice(8, 16), slice(2, 10)),
        "spleen": (slice(2, 8), slice(8, 16), slice(12, 20)),
    }


def _write_scan(raw: Path, scan: str, corrupt: bool = False, missing: str | None = None):
    seg = raw / scan / "segmentations"
    seg.mkdir(parents=True, exist_ok=True)
    shape = SIZE_XYZ[::-1]
    image = np.full(shape, -1000.0, np.float32)
    for name, box in _boxes().items():
        mask = np.zeros(shape, np.uint8)
        mask[box] = 1
        image[box] = VALUES[name]
        path = seg / f"{name}.nii.gz"
        if name == missing:
            path.unlink(missing_ok=True)
            continue
        m = sitk.GetImageFromArray(mask)
        m.SetSpacing(SPACING)
        sitk.WriteImage(m, path)
    if corrupt:
        (raw / scan / "ct.nii.gz").write_bytes(b"not a nifti")
    else:
        img = sitk.GetImageFromArray(image)
        img.SetSpacing(SPACING)
        sitk.WriteImage(img, raw / scan / "ct.nii.gz")


@pytest.fixture
def organ_env(tmp_path, monkeypatch):
    from nnfoundation_bench import preprocessing
    from nnfoundation_bench.datasets import totalsegmentator as ts
    from nnfoundation_bench.modeling import predict, runs, survival

    raw = tmp_path / "raw"
    processed, interim = tmp_path / "processed", tmp_path / "interim"
    for module in (preprocessing, ts, features, predict, survival):
        monkeypatch.setattr(module, "PROCESSED_DATA_DIR", processed, raising=False)
    monkeypatch.setattr(preprocessing, "INTERIM_DATA_DIR", interim)
    for module in (features, predict, runs):
        monkeypatch.setattr(module, "RESULTS_DIR", tmp_path / "results")
    state = {"scans": [f"s{i:04d}" for i in range(1, 9)]}
    splits = ["train"] * 5 + ["test"] * 3

    def load_scans():
        return pd.DataFrame(
            {"scan_id": state["scans"], "age": 50, "split": splits[: len(state["scans"])]}
        )

    task = replace(
        ts.TOTALSEG_CT_ORGANS,
        load_scans=load_scans,
        scan_paths=lambda s: (raw / s / "ct.nii.gz", raw / s / "segmentations"),
        organ_classes=lambda: STRUCTURES,
    )
    for scan in state["scans"]:
        _write_scan(raw, scan)
    return task, raw, processed / CT_TASK, state


def _prepare(task, **kwargs):
    from nnfoundation_bench.preprocessing import prepare_task

    prepare_task(task, device="cpu", seed=0, n_splits=2, **kwargs)


def _fake_encoder(monkeypatch, seen):
    import torch

    def fake_feature_map(encoder, x):
        seen.append(x.detach().float().cpu().numpy()[0, 0])
        cells = torch.nn.functional.avg_pool3d(x.float(), 32)
        return torch.cat([cells, torch.ones_like(cells)], dim=1)

    monkeypatch.setattr(features, "build_encoder", lambda model, device: object())
    monkeypatch.setattr(features, "feature_map", fake_feature_map)


def test_prepare_then_extract_crops_each_organ_from_the_prepared_scans(organ_env, monkeypatch):
    task, _raw, out, _ = organ_env
    _prepare(task)
    seen = []
    _fake_encoder(monkeypatch, seen)
    features.extract(CT_TASK, model="cnn", device="cpu", save_maps=False, overwrite=False)
    labels = json.loads((out / "labels.json").read_text())
    assert len(labels) == 8 * 3 and len(seen) == 24
    scans = pd.read_csv(out / "scans.csv", dtype={"scan_id": str}).set_index("scan_id")
    pad_z = scans["pad_z"].iloc[0]
    for crop in seen:
        assert crop.shape == (192, 192, 192)
        assert crop[0, 0, 0] == pytest.approx(pad_z, abs=1e-4)
    # The crop centre lies in the organ: its physical value (z-scored back) is the organ's.
    mean, std = scans["mean"].iloc[0], scans["std"].iloc[0]
    centres = sorted(round(float(c[95, 95, 95] * std + mean)) for c in seen)
    assert centres == sorted([45, 60, 300] * 8)
    for case in labels:
        pooled = np.load(out / "features" / "cnn" / f"{case}.npz")
        assert float(pooled["roi_inside"]) == 1.0
        assert pooled["roi_mean"][1] == pytest.approx(1.0)


def test_extract_resumes_and_overwrite_recomputes(organ_env, monkeypatch):
    task, _raw, out, _ = organ_env
    _prepare(task)
    seen = []
    _fake_encoder(monkeypatch, seen)
    feats = out / "features" / "cnn"
    feats.mkdir(parents=True)
    done = "s0003__liver"
    np.savez(feats / f"{done}.npz", mean=np.zeros(2), marker=np.ones(1))
    features.extract(CT_TASK, model="cnn", device="cpu", save_maps=False, overwrite=False)
    assert len(seen) == 23
    assert "marker" in np.load(feats / f"{done}.npz").files  # left alone
    assert not list(feats.glob(".partial*"))
    seen.clear()
    features.extract(CT_TASK, model="cnn", device="cpu", save_maps=False, overwrite=False)
    assert seen == []
    features.extract(CT_TASK, model="cnn", device="cpu", save_maps=False, overwrite=True)
    assert len(seen) == 24
    assert "marker" not in np.load(feats / f"{done}.npz").files


def test_extract_on_official_variant_writes_into_the_parent_features(organ_env, monkeypatch):
    from nnfoundation_bench.datasets.base import official_variant

    task, _raw, out, _ = organ_env
    _prepare(task)
    _prepare(official_variant(task))
    variant = f"{CT_TASK}_official"
    (split,) = json.loads((out.parent / variant / "splits.json").read_text())
    scans = {k: {c.split("__")[0] for c in v} for k, v in split.items()}
    assert scans["test"] == {"s0006", "s0007", "s0008"}
    assert len(scans["val"]) == 1 and len(scans["train"]) == 4
    seen = []
    _fake_encoder(monkeypatch, seen)
    features.extract(variant, model="cnn", device="cpu", save_maps=False, overwrite=False)
    assert len(seen) == 24
    assert len(list((out / "features" / "cnn").glob("*.npz"))) == 24  # parent's folder
    seen.clear()
    features.extract(CT_TASK, model="cnn", device="cpu", save_maps=False, overwrite=False)
    assert seen == []  # computed once for both
    features.probe(variant, model="cnn", pooling=["roi_mean"])
    res = Path(features.RESULTS_DIR) / variant / "cnn_frozen_roi_mean_logreg" / "fold0"
    metrics = json.loads((res / "metrics.json").read_text())
    assert metrics["n"] == 9


def test_organ_absent_after_resampling_is_not_a_sample_and_extract_runs(organ_env, monkeypatch):
    """A structure file that is empty (0 voxels after resampling) stays in organs.csv only."""
    task, raw, out, _ = organ_env
    seg = raw / "s0002" / "segmentations" / "spleen.nii.gz"
    empty = sitk.GetImageFromArray(np.zeros(SIZE_XYZ[::-1], np.uint8))
    empty.SetSpacing(SPACING)
    sitk.WriteImage(empty, seg)
    _prepare(task)
    organs = pd.read_csv(out / "organs.csv", dtype={"scan_id": str})
    row = organs[(organs["scan_id"] == "s0002") & (organs["organ"] == "spleen")]
    assert row["voxels"].tolist() == [0]
    labels = json.loads((out / "labels.json").read_text())
    assert "s0002__spleen" not in labels and "s0002__liver" in labels
    seen = []
    _fake_encoder(monkeypatch, seen)
    features.extract(CT_TASK, model="cnn", device="cpu", save_maps=False, overwrite=False)
    assert len(seen) == 23


def test_prepare_rerun_picks_up_a_fixed_scan(organ_env):
    task, raw, out, _ = organ_env
    _write_scan(raw, "s0004", corrupt=True)
    _prepare(task)
    summary = json.loads((out / "dataset.json").read_text())
    assert "s0004" in summary["failed"]
    assert not any(c.startswith("s0004__") for c in json.loads((out / "labels.json").read_text()))
    _write_scan(raw, "s0004")
    _prepare(task)
    summary = json.loads((out / "dataset.json").read_text())
    assert summary["failed"] == {} or "s0004" not in summary["failed"]
    labels = json.loads((out / "labels.json").read_text())
    assert {"s0004__aorta", "s0004__liver", "s0004__spleen"} <= set(labels)
    splits = json.loads((out / "splits.json").read_text())
    assert all(sorted(c for v in s.values() for c in v) == sorted(labels) for s in splits)


def test_prepare_overwrite_reprocesses_every_scan(organ_env):
    task, _raw, out, _ = organ_env
    _prepare(task)
    image = out / "images" / "s0001.b2nd"
    mtime = image.stat().st_mtime_ns
    first = (out / "labels.json").read_text()
    _prepare(task, overwrite=True)
    assert image.stat().st_mtime_ns != mtime
    assert (out / "labels.json").read_text() == first


def test_prepare_overwrite_failure_is_not_resurrected_by_a_plain_rerun(organ_env):
    """A scan that fails under --overwrite must not come back from its stale outputs."""
    task, raw, out, _ = organ_env
    _prepare(task)
    _write_scan(raw, "s0004", corrupt=True)
    _prepare(task, overwrite=True)
    assert "s0004" in json.loads((out / "dataset.json").read_text())["failed"]
    _prepare(task)
    summary = json.loads((out / "dataset.json").read_text())
    labels = json.loads((out / "labels.json").read_text())
    assert "s0004" in summary["failed"] or not any(c.startswith("s0004__") for c in labels)


def test_prepare_with_a_first_scan_missing_a_structure_fails_clearly(organ_env):
    """The id table comes from the first scan: if that scan lacks a mask, every other scan
    fails. prepare must then stop with a clear message, not a pandas KeyError."""
    task, raw, out, _ = organ_env
    _write_scan(raw, "s0001", missing="spleen")
    try:
        _prepare(task)
    except KeyError as e:  # pragma: no cover - the failure we look for
        pytest.fail(f"bare KeyError from prepare: {e!r}")
    except Exception as e:  # noqa: BLE001 - any clear error is fine
        assert "s0001" in str(e) or "structure" in str(e).lower(), e
        return
    summary = json.loads((out / "dataset.json").read_text())
    assert "s0001" not in summary["failed"] or len(summary["failed"]) < 8


def test_prepare_with_few_scans_finishes(organ_env):
    """Smoke test on 2 scans (issue step 4): writes splits.json and dataset.json."""
    task, _raw, out, state = organ_env
    state["scans"] = ["s0001", "s0002"]
    _prepare(task)
    assert (out / "splits.json").exists() and (out / "dataset.json").exists()


# ---------------------------------------------------------------- guards, task lookup


def test_is_organ_task_on_unknown_fused_and_variant_tasks(tmp_path, monkeypatch):
    from nnfoundation_bench.modeling import runs

    monkeypatch.setattr(runs, "RESULTS_DIR", tmp_path)
    assert not runs.is_organ_task("no_such_task")
    assert runs.is_organ_task("totalseg_mri_organs_official")
    assert not runs.is_organ_task("kits_liver")
    for name, sources in (("fused_organ", [CT_TASK]), ("fused_bin", ["kits_liver"])):
        (tmp_path / name).mkdir()
        (tmp_path / name / "fusion.json").write_text(json.dumps({"tasks": sources}))
    assert runs.is_organ_task("fused_organ") and not runs.is_organ_task("fused_bin")


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    from nnfoundation_bench.modeling import cox_probe, predict, runs, survival

    for module in (features, predict, survival, cox_probe):
        monkeypatch.setattr(module, "PROCESSED_DATA_DIR", tmp_path / "processed")
    for module in (features, predict, runs):
        monkeypatch.setattr(module, "RESULTS_DIR", tmp_path / "results")
    (tmp_path / "results").mkdir()
    return tmp_path


def _refused(result):
    assert result.exit_code != 0, result.output
    text = f"{result.output} {result.exception}"
    assert "organ" in text.lower(), text


@pytest.mark.parametrize(
    ("cli", "args"),
    [
        ("cox", ["combine", CT_TASK, "--model", "radiomics"]),
        ("cox", ["probe", "totalseg_mri_organs", "--model", "covariates"]),
        ("features", ["combine", CT_TASK, "--model", "radiomics", "--column", "group"]),
        ("features", ["fuse", "kits_liver", CT_TASK, "--name", "fx", "--model", "cnn"]),
        ("predict", ["table", "totalseg_*", "--within"]),
        ("predict", ["covariate-baseline", CT_TASK, "--column", "age", "--column", "group"]),
    ],
)
def test_more_guards_refuse_organ_tasks(cli_env, cli, args):
    from typer.testing import CliRunner

    from nnfoundation_bench.modeling import cox_probe, predict

    app = {"cox": cox_probe.app, "features": features.app, "predict": predict.app}[cli]
    _refused(CliRunner().invoke(app, args))
    assert not (cli_env / "results" / "fx").exists()


def test_table_within_with_a_glob_over_binary_tasks_is_not_refused(cli_env):
    """``table '*_official' --within`` asks for binary official tasks; design-within skips
    organ tasks by itself, so the glob also matching registered organ tasks must not
    refuse it (no organ results exist)."""
    from typer.testing import CliRunner

    from nnfoundation_bench.modeling import predict

    result = CliRunner().invoke(predict.app, ["table", "*_official", "--within"])
    assert result.exit_code == 0, f"{result.output} {result.exception}"

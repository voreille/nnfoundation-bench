import json

import numpy as np
import pandas as pd
import pytest

from nnfoundation_bench.modeling import features
from nnfoundation_bench.modeling.features import cell_weights, fit_probe, pool_features


def test_cell_weights_are_roi_fractions():
    mask = np.zeros((8, 8, 8), np.uint8)
    mask[:4, :4, :2] = 1  # half of cell (0, 0, 0)
    w = cell_weights(mask, (2, 2, 2))
    assert w[0, 0, 0] == 0.5 and w.sum() == 0.5


def test_roi_pooling_uses_only_roi_cells():
    fmap = np.zeros((2, 2, 2, 2), np.float32)
    fmap[:, 0, 0, 0] = [4.0, -1.0]
    fmap[:, 1, 1, 1] = [0.0, 8.0]
    mask = np.zeros((64, 64, 64), np.uint8)
    mask[:8, :8, :8] = 1  # inside cell (0, 0, 0) only, far from its border
    pooled = pool_features(fmap, mask)
    np.testing.assert_allclose(pooled["roi_mean"], [4.0, -1.0])
    np.testing.assert_allclose(pooled["mean"], [0.5, 7 / 8])
    assert set(pooled) == set(features.POOLINGS)
    # A 10 mm margin stays inside the first cell (32 voxels wide).
    np.testing.assert_allclose(pooled["roi_margin_mean"], [4.0, -1.0])


def test_roi_pooling_without_roi_falls_back_to_whole_crop():
    fmap = np.random.default_rng(0).normal(size=(3, 2, 2, 2)).astype(np.float32)
    pooled = pool_features(fmap, np.zeros((4, 4, 4), np.uint8))
    np.testing.assert_allclose(pooled["roi_mean"], pooled["mean"])


def test_fit_probe_separable():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 20)
    x = rng.normal(size=(40, 5)) + y[:, None] * 3
    idx = np.arange(40)
    split = {"train": idx[::2], "val": idx[1::4], "test": idx[3::4]}
    scores, params, val_auc = fit_probe(x, y, split)
    assert val_auc == 1.0 and params["C"] in features.PROBE_CS and params["l1_ratio"] == 0
    assert ((scores > 0.5) == y[split["test"]]).all()


def test_probe_writes_results_like_finetuned_runs(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    cases = [f"c{i:02d}" for i in range(40)]
    labels = {c: int(i % 2) for i, c in enumerate(cases)}
    root = tmp_path / "processed" / "t"
    (root / "features" / "cnn").mkdir(parents=True)
    (root / "labels.json").write_text(json.dumps(labels))
    split = {"train": cases[:16], "val": cases[16:24], "test": cases[24:]}
    (root / "splits.json").write_text(json.dumps([split]))
    for c, lab in labels.items():
        vec = rng.normal(size=4) + 3 * lab
        np.savez(root / "features" / "cnn" / f"{c}.npz", mean=vec)
    monkeypatch.setattr(features, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(
        features, "results_dir", lambda task, run, fold: tmp_path / "results" / run / str(fold)
    )
    features.probe("t", model="cnn", pooling=["mean"])
    metrics = json.loads(
        (tmp_path / "results" / "cnn_frozen_mean_logreg/0/metrics.json").read_text()
    )
    assert metrics["auroc"] == 1.0 and metrics["n"] == 16


def test_stack_scores_learns_which_modality_to_trust():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 30)
    good = np.where(y == 1, 0.8, 0.2) + rng.normal(0, 0.05, 60)
    noise = rng.uniform(0.1, 0.9, 60)
    scores = np.column_stack([noise, good]).clip(0.01, 0.99)
    test = features.stack_scores(scores[::2], y[::2], scores[1::2])
    assert ((test > 0.5) == y[1::2]).mean() > 0.9


def _fusion_fixture(tmp_path, monkeypatch, tasks=("a", "b")):
    """Two modality tasks of the same 40 cases; only modality "b" carries the label."""
    from nnfoundation_bench.modeling import runs

    rng = np.random.default_rng(0)
    cases = [f"c{i:02d}" for i in range(40)]
    labels = {c: int(i % 2) for i, c in enumerate(cases)}
    split = {"train": cases[:16], "val": cases[16:24], "test": cases[24:]}
    for task in tasks:
        root = tmp_path / "processed" / task
        (root / "features" / "vit").mkdir(parents=True)
        (root / "labels.json").write_text(json.dumps(labels))
        (root / "splits.json").write_text(json.dumps([split]))
        for c, lab in labels.items():
            vec = rng.normal(size=4) + (3 * lab if task == "b" else 0)
            np.savez(root / "features" / "vit" / f"{c}.npz", roi_mean=vec)
    monkeypatch.setattr(features, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(features, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(runs, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(
        features, "results_dir", lambda t, run, fold: tmp_path / "results" / t / run / str(fold)
    )


def test_fuse_writes_every_method_under_the_fused_name(tmp_path, monkeypatch):
    _fusion_fixture(tmp_path, monkeypatch)
    features.fuse(["a", "b"], name="ab", model="vit", pooling=["roi_mean"], method=None)
    out = tmp_path / "results" / "ab"
    assert json.loads((out / "fusion.json").read_text()) == {"tasks": ["a", "b"]}
    for m in features.FUSIONS:
        metrics = json.loads((out / f"vit_fused_{m}_roi_mean_logreg/0/metrics.json").read_text())
        assert metrics["task"] == "ab" and metrics["n"] == 16
        if m != "mean":  # the mean blurs the informative modality with noise
            assert metrics["auroc"] > 0.9


def test_fuse_refuses_tasks_with_different_splits(tmp_path, monkeypatch):
    import pytest
    import typer

    _fusion_fixture(tmp_path, monkeypatch)
    (tmp_path / "processed" / "b" / "splits.json").write_text("[]")
    with pytest.raises(typer.Exit):
        features.fuse(["a", "b"], name="ab", model="vit", pooling=["roi_mean"], method=None)


def test_pool_features_per_named_roi():
    import numpy as np

    from nnfoundation_bench.modeling.features import pool_features

    fmap = np.zeros((1, 2, 1, 1), np.float32)
    fmap[0, 0], fmap[0, 1] = 1.0, 3.0  # two cells along z
    mask = np.zeros((4, 2, 2), np.uint8)
    mask[:2] = 1  # GTVp fills the first cell
    pooled = pool_features(fmap, mask, ((1, "gtvp"), (2, "gtvn")))
    assert pooled["gtvp_mean"].tolist() == [1.0]
    assert np.isnan(pooled["gtvn_mean"]).all()  # absent ROI -> NaN, imputed by the probe
    mask[2:] = 2
    pooled = pool_features(fmap, mask, ((1, "gtvp"), (2, "gtvn")))
    assert pooled["gtvn_mean"].tolist() == [3.0]
    assert pooled["roi_mean"].tolist() == [2.0]  # union


def test_design_table_survival_rows_use_cox_runs():
    import pandas as pd

    from nnfoundation_bench.modeling.predict import design_table

    summary = pd.Series(
        {"vit_frozen_gtvp_mean_cox": "0.65 ± 0.02", "radiomics_original_cox": "0.66 ± 0.01"}
    )
    lines = design_table(
        {"h22": summary}, kinds={"h22": "cox"}, headers={"h22": "h22 (C-index)"}
    ).splitlines()
    assert lines[0] == "| ID | h22 (C-index) |"
    assert "| F5 | - / 0.65 ± 0.02 |" in lines
    assert "| R1 | 0.66 ± 0.01 |" in lines
    assert not any(line.startswith("| F7 ") for line in lines)  # no F7 run anywhere


# ------------------------------------------------------------ organ tasks (issue #4)

ORGAN_TASK = "totalseg_ct_organs"


def _check_crop_mapping(image, crop, organ_mask_crop, mask, organ, pad):
    """The crop is a window of ``image`` (pad outside), located via the organ's bbox."""
    lo_orig = np.array([np.nonzero(mask == organ)[a].min() for a in range(3)])
    lo_crop = np.array([np.nonzero(organ_mask_crop)[a].min() for a in range(3)])
    offset = lo_orig - lo_crop  # crop index + offset = image index
    for idx in np.ndindex(crop.shape):
        src = np.array(idx) + offset
        inside = np.all(src >= 0) and np.all(src < np.array(image.shape))
        expected = image[tuple(src)] if inside else pad
        assert crop[idx] == expected, (idx, crop[idx], expected)
        expected_mask = inside and mask[tuple(src)] == organ
        assert bool(organ_mask_crop[idx]) == expected_mask
    return offset


def _volume(shape=(20, 20, 20)):
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 100.0
    return image, np.zeros(shape, np.uint8)


def test_organ_crop_is_exact_size_centred_and_binary():
    from nnfoundation_bench.modeling.features import organ_crop

    image, mask = _volume()
    mask[5:8, 9:12, 10:13] = 3  # 3^3 organ, centre (6, 10, 11)
    mask[5:8, 9:12, 13:15] = 2  # a neighbour inside the crop
    crop, organ_mask, inside = organ_crop(image, mask, 3, size=8, pad=-5.0)
    assert crop.shape == organ_mask.shape == (8, 8, 8)
    assert set(np.unique(organ_mask).tolist()) <= {0, 1}
    assert int(organ_mask.sum()) == 27  # the neighbour (label 2) is not in the mask
    assert inside == 1.0
    _check_crop_mapping(image, crop, organ_mask, mask, 3, -5.0)
    nz = np.nonzero(organ_mask)
    for axis in range(3):  # equal margins (+-1) on both sides of the bbox
        before = nz[axis].min()
        after = 8 - 1 - nz[axis].max()
        assert abs(before - after) <= 1


def test_organ_crop_pads_at_the_scan_border():
    from nnfoundation_bench.modeling.features import organ_crop

    image, mask = _volume()
    mask[0:3, 0:3, 17:20] = 7  # organ in a corner of the scan
    crop, organ_mask, inside = organ_crop(image, mask, 7, size=8, pad=-1024.0)
    assert crop.shape == (8, 8, 8) and inside == 1.0
    assert crop[0, 0, -1] == -1024.0  # outside the scan
    assert (crop == -1024.0).sum() > 0 and int(organ_mask.sum()) == 27
    _check_crop_mapping(image, crop, organ_mask, mask, 7, -1024.0)


def test_organ_crop_mri_pad_is_zero():
    from nnfoundation_bench.datasets.totalsegmentator import MR_PAD_VALUE
    from nnfoundation_bench.modeling.features import organ_crop

    image, mask = _volume()
    mask[18:20, 0:2, 0:2] = 1
    crop, organ_mask, _ = organ_crop(image, mask, 1, size=8, pad=MR_PAD_VALUE)
    assert crop[-1, 0, 0] == 0.0 and crop.min() == 0.0  # image values are all >= 100
    _check_crop_mapping(image, crop, organ_mask, mask, 1, 0.0)


def test_organ_larger_than_the_crop_reports_the_fraction_inside():
    from nnfoundation_bench.modeling.features import organ_crop

    image, mask = _volume()
    mask[2:18, 9:12, 10:13] = 4  # 16 voxels long along z, crop is 8
    crop, organ_mask, inside = organ_crop(image, mask, 4, size=8, pad=0.0)
    assert inside == pytest.approx(0.5)
    assert int(organ_mask.sum()) == 8 * 9
    assert crop.shape == (8, 8, 8)


def test_organ_crop_default_size_is_the_protocol_crop():
    from nnfoundation_bench.modeling.features import organ_crop

    image, mask = _volume((10, 12, 14))
    mask[4:6, 4:6, 4:6] = 1
    crop, organ_mask, inside = organ_crop(image, mask, 1, pad=-1024.0)
    assert crop.shape == organ_mask.shape == (192, 192, 192)
    assert inside == 1.0 and int(organ_mask.sum()) == 8


def _write_scan_layout(root):
    """Hand-written organ-task layout (issue #4 contract) with one 40^3 scan."""
    from nnfoundation_bench.preprocessing import write_b2nd

    for name in ("images", "masks"):
        (root / name).mkdir(parents=True)
    mean, std = 24.0, 500.0
    pad_z = (-1024 - mean) / std
    image = np.full((40, 40, 40), -1.0, np.float32)
    mask = np.zeros((40, 40, 40), np.uint8)
    image[10:22, 10:22, 10:22], mask[10:22, 10:22, 10:22] = 2.0, 3  # liver, centred
    image[0:12, 28:40, 28:40], mask[0:12, 28:40, 28:40] = 3.0, 5  # spleen, at the border
    write_b2nd(image[None], root / "images" / "s0001.b2nd")
    write_b2nd(mask[None], root / "masks" / "s0001.b2nd")
    cases = ["s0001__liver", "s0001__spleen"]
    pd.DataFrame(
        {"scan_id": ["s0001"], "shape": ["(40, 40, 40)"], "mean": [mean], "std": [std],
         "pad_z": [pad_z], "split": ["train"]}
    ).to_csv(root / "scans.csv", index=False)  # fmt: skip
    pd.DataFrame(
        {"scan_id": ["s0001"] * 2, "organ": ["liver", "spleen"], "label_id": [3, 5],
         "voxels": [1728, 1728], "overlap_voxels": [0, 0]}
    ).to_csv(root / "organs.csv", index=False)  # fmt: skip
    pd.DataFrame(
        {"case_id": cases, "scan_id": ["s0001"] * 2, "organ": ["liver", "spleen"],
         "label": [0, 1], "official_split": ["train"] * 2}
    ).to_csv(root / "labels.csv", index=False)  # fmt: skip
    (root / "labels.json").write_text(json.dumps({"s0001__liver": 0, "s0001__spleen": 1}))
    (root / "splits.json").write_text(json.dumps([{"train": cases, "val": [], "test": []}]))
    (root / "dataset.json").write_text(
        json.dumps({"layout": "scan", "num_classes": 2, "classes": {"0": "liver", "1": "spleen"}})
    )
    return cases, pad_z


def test_extract_organ_task_pools_each_organ_centred_crop(tmp_path, monkeypatch):
    import blosc2
    import torch

    from nnfoundation_bench.modeling import runs

    root = tmp_path / "processed" / ORGAN_TASK
    cases, pad_z = _write_scan_layout(root)
    monkeypatch.setattr(features, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(runs, "RESULTS_DIR", tmp_path / "results")
    seen = []

    def fake_feature_map(encoder, x):
        # Fake encoder: channel 0 = mean of each 32^3 cell, channel 1 = 1.
        seen.extend(x.detach().float().cpu().numpy()[:, 0])
        cells = torch.nn.functional.avg_pool3d(x.float(), 32)
        return torch.cat([cells, torch.ones_like(cells)], dim=1)

    monkeypatch.setattr(features, "build_encoder", lambda model, device: object())
    monkeypatch.setattr(features, "feature_map", fake_feature_map)
    features.extract(ORGAN_TASK, model="cnn", device="cpu", save_maps=False, overwrite=False)

    # Reference: the organ's bbox centre sits at crop index 95.5 on every axis; outside
    # the scan the crop holds the z-scored pad. Pooled with the fake encoder above.
    image = blosc2.open(str(root / "images" / "s0001.b2nd"), mode="r")[...][0]
    label_map = blosc2.open(str(root / "masks" / "s0001.b2nd"), mode="r")[...][0]
    ideal = {}
    for organ, label_id in (("liver", 3), ("spleen", 5)):
        nz = np.nonzero(label_map == label_id)
        start = [int(np.floor((nz[a].min() + nz[a].max()) / 2 - 95.5 + 0.5)) for a in range(3)]
        crop = np.full((192, 192, 192), pad_z, np.float32)
        organ_crop = np.zeros((192, 192, 192), np.uint8)
        src = tuple(
            slice(max(0, s), min(n, s + 192)) for s, n in zip(start, image.shape, strict=True)
        )
        dst = tuple(slice(sl.start - s, sl.stop - s) for sl, s in zip(src, start, strict=True))
        crop[dst] = image[src]
        organ_crop[dst] = label_map[src] == label_id
        fmap = fake_feature_map(None, torch.from_numpy(crop)[None, None])[0].numpy()
        ideal[organ] = pool_features(fmap, organ_crop)
    del seen[-2:]  # drop the reference crops; keep the ones extract passed in

    out = root / "features" / "cnn"
    assert sorted(p.stem for p in out.glob("*.npz")) == cases
    assert len(seen) == 2
    for crop in seen:
        assert crop.shape == (192, 192, 192)
        assert crop[0, 0, 0] == pytest.approx(pad_z, abs=0.02)  # z-scored pad outside scan
    centre_values = sorted(round(float(c[96, 96, 96]), 1) for c in seen)
    assert centre_values == [2.0, 3.0]  # each crop is centred on its organ
    for case in cases:
        pooled = np.load(out / f"{case}.npz")
        assert {"mean", "roi_mean", "roi_margin_mean", "roi_inside"} <= set(pooled.files)
        assert pooled["mean"].shape == (2,)
        assert float(pooled["roi_inside"]) == 1.0
        assert pooled["roi_mean"][1] == pytest.approx(1.0)
        # Exact values from an ideal crop built independently with numpy slicing.
        expected = ideal[case.split("__")[1]]
        for key in ("mean", "roi_mean", "roi_margin_mean"):
            assert pooled[key][0] == pytest.approx(expected[key][0], abs=0.01), key
        assert pooled["roi_mean"][0] > pooled["mean"][0]  # organ cells hold scan voxels


def _organ_probe_fixture(tmp_path, monkeypatch, n_scans=30, drop_test_class=None):
    """Organ task with 3 classes, separable features, splits on scans."""
    from nnfoundation_bench.modeling import predict, runs, survival

    rng = np.random.default_rng(0)
    organs = ("aorta", "heart", "liver")
    root = tmp_path / "processed" / ORGAN_TASK
    (root / "features" / "cnn").mkdir(parents=True)
    rows = []
    for i in range(n_scans):
        for label, organ in enumerate(organs):
            case = f"s{i:04d}__{organ}"
            rows.append({"case_id": case, "scan_id": f"s{i:04d}", "organ": organ,
                         "label": label})  # fmt: skip
            vec = rng.normal(size=6) * 0.3
            vec[label] += 4.0
            np.savez(root / "features" / "cnn" / f"{case}.npz", mean=vec)
    table = pd.DataFrame(rows).sort_values("case_id")
    table.to_csv(root / "labels.csv", index=False)
    (root / "labels.json").write_text(
        json.dumps(dict(zip(table["case_id"], table["label"].astype(int), strict=True)))
    )
    scans = [f"s{i:04d}" for i in range(n_scans)]

    def cases_of(scan_ids):
        return sorted(table.loc[table["scan_id"].isin(scan_ids), "case_id"])

    fold0 = {"train": cases_of(scans[:12]), "val": cases_of(scans[12:15]),
             "test": cases_of(scans[15:])}  # fmt: skip
    fold1 = dict(fold0)
    fold1["test"] = [c for c in fold0["test"] if not c.endswith("__liver")]
    (root / "splits.json").write_text(json.dumps([fold0, fold1]))
    (root / "dataset.json").write_text(
        json.dumps({"layout": "scan", "num_classes": 3,
                    "classes": {str(i): o for i, o in enumerate(organs)}})
    )  # fmt: skip
    for module in (features, predict, survival):
        monkeypatch.setattr(module, "PROCESSED_DATA_DIR", tmp_path / "processed")
    for module in (features, predict, runs):
        monkeypatch.setattr(module, "RESULTS_DIR", tmp_path / "results")
    return table, (fold0, fold1), organs


def test_fit_probe_multiclass_separable():
    from sklearn.metrics import balanced_accuracy_score

    rng = np.random.default_rng(1)
    y = np.repeat([0, 1, 2], 30)
    x = rng.normal(size=(90, 5)) * 0.3
    x[np.arange(90), y] += 4.0
    order = rng.permutation(90)
    split = {"train": order[:45], "val": order[45:60], "test": order[60:]}
    proba, c, val_bacc = fit_probe(x, y, split, multiclass=True)
    assert proba.shape == (30, 3)
    np.testing.assert_allclose(proba.sum(1), 1.0, rtol=1e-6)
    assert (proba.argmax(1) == y[split["test"]]).all()  # column k = class k
    assert c in features.PROBE_CS and val_bacc == 1.0
    assert balanced_accuracy_score(y[split["test"]], proba.argmax(1)) == 1.0


def test_fit_probe_binary_default_is_unchanged():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 20)
    x = rng.normal(size=(40, 5)) + y[:, None] * 3
    idx = np.arange(40)
    split = {"train": idx[::2], "val": idx[1::4], "test": idx[3::4]}
    default = fit_probe(x, y, split)
    explicit = fit_probe(x, y, split, multiclass=False)
    assert default[0].shape == (10,) and isinstance(default[1], dict)
    np.testing.assert_allclose(default[0], explicit[0])
    assert default[1] == explicit[1] and default[2] == explicit[2] == 1.0


def _classes_of(probs_npz, organs):
    classes = probs_npz["classes"]
    if np.issubdtype(classes.dtype, np.number):
        return [int(c) for c in classes]
    return [organs.index(str(c)) for c in classes]


def test_organ_probe_writes_multiclass_metrics_predictions_and_probs(tmp_path, monkeypatch):
    table, (fold0, _), organs = _organ_probe_fixture(tmp_path, monkeypatch)
    features.probe(ORGAN_TASK, model="cnn", pooling=["mean"])
    out = tmp_path / "results" / ORGAN_TASK / "cnn_frozen_mean_logreg" / "fold0"
    metrics = json.loads((out / "metrics.json").read_text())
    required = {"balanced_accuracy", "accuracy", "auroc_macro", "n", "n_classes",
                "val_balanced_accuracy", "probe_C"}  # fmt: skip
    assert required <= set(metrics)
    assert not {"auroc", "ap", "n_positive", "val_auroc"} & set(metrics)
    assert metrics["balanced_accuracy"] == 1.0 and metrics["accuracy"] == 1.0
    assert metrics["auroc_macro"] == 1.0 and metrics["val_balanced_accuracy"] == 1.0
    assert metrics["n"] == len(fold0["test"]) == 45 and metrics["n_classes"] == 3
    assert metrics["probe_C"] in features.PROBE_CS

    pred = pd.read_csv(out / "predictions.csv", dtype={"case_id": str, "scan_id": str})
    assert list(pred.columns) == ["case_id", "scan_id", "organ", "label", "pred", "p_pred"]
    assert sorted(pred["case_id"]) == fold0["test"]
    truth = table.set_index("case_id").loc[pred["case_id"]]
    assert pred["scan_id"].tolist() == truth["scan_id"].tolist()
    assert pred["organ"].tolist() == truth["organ"].tolist()
    assert pred["label"].tolist() == truth["label"].tolist() == pred["pred"].tolist()

    probs = np.load(out / "probs.npz", allow_pickle=False)
    assert set(probs.files) == {"case_id", "classes", "probs"}
    assert [str(c) for c in probs["case_id"]] == pred["case_id"].tolist()
    assert probs["probs"].shape == (45, 3)
    np.testing.assert_allclose(probs["probs"].sum(1), 1.0, rtol=1e-5)
    classes = _classes_of(probs, organs)
    assert classes == [0, 1, 2]
    best = probs["probs"].argmax(1)
    assert [classes[b] for b in best] == pred["pred"].tolist()
    np.testing.assert_allclose(probs["probs"].max(1), pred["p_pred"], rtol=1e-5)


def test_organ_probe_test_set_missing_a_class_still_scores(tmp_path, monkeypatch):
    from sklearn.metrics import roc_auc_score

    _, (_, fold1), organs = _organ_probe_fixture(tmp_path, monkeypatch)
    features.probe(ORGAN_TASK, model="cnn", pooling=["mean"])
    out = tmp_path / "results" / ORGAN_TASK / "cnn_frozen_mean_logreg" / "fold1"
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["n"] == len(fold1["test"]) == 30
    assert np.isfinite(metrics["auroc_macro"])
    pred = pd.read_csv(out / "predictions.csv")
    probs = np.load(out / "probs.npz")
    assert probs["probs"].shape == (30, 3)  # all classes kept as columns
    classes = _classes_of(probs, organs)
    present = sorted(set(pred["label"]))
    assert present == [0, 1]
    expected = np.mean(
        [roc_auc_score(pred["label"] == k, probs["probs"][:, classes.index(k)]) for k in present]
    )
    assert metrics["auroc_macro"] == pytest.approx(expected)
    assert metrics["balanced_accuracy"] == 1.0

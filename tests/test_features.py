import json

import numpy as np

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

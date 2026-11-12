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
    scores, c, val_auc = fit_probe(x, y, split)
    assert val_auc == 1.0 and c in features.PROBE_CS
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

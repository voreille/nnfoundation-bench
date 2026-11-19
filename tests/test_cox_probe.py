import json

import numpy as np
import pandas as pd

from nnfoundation_bench.modeling import cox_probe, features, predict, runs


def test_cindex_counts_correctly_ordered_pairs():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.ones(4, int)
    assert (
        cox_probe.cindex(time, event, np.array([4.0, 3.0, 2.0, 1.0])) == 1.0
    )  # high risk dies first
    assert cox_probe.cindex(time, event, np.array([1.0, 2.0, 3.0, 4.0])) == 0.0
    assert cox_probe.cindex(time, event, np.zeros(4)) == 0.5


def test_fit_cox_recovers_a_prognostic_feature():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 6))
    time = rng.exponential(np.exp(-3.0 * x[:, 0])) * 100  # feature 0 shortens survival
    event = np.ones(120, int)
    idx = np.arange(120)
    split = {"train": idx[:60], "val": idx[60:80], "test": idx[80:]}
    risk, alpha, val_c = cox_probe.fit_cox(x, time, event, split)
    assert alpha in cox_probe.COX_ALPHAS and val_c > 0.7
    assert cox_probe.cindex(time[80:], event[80:], risk) > 0.7


def _survival_task(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    cases = [f"c{i:02d}" for i in range(60)]
    root = tmp_path / "processed" / "t"
    (root / "features" / "vit").mkdir(parents=True)
    x = rng.normal(size=(60, 3))
    time = rng.exponential(np.exp(-3.0 * x[:, 0])) * 100
    (root / "labels.json").write_text(
        json.dumps({c: int(t >= 50) for c, t in zip(cases, time, strict=True)})
    )
    split = {"train": cases[:30], "val": cases[30:40], "test": cases[40:]}
    (root / "splits.json").write_text(json.dumps([split]))
    pd.DataFrame(
        {"case_id": cases, "label": 0, "time": time, "event": 1, "age": x[:, 0],
         "group": ["A", "B"] * 30}
    ).to_csv(root / "labels.csv", index=False)  # fmt: skip
    for c, row in zip(cases, x, strict=True):
        np.savez(root / "features" / "vit" / f"{c}.npz", roi_mean=row)
    for module in (features, cox_probe, predict):
        monkeypatch.setattr(module, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(predict, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(runs, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(
        cox_probe,
        "results_dir",
        lambda t, run, fold: tmp_path / "results" / t / run / f"fold{fold}",
    )


def test_cox_probe_writes_cindex_results_and_the_cindex_grid(tmp_path, monkeypatch):
    _survival_task(tmp_path, monkeypatch)
    cox_probe.probe("t", model="vit", pooling=["roi_mean"], column=None)
    cox_probe.probe("t", model="covariates", pooling=None, column=["age"])
    out = tmp_path / "results" / "t"
    metrics = json.loads((out / "vit_frozen_roi_mean_cox/fold0/metrics.json").read_text())
    assert metrics["cindex"] > 0.6 and metrics["n"] == 20 and "auroc" not in metrics
    preds = pd.read_csv(out / "vit_frozen_roi_mean_cox/fold0/predictions.csv")
    assert list(preds.columns) == ["case_id", "time", "event", "risk"]
    # The C-index grid shows the Cox runs; the AUROC tables ignore them.
    grid = predict.render_block("cindex t").splitlines()
    assert grid[0] == "| ID | t |"
    assert any(line.startswith("| F2 | - / 0.") for line in grid)
    assert any(line.startswith("| ref-age | 0.") for line in grid)
    assert len(predict.auroc_stats("t")) == 0


def test_cli_exposes_probe_as_a_subcommand():
    from typer.testing import CliRunner

    result = CliRunner().invoke(cox_probe.app, ["probe", "--help"])
    assert result.exit_code == 0 and "--model" in result.output


def test_survival_only_needs_label_equal_to_event(tmp_path, monkeypatch):
    """D33: time/event plus a binary label of its own stays a classification task
    (nnfb-cox gives its C-index); label = event makes it survival-only (D27)."""
    import pytest
    import typer

    from nnfoundation_bench.modeling import survival

    _survival_task(tmp_path, monkeypatch)
    monkeypatch.setattr(survival, "PROCESSED_DATA_DIR", tmp_path / "processed")
    assert not survival.is_survival("t")  # label 0, event 1: 1-year-style label
    path = tmp_path / "processed" / "t" / "labels.csv"
    labels = pd.read_csv(path)
    labels.assign(label=labels["event"]).to_csv(path, index=False)
    assert survival.is_survival("t") and survival.probe_kind("t") == "cox"
    with pytest.raises(typer.Exit):
        cox_probe.probe("t", model="vit", pooling=["roi_mean"], column=None)

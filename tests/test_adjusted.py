import numpy as np
import pytest

from nnfoundation_bench.modeling import cox_probe, features, predict
from tests.test_cox_probe import _survival_task


def test_within_group_cindex_ignores_cross_group_pairs():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.ones(4, int)
    groups = np.array(["a", "a", "b", "b"])
    # A risk set by the group alone ties every within-group pair: exactly 0.5.
    assert cox_probe.within_group_cindex(time, event, np.array([1, 1, 0, 0.0]), groups) == 0.5
    # Right order inside each group, wrong order across groups: 1.0 within, < 1 overall.
    risk = np.array([2.0, 1.0, 4.0, 3.0])
    assert cox_probe.within_group_cindex(time, event, risk, groups) == 1.0
    assert cox_probe.cindex(time, event, risk) < 1.0
    # One group: the plain C-index.
    one = np.zeros(4)
    assert cox_probe.within_group_cindex(time, event, risk, one) == cox_probe.cindex(
        time, event, risk
    )


def test_blend_picks_the_informative_score():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    good, noise = y + rng.normal(0, 0.5, 200), rng.normal(size=200)
    from sklearn.metrics import roc_auc_score

    def metric(s):
        return roc_auc_score(y[:100], s)

    _, w = features.blend(np.c_[good, noise][:100], np.c_[good, noise][100:], metric)
    assert w >= 0.8
    _, w = features.blend(np.c_[noise, good][:100], np.c_[noise, good][100:], metric)
    assert w <= 0.2
    assert features.plus_run_name("vit_frozen_roi_mean_logreg", ["age", "group"]) == (
        "vit_frozen_roi_mean_plus_age_group_logreg"
    )


@pytest.fixture
def task(tmp_path, monkeypatch):
    _survival_task(tmp_path, monkeypatch)
    monkeypatch.setattr(
        features,
        "results_dir",
        lambda t, run, fold: tmp_path / "results" / t / run / f"fold{fold}",
    )
    return "t"


def test_combine_writes_blends_and_the_adjusted_table(task):
    features.combine(task, model="vit", pooling=["roi_mean"], column=["age"])
    cox_probe.combine(task, model="vit", pooling=["roi_mean"], column=["age"])
    cox_probe.probe(task, model="vit", pooling=["roi_mean"], column=None)
    cox_probe.probe(task, model="covariates", pooling=None, column=["age"])
    table = predict.render_block(f"adjusted {task}").splitlines()
    assert table[0] == "| Run | AUROC | Within-centre AUROC | C-index | Within-centre C-index |"
    rows = [line.split(" | ")[0] for line in table[2:]]
    assert rows == [
        "| `covariates_age`",
        "| `vit_frozen_roi_mean`",
        "| `vit_frozen_roi_mean_plus_age`",
    ]
    blend_row = table[4].split(" | ")
    assert all(cell.startswith("0.") for cell in blend_row[1:])  # all four metrics present
    within = predict.render_block(f"cindex-within {task}").splitlines()
    assert within[0] == f"| ID | {task} |"
    assert any(line.startswith("| F2 | - / 0.") for line in within)

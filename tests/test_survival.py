import numpy as np
import pandas as pd

from nnfoundation_bench.modeling.survival import (
    cindex,
    fit_cox_probe,
    group_hazard_scores,
    within_group_cindex,
)


def test_cindex_orders_risk_against_time():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 1, 1, 0])
    assert cindex(event, time, np.array([4.0, 3.0, 2.0, 1.0])) == 1.0
    assert cindex(event, time, np.array([1.0, 2.0, 3.0, 4.0])) == 0.0
    assert cindex(event, time, np.ones(4)) == 0.5
    assert np.isnan(cindex(np.zeros(4), time, np.ones(4)))  # no event, no comparable pair


def test_within_group_cindex_ignores_cross_group_pairs():
    groups = np.array(["a", "a", "b", "b"])
    time = np.array([1.0, 2.0, 1.0, 2.0])
    event = np.array([1, 0, 1, 0])
    # Group a has all the high risks: only the group, no within-group ranking.
    assert within_group_cindex(event, time, np.array([9.0, 9.0, 1.0, 1.0]), groups) == 0.5
    assert within_group_cindex(event, time, np.array([9.0, 8.0, 2.0, 1.0]), groups) == 1.0


def test_cox_probe_finds_the_risk_feature():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=(n, 20))
    t_event = rng.exponential(np.exp(-1.5 * x[:, 0]))
    t_cens = rng.exponential(2.0, size=n)
    time, event = np.minimum(t_event, t_cens), (t_event < t_cens).astype(int)
    split = {"train": np.arange(150), "val": np.arange(150, 200), "test": np.arange(200, n)}
    risk, params, val = fit_cox_probe(x, time, event, split)
    assert cindex(event[200:], time[200:], risk) > 0.7
    assert "alpha" in params and val > 0.6


def test_group_hazard_scores_use_events_per_follow_up():
    labels = pd.DataFrame(
        {
            "case_id": ["a1", "a2", "b1", "b2", "a3", "b3"],
            "label": [1, 0, 0, 0, 1, 0],
            "event": [1, 0, 0, 0, 1, 0],
            "time": [100.0, 100.0, 100.0, 300.0, 50.0, 50.0],
            "group": ["A", "A", "B", "B", "A", "B"],
        }
    )
    split = {"train": ["a1", "a2", "b1"], "val": ["b2"], "test": ["a3", "b3"]}
    scores = group_hazard_scores(labels, split).set_index("case_id")["score"]
    assert scores["a3"] == 1 / 200 and scores["b3"] == 0.0

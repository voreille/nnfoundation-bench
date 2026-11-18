"""Survival tasks: Cox probe and C-index (scikit-survival), D27.

A task is a survival task when its ``labels.csv`` has ``time`` and ``event`` columns
(e.g. HECKTOR 2022 RFS). Its probes are ridge Cox models instead of logistic regressions,
its runs end in ``_cox`` instead of ``_logreg``, and its metric is Harrell's C-index
(``cindex`` in metrics.json) instead of AUROC. ``label`` (= event) is still used to
stratify the splits.
"""

import numpy as np
import pandas as pd

from nnfoundation_bench.config import PROCESSED_DATA_DIR

# Ridge penalties tried on the validation split (features are standardised).
COX_ALPHAS = tuple(10.0**k for k in range(-1, 6))


def survival_table(task: str) -> pd.DataFrame | None:
    """``case_id``-indexed ``time``/``event`` of a survival task, None for other tasks."""
    path = PROCESSED_DATA_DIR / task / "labels.csv"
    if not path.exists():
        return None
    labels = pd.read_csv(path, dtype={"case_id": str})
    if not {"time", "event"} <= set(labels.columns):
        return None
    return labels.set_index("case_id")[["time", "event"]]


def is_survival(task: str) -> bool:
    return survival_table(task) is not None


def probe_kind(task: str) -> str:
    """Suffix of probe run names: ``cox`` for survival tasks, ``logreg`` otherwise."""
    return "cox" if is_survival(task) else "logreg"


def cindex_counts(event: np.ndarray, time: np.ndarray, risk: np.ndarray) -> tuple[float, int]:
    """(concordant + tied_risk / 2, comparable pairs) of Harrell's C-index.

    A pair is comparable when the shorter time is an event. Higher risk should go with
    the shorter time.
    """
    from sksurv.metrics import concordance_index_censored

    event = np.asarray(event, bool)
    if not event.any():
        return 0.0, 0
    try:
        _, concordant, discordant, tied_risk, _ = concordance_index_censored(
            event, np.asarray(time, float), np.asarray(risk, float)
        )
    except Exception:  # noqa: BLE001 - sksurv raises when no pair is comparable
        return 0.0, 0
    return concordant + 0.5 * tied_risk, int(concordant + discordant + tied_risk)


def cindex(event: np.ndarray, time: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C-index (challenge metric); NaN without comparable pairs."""
    score, pairs = cindex_counts(event, time, risk)
    return score / pairs if pairs else float("nan")


def within_group_cindex(
    event: np.ndarray, time: np.ndarray, risk: np.ndarray, groups: np.ndarray
) -> float:
    """C-index over comparable pairs from the same group only, pooled over groups.

    The survival version of ``within_group_auroc`` (D24): a risk that depends on the group
    alone gets exactly 0.5.
    """
    score, pairs = 0.0, 0
    for group in np.unique(groups):
        inside = groups == group
        s, p = cindex_counts(event[inside], time[inside], risk[inside])
        score, pairs = score + s, pairs + p
    return score / pairs if pairs else float("nan")


def survival_metrics(event: np.ndarray, time: np.ndarray, risk: np.ndarray) -> dict:
    return {
        "cindex": cindex(event, time, risk),
        "n": len(event),
        "n_events": int(np.sum(event)),
    }


def fit_cox_probe(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    split: dict[str, np.ndarray],
    alphas=COX_ALPHAS,
) -> tuple[np.ndarray, dict, float]:
    """Ridge Cox model on train, penalty picked by val C-index.

    Same pipeline as the logistic probe: median imputation and standardisation with train
    statistics, fit on train only. Returns test risk scores, ``{"alpha"}`` and the val
    C-index.
    """
    import warnings

    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.util import Surv

    tr, va, te = split["train"], split["val"], split["test"]
    y_train = Surv.from_arrays(event[tr].astype(bool), time[tr])
    best = (-np.inf, None, None)
    for alpha in alphas:
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            CoxPHSurvivalAnalysis(alpha=alpha, n_iter=200),
        )
        with warnings.catch_warnings():
            # Weak penalties on many features may not converge; val decides.
            warnings.simplefilter("ignore")
            try:
                model.fit(x[tr], y_train)
            except (ValueError, np.linalg.LinAlgError):
                continue
        val = cindex(event[va], time[va], model.predict(x[va]))
        if best[1] is None or val > best[0]:
            best = (val, {"alpha": alpha}, model)
    val_cindex, params, model = best
    if model is None:
        raise RuntimeError("Cox probe: no penalty could be fitted")
    return model.predict(x[te]), params, float(val_cindex)


def group_hazard_scores(labels: pd.DataFrame, split: dict[str, list[str]]) -> pd.DataFrame:
    """Survival ``group_prior``: risk = events per day of follow-up of the case's group in
    train+val (exponential model per centre)."""
    fit = labels[labels["case_id"].isin(split["train"] + split["val"])]
    per_group = fit.groupby("group").agg(events=("event", "sum"), exposure=("time", "sum"))
    rate = per_group["events"] / per_group["exposure"].clip(lower=1.0)
    overall = fit["event"].sum() / max(fit["time"].sum(), 1.0)
    test = labels[labels["case_id"].isin(split["test"])]
    score = test["group"].map(rate).fillna(overall)
    return test[["case_id", "label", "time", "event"]].assign(score=score.to_numpy())

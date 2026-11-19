"""Cox survival probes on frozen features, radiomics or covariates; C-index per fold
(``uv run nnfb-cox --help``, D30).

The survival counterpart of the logistic-regression probes in ``features.py``: same cached
features, same splits (``splits.json``), same results layout
(``outputs/results/<task>/<run>/fold<k>/``). The task's ``labels.csv`` must have ``time``
and ``event`` columns. metrics.json holds ``cindex`` (Harrell) instead of ``auroc``.
"""

import json
import warnings
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from loguru import logger

from nnfoundation_bench.config import PROCESSED_DATA_DIR
from nnfoundation_bench.modeling.features import POOLINGS, load_pooled, task_labels
from nnfoundation_bench.modeling.runs import processed_task, results_dir
from nnfoundation_bench.weights import MODELS

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Cox survival probes (C-index). A callback keeps ``probe`` a subcommand."""


# Ridge strengths tried on val (standardised features; sksurv penalty 0.5 * alpha * |b|^2).
COX_ALPHAS = tuple(10.0**k for k in range(-1, 6))


def cox_run_name(source: str, detail: str | None = None) -> str:
    """``vit_frozen_roi_mean_cox``, ``radiomics_original_cox``, ``covariates_age_cox``."""
    if source in MODELS:
        return f"{source}_frozen_{detail}_cox"
    return f"{source}_{detail}_cox"


def cindex(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C: fraction of comparable pairs ranked correctly by risk (ties 1/2)."""
    from sksurv.metrics import concordance_index_censored

    return float(concordance_index_censored(event.astype(bool), time, risk)[0])


def fit_cox(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    split: dict[str, np.ndarray],
    alphas=COX_ALPHAS,
) -> tuple[np.ndarray, float, float]:
    """Ridge Cox model on train, alpha picked by val C-index (as ``fit_probe`` picks C).

    Missing values get the train median, features are standardised on train. Returns test
    risk scores (higher = shorter survival), the chosen alpha and the val C-index.
    """
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
            # Weak penalties on many features may not converge; val C-index decides.
            warnings.simplefilter("ignore")
            try:
                model.fit(x[tr], y_train)
            except (ValueError, np.linalg.LinAlgError):
                continue
        c = cindex(time[va], event[va], model.predict(x[va]))
        if best[1] is None or c > best[0]:
            best = (c, alpha, model)
    val_c, alpha, model = best
    if model is None:
        raise RuntimeError("No Cox fit converged for any alpha")
    return model.predict(x[te]), float(alpha), float(val_c)


def survival_columns(task: str, cases: list[str]) -> tuple[np.ndarray, np.ndarray]:
    labels = pd.read_csv(
        PROCESSED_DATA_DIR / processed_task(task) / "labels.csv", dtype={"case_id": str}
    ).set_index("case_id")
    if not {"time", "event"} <= set(labels):
        raise typer.Exit(f"{task}: labels.csv has no time/event columns")
    return labels.loc[cases, "time"].to_numpy(float), labels.loc[cases, "event"].to_numpy(int)


def cox_folds(task: str, run: str, x: np.ndarray) -> list[float]:
    """``fit_cox`` on every fold; ``x`` rows follow ``task_labels`` order. Writes
    predictions.csv (case_id, time, event, risk) and metrics.json (cindex) per fold."""
    cases, _, splits = task_labels(task)
    time, event = survival_columns(task, cases)
    index = {c: i for i, c in enumerate(cases)}
    scores = []
    for fold, split in enumerate(splits):
        idx = {k: np.array([index[c] for c in v]) for k, v in split.items()}
        risk, alpha, val_c = fit_cox(x, time, event, idx)
        te = idx["test"]
        out_dir = results_dir(task, run, fold)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"case_id": split["test"], "time": time[te], "event": event[te], "risk": risk}
        ).to_csv(out_dir / "predictions.csv", index=False)
        metrics = {
            "task": task,
            "run": run,
            "fold": fold,
            "cindex": cindex(time[te], event[te], risk),
            "cox_alpha": alpha,
            "val_cindex": val_c,
            "num_features": int(x.shape[1]),
            "n": len(te),
            "n_events": int(event[te].sum()),
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        scores.append(metrics["cindex"])
    logger.info(f"{run}: test C-index {np.mean(scores):.4f} ± {np.std(scores, ddof=1):.4f}")
    return scores


@app.command()
def probe(
    task: str,
    model: str = typer.Option(
        "vit", help=f"{', '.join(MODELS)} (frozen features), radiomics or covariates"
    ),
    pooling: Annotated[
        list[str] | None, typer.Option(help="Poolings of the frozen features (default: all)")
    ] = None,
    column: Annotated[
        list[str] | None, typer.Option(help="labels.csv columns for covariates (default: age)")
    ] = None,
) -> None:
    """Cox probe per fold, C-index on test; results like the AUROC probes, run ``*_cox``."""
    cases, _, _ = task_labels(task)
    if model in MODELS:
        for pool in pooling or POOLINGS:
            x = load_pooled(task, model, pool, cases)
            cox_folds(task, cox_run_name(model, pool), x)
    elif model == "radiomics":
        from nnfoundation_bench.modeling.radiomics_baseline import load_features

        cox_folds(
            task, cox_run_name("radiomics", "original"), load_features(task, "original", cases)[0]
        )
    elif model == "covariates":
        from nnfoundation_bench.modeling.predict import covariate_matrix

        columns = column or ["age"]
        labels = pd.read_csv(
            PROCESSED_DATA_DIR / processed_task(task) / "labels.csv", dtype={"case_id": str}
        ).set_index("case_id")
        x = covariate_matrix(labels.loc[cases], columns)
        cox_folds(task, cox_run_name("covariates", "_".join(columns)), x)
    else:
        raise typer.Exit(f"Unknown model {model!r}")


if __name__ == "__main__":
    app()

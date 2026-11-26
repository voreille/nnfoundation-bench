"""Cox survival probes on frozen features, radiomics or covariates; C-index per fold
(``uv run nnfb-cox --help``, D32).

The survival counterpart of the logistic-regression probes in ``features.py``: same cached
features, same splits (``splits.json``), same results layout
(``outputs/results/<task>/<run>/fold<k>/``). The task's ``labels.csv`` must have ``time``
and ``event`` columns. metrics.json holds ``cindex`` (Harrell) instead of ``auroc``.
``combine`` blends an image Cox model with a covariate Cox model (D36);
``within_group_cindex`` is the within-centre C-index (D35).
"""

import json
import warnings
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from loguru import logger

from nnfoundation_bench.config import PROCESSED_DATA_DIR
from nnfoundation_bench.modeling.features import (
    FEATURE_MODELS,
    POOLINGS,
    load_pooled,
    task_labels,
)
from nnfoundation_bench.modeling.runs import processed_task, results_dir

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Cox survival probes (C-index). A callback keeps ``probe`` a subcommand."""


# Ridge strengths tried on val (standardised features; sksurv penalty 0.5 * alpha * |b|^2).
COX_ALPHAS = tuple(10.0**k for k in range(-1, 6))


def cox_run_name(source: str, detail: str | None = None) -> str:
    """``vit_frozen_roi_mean_cox``, ``radiomics_original_cox``, ``covariates_age_cox``."""
    if source in FEATURE_MODELS:
        return f"{source}_frozen_{detail}_cox"
    return f"{source}_{detail}_cox"


def cindex(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C: fraction of comparable pairs ranked correctly by risk (ties 1/2)."""
    from sksurv.metrics import concordance_index_censored

    return float(concordance_index_censored(event.astype(bool), time, risk)[0])


def within_group_cindex(
    time: np.ndarray, event: np.ndarray, risk: np.ndarray, groups: np.ndarray
) -> float:
    """Harrell's C over comparable pairs from the same group only, pooled over groups (D35):
    ``survival.within_group_cindex`` (D27) with this module's (time, event, risk) order. A
    risk that depends on the group alone gets exactly 0.5."""
    from nnfoundation_bench.modeling.survival import within_group_cindex as pooled

    return pooled(np.asarray(event), np.asarray(time), np.asarray(risk), np.asarray(groups))


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


def save_cox_fold(task, run, fold, cases, time, event, risk, info: dict) -> float:
    """Write one fold's predictions.csv (case_id, time, event, risk) and metrics.json."""
    out_dir = results_dir(task, run, fold)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"case_id": cases, "time": time, "event": event, "risk": risk}).to_csv(
        out_dir / "predictions.csv", index=False
    )
    metrics = {"task": task, "run": run, "fold": fold, "cindex": cindex(time, event, risk)}
    metrics |= info | {"n": len(cases), "n_events": int(event.sum())}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics["cindex"]


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
        info = {"cox_alpha": alpha, "val_cindex": val_c, "num_features": int(x.shape[1])}
        scores.append(
            save_cox_fold(task, run, fold, split["test"], time[te], event[te], risk, info)
        )
    logger.info(f"{run}: test C-index {np.mean(scores):.4f} ± {np.std(scores, ddof=1):.4f}")
    return scores


@app.command()
def probe(
    task: str,
    model: str = typer.Option(
        "vit", help=f"{', '.join(FEATURE_MODELS)} (frozen features), radiomics or covariates"
    ),
    pooling: Annotated[
        list[str] | None, typer.Option(help="Poolings of the frozen features (default: all)")
    ] = None,
    column: Annotated[
        list[str] | None, typer.Option(help="labels.csv columns for covariates (default: age)")
    ] = None,
) -> None:
    """Cox probe per fold, C-index on test; results like the AUROC probes, run ``*_cox``."""
    from nnfoundation_bench.modeling.survival import is_survival

    if is_survival(task):  # its own probes are Cox already, under the same run names
        raise typer.Exit(f"{task} is a survival-only task: use nnfb-features probe (D27, D33)")
    cases, _, _ = task_labels(task)
    if model in FEATURE_MODELS:
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


def combine_cox_folds(task: str, run: str, x_image: np.ndarray, x_cov: np.ndarray) -> list[float]:
    """Per fold: an image Cox model and a covariate Cox model (``fit_cox``: train, alpha by
    val), then ``blend`` of their risks with the image weight chosen by val C-index (D36)."""
    from nnfoundation_bench.modeling.features import blend

    cases, _, splits = task_labels(task)
    time, event = survival_columns(task, cases)
    index = {c: i for i, c in enumerate(cases)}
    scores = []
    for fold, split in enumerate(splits):
        idx = {k: np.array([index[c] for c in v]) for k, v in split.items()}
        va, te, n_val = idx["val"], idx["test"], len(idx["val"])
        both = {**idx, "test": np.concatenate([va, te])}
        risks = np.column_stack([fit_cox(x, time, event, both)[0] for x in (x_image, x_cov)])
        risk, w = blend(
            risks[:n_val], risks[n_val:], lambda r, va=va: cindex(time[va], event[va], r)
        )
        info = {"blend_weight_image": w}
        scores.append(
            save_cox_fold(task, run, fold, split["test"], time[te], event[te], risk, info)
        )
    logger.info(f"{run}: test C-index {np.mean(scores):.4f} ± {np.std(scores, ddof=1):.4f}")
    return scores


@app.command()
def combine(
    task: str,
    model: str = typer.Option("vit", help=f"{', '.join(FEATURE_MODELS)} or radiomics"),
    pooling: Annotated[
        list[str] | None, typer.Option(help="Poolings (default: all; not radiomics)")
    ] = None,
    column: Annotated[
        list[str] | None, typer.Option(help="labels.csv covariates (default: age)")
    ] = None,
) -> None:
    """Image Cox + covariate Cox blended on val (D36). Run ``<image run>_plus_<columns>_cox``."""
    from nnfoundation_bench.modeling.features import (
        image_features,
        plus_run_name,
        task_covariates,
    )

    cases, _, _ = task_labels(task)
    columns = column or ["age"]
    x_cov = task_covariates(task, cases, columns)
    for pool in [None] if model == "radiomics" else pooling or POOLINGS:
        base, x_image = image_features(task, model, pool, cases)
        run = plus_run_name(base.removesuffix("_logreg") + "_cox", columns)
        combine_cox_folds(task, run, x_image, x_cov)


if __name__ == "__main__":
    app()

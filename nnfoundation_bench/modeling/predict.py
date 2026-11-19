"""Test-set inference and AUROC evaluation (``uv run nnfb-predict --help``)."""

import json
import re
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

from nnfoundation_bench.config import PROCESSED_DATA_DIR, RESULTS_DIR
from nnfoundation_bench.modeling.runs import (
    eval_run_name,
    processed_task,
    results_dir,
    run_dir,
    run_name,
)
from nnfoundation_bench.weights import MODELS

app = typer.Typer(no_args_is_help=True, add_completion=False)


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """AUROC (paper metric) and AP; NaN when the split holds a single class."""
    single_class = len(np.unique(labels)) < 2
    return {
        "auroc": float("nan") if single_class else float(roc_auc_score(labels, scores)),
        "ap": float("nan") if single_class else float(average_precision_score(labels, scores)),
        "n": len(labels),
        "n_positive": int(labels.sum()),
    }


def within_group_auroc(labels: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> float:
    """AUROC over (positive, negative) pairs from the same group only, pooled over groups.

    A stratified Mann-Whitney statistic: each pair counts once, ties count 1/2. A score
    that depends on the group alone (``group_prior``) gets exactly 0.5, so the group
    shortcut is removed from the metric. NaN if no group holds both classes.
    """
    concordant, pairs = 0.0, 0
    for group in np.unique(groups):
        in_group = groups == group
        pos = scores[in_group & (labels == 1)]
        neg = scores[in_group & (labels == 0)]
        if len(pos) and len(neg):
            diff = pos[:, None] - neg[None, :]
            concordant += (diff > 0).sum() + 0.5 * (diff == 0).sum()
            pairs += diff.size
    return concordant / pairs if pairs else float("nan")


def evaluate_predictions(pred_path: Path) -> pd.DataFrame:
    """Turn glovita's ``predictions.pt`` into a per-case table (score = P(class 1))."""
    import torch

    out = torch.load(pred_path, map_location="cpu")
    return pd.DataFrame(
        {
            "case_id": out["case_ids"],
            "label": out["labels"].numpy(),
            "score": out["probs"][:, 1].float().numpy(),
        }
    )


@app.command()
def predict(
    task: str,
    model: str = typer.Option("cnn", help=f"Pretrained model: {', '.join(MODELS)}"),
    fold: int = typer.Option(0),
    lr: float | None = typer.Option(None, help="LR of the run (default: paper value)"),
    scratch: bool = typer.Option(False),
    tag: str = typer.Option(""),
    checkpoint: str = typer.Option(
        "best",
        help="best (by val AUROC) or last (end of the schedule); last goes to <run>_last",
    ),
    num_workers: int = typer.Option(8),
) -> None:
    """Run a checkpoint of a run on the fold's test split and compute AUROC.

    Results go to ``outputs/results/<task>/<run>/`` for ``best`` and
    ``<run>_last/`` for ``last``, so evaluating one never overwrites the other.
    """
    run = run_name(model, lr, scratch, tag)
    result_run = eval_run_name(run, checkpoint)
    out_dir = results_dir(task, result_run, fold)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.pt"
    cmd = [
        sys.executable, "-m", "glovita.cli.infer",
        "--exp_dir", str(run_dir(task, run)),
        "--fold", str(fold),
        "--checkpoint", checkpoint,
        "--dataloading.batch_size", "1",
        "--dataloading.num_workers", str(num_workers),
        "--pred_output", str(pred_path),
        "data:generic3d_dataset_config",
        "--data.data_root_dir", str(PROCESSED_DATA_DIR / task),
        "--data.fold", str(fold),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)

    cases = evaluate_predictions(pred_path)
    cases.to_csv(out_dir / "predictions.csv", index=False)
    metrics = {"task": task, "run": result_run, "fold": fold, "checkpoint": checkpoint}
    metrics |= binary_metrics(cases["label"].to_numpy(), cases["score"].to_numpy())
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.success(f"{task} {result_run} fold {fold}: AUROC {metrics['auroc']:.4f}")


def group_prior_scores(labels: pd.DataFrame, split: dict[str, list[str]]) -> pd.DataFrame:
    """Score test cases by the positive rate of their group in train+val."""
    fit = labels[labels["case_id"].isin(split["train"] + split["val"])]
    prior = fit.groupby("group")["label"].mean()
    test = labels[labels["case_id"].isin(split["test"])]
    score = test["group"].map(prior).fillna(fit["label"].mean())
    return test[["case_id", "label"]].assign(score=score.to_numpy())


@app.command("group-baseline")
def group_baseline(task: str) -> None:
    """Reference run 'group_prior': predict from the case's group (e.g. centre) alone.

    Needs a ``group`` column in labels.csv. Shows how much of a task is solvable from
    confounders; model AUROCs should be read against it.
    """
    root = PROCESSED_DATA_DIR / processed_task(task)
    labels = pd.read_csv(root / "labels.csv")
    if "group" not in labels:
        raise typer.Exit(f"{task} has no group column in labels.csv")
    labels = labels[labels["case_id"].isin(json.loads((root / "labels.json").read_text()))]
    for fold, split in enumerate(json.loads((root / "splits.json").read_text())):
        cases = group_prior_scores(labels, split)
        out_dir = results_dir(task, "group_prior", fold)
        out_dir.mkdir(parents=True, exist_ok=True)
        cases.to_csv(out_dir / "predictions.csv", index=False)
        metrics = {"task": task, "run": "group_prior", "fold": fold}
        metrics |= binary_metrics(cases["label"].to_numpy(), cases["score"].to_numpy())
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        logger.info(f"group_prior fold {fold}: AUROC {metrics['auroc']:.4f}")


def covariate_matrix(labels: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Numeric columns as they are (NaN kept: the probe imputes the train median),
    other columns one-hot encoded (e.g. ``group``)."""
    parts = []
    for column in columns:
        values = labels[column]
        if pd.api.types.is_numeric_dtype(values):
            parts.append(values.to_numpy(dtype=float)[:, None])
        else:
            parts.append(pd.get_dummies(values.astype(str)).to_numpy(dtype=float))
    return np.hstack(parts)


def covariate_run_name(columns: list[str]) -> str:
    return "covariates_" + "_".join(columns) + "_logreg"


@app.command("covariate-baseline")
def covariate_baseline(
    task: str,
    column: Annotated[
        list[str] | None, typer.Option(help="labels.csv columns to use (default: age)")
    ] = None,
) -> None:
    """Reference run from case covariates alone (e.g. age, or age and group).

    Same probe and splits as the frozen features (L2 logistic regression, C chosen on
    val). Run ``covariates_<columns>_logreg``; design rows ``ref-age`` and
    ``ref-age+site`` (D27).
    """
    from nnfoundation_bench.modeling.features import probe_folds, task_labels

    columns = column or ["age"]
    labels = pd.read_csv(
        PROCESSED_DATA_DIR / processed_task(task) / "labels.csv", dtype={"case_id": str}
    ).set_index("case_id")
    if missing := [c for c in columns if c not in labels]:
        raise typer.Exit(f"{task}: no column {missing} in labels.csv")
    cases, _, _ = task_labels(task)
    probe_folds(task, covariate_run_name(columns), covariate_matrix(labels.loc[cases], columns))


@app.command()
def summarize(task: str) -> None:
    """Mean/std AUROC over folds for every evaluated run of TASK."""
    rows = [
        json.loads(p.read_text())
        for p in sorted((RESULTS_DIR / task).glob("*/fold*/metrics.json"))
    ]
    if not rows:
        raise typer.Exit(f"No results under {RESULTS_DIR / task}")
    df = pd.DataFrame(rows)
    summary = df.groupby("run")["auroc"].agg(["mean", "std", "count"]).sort_values("mean")
    summary.to_csv(RESULTS_DIR / task / "summary.csv")
    typer.echo(summary.to_string(float_format="%.4f"))


# Design-benchmark rows (docs/progress.md) -> run name per pretrained model.
DESIGN_RUNS = {
    "F1": lambda m: f"{m}_frozen_mean_logreg",
    "F2": lambda m: f"{m}_frozen_roi_mean_logreg",
    "F3": lambda m: f"{m}_frozen_roi_margin_mean_logreg",
    "T1": lambda m: run_name(m),
    "T1-last": lambda m: eval_run_name(run_name(m), "last"),
    # Model-independent rows (one value per cell): handcrafted radiomics baselines.
    # R2-R5 (filtered set, correlation filter, elastic net) were dropped (D25); their past
    # runs still show in the runs tables.
    "R1": "radiomics_original_logreg",
    # Late fusion of per-modality ROI-mean embeddings (D14, D28), on a fused task
    # (``nnfb-features fuse``): concatenation, mean, score stacking on val.
    "M2": lambda m: f"{m}_fused_concat_roi_mean_logreg",
    "M3": lambda m: f"{m}_fused_mean_roi_mean_logreg",
    "M4": lambda m: f"{m}_fused_stack_roi_mean_logreg",
    # Confounder references from labels.csv covariates (D27); ``ref`` (group alone) is
    # the last row of the grid.
    "ref-age": "covariates_age_logreg",
    "ref-age+site": "covariates_age_group_logreg",
    "ref-age+hpv": "covariates_age_hpv_logreg",
}


def task_groups(task: str) -> pd.Series | None:
    """case_id -> group (e.g. centre) from labels.csv, or None if the task has no groups."""
    path = PROCESSED_DATA_DIR / processed_task(task) / "labels.csv"
    if not path.exists():
        return None
    labels = pd.read_csv(path, dtype={"case_id": str})
    return labels.set_index("case_id")["group"] if "group" in labels else None


def auroc_stats(task: str, within_group: bool = False) -> pd.DataFrame:
    """Test AUROC ``mean``, ``std`` and fold ``count`` per run of TASK (empty if none).

    ``within_group`` recomputes each fold's AUROC from its ``predictions.csv`` with
    ``within_group_auroc`` (tasks with a ``group`` column; empty otherwise).
    """
    empty = pd.DataFrame(columns=["mean", "std", "count"])
    if not within_group:
        paths = (RESULTS_DIR / task).glob("*/fold*/metrics.json")
        # Cox runs (nnfb-cox) report a C-index instead and are left out here.
        rows = [r for p in paths if "auroc" in (r := json.loads(p.read_text()))]
    elif (groups := task_groups(task)) is None:
        return empty
    else:
        rows = []
        for path in (RESULTS_DIR / task).glob("*/fold*/predictions.csv"):
            cases = pd.read_csv(path, dtype={"case_id": str})
            if "score" not in cases:  # Cox runs: risk, not a class score
                continue
            auroc = within_group_auroc(
                cases["label"].to_numpy(),
                cases["score"].to_numpy(),
                cases["case_id"].map(groups).to_numpy(),
            )
            rows.append({"run": path.parent.parent.name, "auroc": auroc})
    if not rows:
        return empty
    return pd.DataFrame(rows).groupby("run")["auroc"].agg(["mean", "std", "count"])


def format_auroc(row: pd.Series, n_folds: int = 5) -> str:
    """``"mean ± std"``, with the fold count if it is not the full set."""
    folds = "" if row["count"] == n_folds else f" (n={int(row['count'])})"
    return f"{row['mean']:.3f} ± {row['std']:.3f}{folds}"


def auroc_summary(task: str, within_group: bool = False) -> pd.Series:
    """``"mean ± std"`` test AUROC per run of TASK (empty if nothing is evaluated)."""
    stats = auroc_stats(task, within_group)
    return stats.apply(format_auroc, axis=1) if len(stats) else pd.Series(dtype=str)


def design_table(
    summaries: dict[str, pd.Series],
    models: tuple[str, ...] = ("cnn", "vit"),
    designs: dict | None = None,
    ref: str | None = "group_prior",
) -> str:
    """Markdown grid: one row per design, one column per task, cells ``CNN / ViT``
    (one value for model-independent rows such as radiomics). ``designs`` defaults to
    ``DESIGN_RUNS``; ``ref`` (last row) is left out when None."""
    lines = ["| ID | " + " | ".join(summaries) + " |", "|----|" + "---|" * len(summaries)]
    for design, run in (designs or DESIGN_RUNS).items():
        if isinstance(run, str):
            cells = [s.get(run, "-") for s in summaries.values()]
        else:
            cells = [" / ".join(s.get(run(m), "-") for m in models) for s in summaries.values()]
        lines.append(f"| {design} | " + " | ".join(cells) + " |")
    if ref is not None:
        cells = [s.get(ref, "n/a") for s in summaries.values()]
        lines.append("| ref | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# C-index grid (``<!-- generated: cindex [patterns] -->``): Cox probes of nnfb-cox (D30).
COX_DESIGN_RUNS = {
    "F1": lambda m: f"{m}_frozen_mean_cox",
    "F2": lambda m: f"{m}_frozen_roi_mean_cox",
    "F3": lambda m: f"{m}_frozen_roi_margin_mean_cox",
    "R1": "radiomics_original_cox",
    "ref-age": "covariates_age_cox",
    "ref-age+hpv": "covariates_age_hpv_cox",
    "ref-site": "covariates_group_cox",
}


def cindex_summary(task: str) -> pd.Series:
    """``"mean ± std"`` test C-index per Cox run of TASK (empty if none)."""
    paths = (RESULTS_DIR / task).glob("*/fold*/metrics.json")
    rows = [r for p in paths if "cindex" in (r := json.loads(p.read_text()))]
    if not rows:
        return pd.Series(dtype=str)
    stats = pd.DataFrame(rows).groupby("run")["cindex"].agg(["mean", "std", "count"])
    return stats.apply(format_auroc, axis=1)


def design_ids(models: tuple[str, ...] = ("cnn", "vit")) -> dict[str, str]:
    """Run name -> design ID (``F1``...), plus ``ref`` for ``group_prior``."""
    ids = {"group_prior": "ref"}
    for design, run in DESIGN_RUNS.items():
        for name in [run] if isinstance(run, str) else [run(m) for m in models]:
            ids[name] = design
    return ids


def runs_table(stats: pd.DataFrame, within: pd.DataFrame | None = None) -> str:
    """Markdown table of every run of one task, best mean AUROC first; with ``within``
    (within-group AUROC per run), a second AUROC column."""
    ids = design_ids()
    header = "| Run | Design | AUROC |" + (" Within-centre AUROC |" if within is not None else "")
    lines = [header, "|" + "---|" * header.count(" |")]
    for run, row in stats.sort_values("mean", ascending=False).iterrows():
        line = f"| `{run}` | {ids.get(run, '')} | {format_auroc(row)} |"
        if within is not None:
            line += f" {format_auroc(within.loc[run]) if run in within.index else '-'} |"
        lines.append(line)
    return "\n".join(lines)


# Generated blocks in a markdown file (task patterns are shell globs, default: all):
#   <!-- generated: design [patterns...] -->         ...  <!-- end generated -->
#   <!-- generated: design-within [patterns...] -->  ...  <!-- end generated -->
#   <!-- generated: runs <task> -->                  ...  <!-- end generated -->
#   <!-- generated: cindex [patterns...] -->         ...  <!-- end generated -->
# design-within: within-centre AUROC, for the tasks that have a group column.
# cindex: test C-index of the Cox probes (nnfb-cox), for tasks with survival labels.
GENERATED_BLOCK = re.compile(
    r"(?P<open><!-- generated: (?P<spec>[^>]*?) -->\n).*?(?P<close><!-- end generated -->)",
    re.DOTALL,
)


def match_tasks(patterns: list[str]) -> list[str]:
    """Tasks with results whose name matches one of the glob ``patterns`` (all if none)."""
    tasks = sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir())
    return [t for t in tasks if not patterns or any(fnmatch(t, p) for p in patterns)]


def render_block(spec: str) -> str:
    kind, *args = spec.split()
    if kind in ("design", "design-within"):
        within = kind == "design-within"
        tasks = [t for t in match_tasks(args) if not within or task_groups(t) is not None]
        if not tasks:
            return "No results yet."
        return design_table({task: auroc_summary(task, within) for task in tasks})
    if kind == "cindex":
        summaries = {t: cindex_summary(t) for t in match_tasks(args)}
        summaries = {t: s for t, s in summaries.items() if len(s)}
        if not summaries:
            return "No results yet."
        return design_table(summaries, designs=COX_DESIGN_RUNS, ref=None)
    if kind == "runs" and len(args) == 1:
        stats = auroc_stats(args[0])
        if not len(stats):
            return "No results yet."
        has_groups = task_groups(args[0]) is not None
        return runs_table(stats, auroc_stats(args[0], within_group=True) if has_groups else None)
    raise ValueError(f"Unknown generated block: {spec!r}")


def update_generated(text: str, render=render_block) -> str:
    """Replace the body of every generated block of ``text`` with ``render(spec)``."""
    return GENERATED_BLOCK.sub(lambda m: f"{m['open']}{render(m['spec'])}\n{m['close']}", text)


@app.command()
def table(
    tasks: Annotated[
        list[str] | None, typer.Argument(help="Task globs (default: all with results)")
    ] = None,
    update: Annotated[
        Path | None,
        typer.Option(help="Rewrite the generated blocks of this markdown file instead"),
    ] = None,
    within: Annotated[
        bool, typer.Option(help="Within-centre AUROC (tasks with a group column)")
    ] = False,
) -> None:
    """Design-benchmark grid (markdown) from every task's results; ref = group_prior.

    With ``--update docs/progress.md``, every ``<!-- generated: ... -->`` block of the
    file is rebuilt from outputs/results, so result tables never come from memory.
    """
    if update is not None:
        update.write_text(update_generated(update.read_text()))
        logger.success(f"Updated the generated tables of {update}")
        return
    kind = "design-within" if within else "design"
    typer.echo(render_block(" ".join([kind, *(tasks or [])])))


if __name__ == "__main__":
    app()

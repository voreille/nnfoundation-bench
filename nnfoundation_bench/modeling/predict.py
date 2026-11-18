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
from nnfoundation_bench.modeling.runs import eval_run_name, results_dir, run_dir, run_name
from nnfoundation_bench.modeling.survival import (
    group_hazard_scores,
    probe_kind,
    survival_metrics,
    survival_table,
    within_group_cindex,
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
    confounders; model AUROCs should be read against it. Survival tasks: risk = the
    group's events per day of follow-up, scored by C-index.
    """
    root = PROCESSED_DATA_DIR / task
    labels = pd.read_csv(root / "labels.csv", dtype={"case_id": str})
    if "group" not in labels:
        raise typer.Exit(f"{task} has no group column in labels.csv")
    labels = labels[labels["case_id"].isin(json.loads((root / "labels.json").read_text()))]
    survival = survival_table(task) is not None
    key = metric_key(task)
    for fold, split in enumerate(json.loads((root / "splits.json").read_text())):
        if survival:
            cases = group_hazard_scores(labels, split)
            scores = survival_metrics(cases["event"], cases["time"], cases["score"])
        else:
            cases = group_prior_scores(labels, split)
            scores = binary_metrics(cases["label"].to_numpy(), cases["score"].to_numpy())
        out_dir = results_dir(task, "group_prior", fold)
        out_dir.mkdir(parents=True, exist_ok=True)
        cases.to_csv(out_dir / "predictions.csv", index=False)
        metrics = {"task": task, "run": "group_prior", "fold": fold} | scores
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        logger.info(f"group_prior fold {fold}: {key} {metrics[key]:.4f}")


@app.command()
def summarize(task: str) -> None:
    """Mean/std AUROC (C-index for survival tasks) over folds for every run of TASK."""
    rows = [
        json.loads(p.read_text())
        for p in sorted((RESULTS_DIR / task).glob("*/fold*/metrics.json"))
    ]
    if not rows:
        raise typer.Exit(f"No results under {RESULTS_DIR / task}")
    df = pd.DataFrame(rows)
    summary = df.groupby("run")[metric_key(task)].agg(["mean", "std", "count"])
    summary = summary.sort_values("mean")
    summary.to_csv(RESULTS_DIR / task / "summary.csv")
    typer.echo(summary.to_string(float_format="%.4f"))


# Design-benchmark rows (docs/progress.md) -> run name, given the pretrained model ``m``
# and the probe kind ``p`` (``logreg``, or ``cox`` on survival tasks).
DESIGN_RUNS = {
    "F1": lambda m, p: f"{m}_frozen_mean_{p}",
    "F2": lambda m, p: f"{m}_frozen_roi_mean_{p}",
    "F3": lambda m, p: f"{m}_frozen_roi_margin_mean_{p}",
    # Tasks with several named ROIs (HECKTOR 2022: GTVp, GTVn; D27).
    "F5": lambda m, p: f"{m}_frozen_gtvp_mean_{p}",
    "F6": lambda m, p: f"{m}_frozen_gtvn_mean_{p}",
    "F7": lambda m, p: f"{m}_frozen_roi_concat_mean_{p}",
    "T1": lambda m, p: run_name(m),
    "T1-last": lambda m, p: eval_run_name(run_name(m), "last"),
    # Model-independent rows (one value per cell): handcrafted radiomics baselines.
    # R2-R5 (filtered set, correlation filter, elastic net) were dropped (D25); their past
    # runs still show in the runs tables.
    "R1": lambda p: f"radiomics_original_{p}",
}


def metric_key(task: str) -> str:
    """metrics.json key of the task's main metric: ``cindex`` or ``auroc``."""
    return "cindex" if survival_table(task) is not None else "auroc"


def metric_label(task: str) -> str:
    return "C-index" if survival_table(task) is not None else "AUROC"


def task_groups(task: str) -> pd.Series | None:
    """case_id -> group (e.g. centre) from labels.csv, or None if the task has no groups."""
    path = PROCESSED_DATA_DIR / task / "labels.csv"
    if not path.exists():
        return None
    labels = pd.read_csv(path, dtype={"case_id": str})
    return labels.set_index("case_id")["group"] if "group" in labels else None


def auroc_stats(task: str, within_group: bool = False) -> pd.DataFrame:
    """Test ``mean``, ``std`` and fold ``count`` of the task's metric per run (AUROC, or
    C-index on survival tasks); empty if nothing is evaluated.

    ``within_group`` recomputes each fold's metric from its ``predictions.csv`` with
    ``within_group_auroc`` / ``within_group_cindex`` (tasks with a ``group`` column;
    empty otherwise).
    """
    empty = pd.DataFrame(columns=["mean", "std", "count"])
    key = metric_key(task)
    if not within_group:
        paths = (RESULTS_DIR / task).glob("*/fold*/metrics.json")
        rows = [json.loads(p.read_text()) for p in paths]
    elif (groups := task_groups(task)) is None:
        return empty
    else:
        rows = []
        for path in (RESULTS_DIR / task).glob("*/fold*/predictions.csv"):
            cases = pd.read_csv(path, dtype={"case_id": str})
            group = cases["case_id"].map(groups).to_numpy()
            score = cases["score"].to_numpy()
            if key == "cindex":
                value = within_group_cindex(
                    cases["event"].to_numpy(), cases["time"].to_numpy(), score, group
                )
            else:
                value = within_group_auroc(cases["label"].to_numpy(), score, group)
            rows.append({"run": path.parent.parent.name, key: value})
    if not rows:
        return empty
    return pd.DataFrame(rows).groupby("run")[key].agg(["mean", "std", "count"])


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
    kinds: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> str:
    """Markdown grid: one row per design, one column per task, cells ``CNN / ViT``
    (one value for model-independent rows such as radiomics).

    ``kinds`` gives each task's probe kind (default ``logreg``), ``headers`` its column
    title (default the task name). Rows without any run in these tasks are left out.
    """
    kinds, headers = kinds or {}, headers or {}
    titles = [headers.get(t, t) for t in summaries]
    lines = ["| ID | " + " | ".join(titles) + " |", "|----|" + "---|" * len(summaries)]
    for design, run in DESIGN_RUNS.items():
        single = run.__code__.co_argcount == 1
        cells, found = [], False
        for task, s in summaries.items():
            p = kinds.get(task, "logreg")
            names = [run(p)] if single else [run(m, p) for m in models]
            found |= any(n in s for n in names)
            cells.append(" / ".join(s.get(n, "-") for n in names))
        if found or design in ("F1", "F2", "F3", "T1", "T1-last", "R1"):
            lines.append(f"| {design} | " + " | ".join(cells) + " |")
    cells = [s.get("group_prior", "n/a") for s in summaries.values()]
    lines.append("| ref | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def design_ids(models: tuple[str, ...] = ("cnn", "vit"), kind: str = "logreg") -> dict[str, str]:
    """Run name -> design ID (``F1``...), plus ``ref`` for ``group_prior``."""
    ids = {"group_prior": "ref"}
    for design, run in DESIGN_RUNS.items():
        single = run.__code__.co_argcount == 1
        for name in [run(kind)] if single else [run(m, kind) for m in models]:
            ids[name] = design
    return ids


def runs_table(
    stats: pd.DataFrame,
    within: pd.DataFrame | None = None,
    kind: str = "logreg",
    metric: str = "AUROC",
) -> str:
    """Markdown table of every run of one task, best mean first; with ``within``
    (within-group metric per run), a second metric column."""
    ids = design_ids(kind=kind)
    header = f"| Run | Design | {metric} |"
    header += f" Within-centre {metric} |" if within is not None else ""
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
# design-within: within-centre AUROC, for the tasks that have a group column.
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
        return design_table(
            {task: auroc_summary(task, within) for task in tasks},
            kinds={task: probe_kind(task) for task in tasks},
            headers={t: f"{t} ({metric_label(t)})" for t in tasks if metric_label(t) != "AUROC"},
        )
    if kind == "runs" and len(args) == 1:
        task = args[0]
        stats = auroc_stats(task)
        if not len(stats):
            return "No results yet."
        within = auroc_stats(task, within_group=True) if task_groups(task) is not None else None
        return runs_table(stats, within, kind=probe_kind(task), metric=metric_label(task))
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

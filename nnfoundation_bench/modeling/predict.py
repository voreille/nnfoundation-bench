"""Test-set inference and AUROC evaluation (``uv run nnfb-predict --help``)."""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

from nnfoundation_bench.config import PROCESSED_DATA_DIR, RESULTS_DIR
from nnfoundation_bench.modeling.runs import eval_run_name, results_dir, run_dir, run_name
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
    root = PROCESSED_DATA_DIR / task
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
    "R1": "radiomics_original_logreg",
    "R2": "radiomics_filtered_logreg",
    "R3": "radiomics_filtered_corr0.9_logreg",
    "R4": "radiomics_filtered_enet",
    "R5": "radiomics_filtered_corr0.9_enet",
}


def auroc_stats(task: str) -> pd.DataFrame:
    """Test AUROC ``mean``, ``std`` and fold ``count`` per run of TASK (empty if none)."""
    rows = [json.loads(p.read_text()) for p in (RESULTS_DIR / task).glob("*/fold*/metrics.json")]
    if not rows:
        return pd.DataFrame(columns=["mean", "std", "count"])
    return pd.DataFrame(rows).groupby("run")["auroc"].agg(["mean", "std", "count"])


def format_auroc(row: pd.Series, n_folds: int = 5) -> str:
    """``"mean ± std"``, with the fold count if it is not the full set."""
    folds = "" if row["count"] == n_folds else f" (n={int(row['count'])})"
    return f"{row['mean']:.3f} ± {row['std']:.3f}{folds}"


def auroc_summary(task: str) -> pd.Series:
    """``"mean ± std"`` test AUROC per run of TASK (empty if nothing is evaluated)."""
    stats = auroc_stats(task)
    return stats.apply(format_auroc, axis=1) if len(stats) else pd.Series(dtype=str)


def design_table(summaries: dict[str, pd.Series], models: tuple[str, ...] = ("cnn", "vit")) -> str:
    """Markdown grid: one row per design, one column per task, cells ``CNN / ViT``
    (one value for model-independent rows such as radiomics)."""
    lines = ["| ID | " + " | ".join(summaries) + " |", "|----|" + "---|" * len(summaries)]
    for design, run in DESIGN_RUNS.items():
        if isinstance(run, str):
            cells = [s.get(run, "-") for s in summaries.values()]
        else:
            cells = [" / ".join(s.get(run(m), "-") for m in models) for s in summaries.values()]
        lines.append(f"| {design} | " + " | ".join(cells) + " |")
    cells = [s.get("group_prior", "n/a") for s in summaries.values()]
    lines.append("| ref | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def design_ids(models: tuple[str, ...] = ("cnn", "vit")) -> dict[str, str]:
    """Run name -> design ID (``F1``...), plus ``ref`` for ``group_prior``."""
    ids = {"group_prior": "ref"}
    for design, run in DESIGN_RUNS.items():
        for name in [run] if isinstance(run, str) else [run(m) for m in models]:
            ids[name] = design
    return ids


def runs_table(stats: pd.DataFrame) -> str:
    """Markdown table of every run of one task, best mean AUROC first."""
    ids = design_ids()
    lines = ["| Run | Design | AUROC |", "|-----|--------|-------|"]
    for run, row in stats.sort_values("mean", ascending=False).iterrows():
        lines.append(f"| `{run}` | {ids.get(run, '')} | {format_auroc(row)} |")
    return "\n".join(lines)


# Generated blocks in a markdown file:
#   <!-- generated: design [tasks...] -->  ...  <!-- end generated -->
#   <!-- generated: runs <task> -->        ...  <!-- end generated -->
GENERATED_BLOCK = re.compile(
    r"(?P<open><!-- generated: (?P<spec>[^>]*?) -->\n).*?(?P<close><!-- end generated -->)",
    re.DOTALL,
)


def render_block(spec: str) -> str:
    kind, *tasks = spec.split()
    if kind == "design":
        tasks = tasks or sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir())
        return design_table({task: auroc_summary(task) for task in tasks})
    if kind == "runs" and len(tasks) == 1:
        stats = auroc_stats(tasks[0])
        return runs_table(stats) if len(stats) else "No results yet."
    raise ValueError(f"Unknown generated block: {spec!r}")


def update_generated(text: str, render=render_block) -> str:
    """Replace the body of every generated block of ``text`` with ``render(spec)``."""
    return GENERATED_BLOCK.sub(lambda m: f"{m['open']}{render(m['spec'])}\n{m['close']}", text)


@app.command()
def table(
    tasks: Annotated[
        list[str] | None, typer.Argument(help="Tasks (default: all with results)")
    ] = None,
    update: Annotated[
        Path | None,
        typer.Option(help="Rewrite the generated blocks of this markdown file instead"),
    ] = None,
) -> None:
    """Design-benchmark grid (markdown) from every task's results; ref = group_prior.

    With ``--update docs/progress.md``, every ``<!-- generated: ... -->`` block of the
    file is rebuilt from outputs/results, so result tables never come from memory.
    """
    if update is not None:
        update.write_text(update_generated(update.read_text()))
        logger.success(f"Updated the generated tables of {update}")
        return
    typer.echo(render_block(" ".join(["design", *(tasks or [])])))


if __name__ == "__main__":
    app()

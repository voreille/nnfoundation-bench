"""Test-set inference and AUROC evaluation (``uv run nnfb-predict --help``)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

from nnfoundation_bench.config import PROCESSED_DATA_DIR, RESULTS_DIR
from nnfoundation_bench.modeling.runs import results_dir, run_dir, run_name
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
    checkpoint: str = typer.Option("best", help="best or last"),
    num_workers: int = typer.Option(8),
) -> None:
    """Run the best checkpoint of a run on the fold's test split and compute AUROC."""
    run = run_name(model, lr, scratch, tag)
    out_dir = results_dir(task, run, fold)
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
    metrics = {"task": task, "run": run, "fold": fold, "checkpoint": checkpoint}
    metrics |= binary_metrics(cases["label"].to_numpy(), cases["score"].to_numpy())
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.success(f"{task} {run} fold {fold}: AUROC {metrics['auroc']:.4f}")


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


if __name__ == "__main__":
    app()

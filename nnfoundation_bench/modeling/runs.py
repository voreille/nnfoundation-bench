"""Naming and locations of fine-tuning runs."""

import json
from pathlib import Path

from nnfoundation_bench.config import FINETUNED_DIR, RESULTS_DIR
from nnfoundation_bench.weights import MODELS


def run_name(model: str, lr: float | None = None, scratch: bool = False, tag: str = "") -> str:
    lr = MODELS[model].lr if lr is None else lr
    name = f"{model}{'_scratch' if scratch else ''}_lr{lr:g}"
    return f"{name}_{tag}" if tag else name


CHECKPOINTS = ("best", "last")


def eval_run_name(run: str, checkpoint: str = "best") -> str:
    """Results name of a run evaluated at ``checkpoint``: ``best`` keeps ``run``,
    ``last`` gives ``<run>_last``, so both can sit side by side in outputs/results."""
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"checkpoint must be one of {CHECKPOINTS}, got {checkpoint!r}")
    return run if checkpoint == "best" else f"{run}_{checkpoint}"


def experiment_dir(task: str) -> Path:
    """glovita ``--exp_dir``; runs land in ``<exp_dir>/generic_3d_dataset/<run>/<fold>/``."""
    return FINETUNED_DIR / task


def run_dir(task: str, run: str) -> Path:
    return experiment_dir(task) / "generic_3d_dataset" / run


def results_dir(task: str, run: str, fold: int) -> Path:
    return RESULTS_DIR / task / run / f"fold{fold}"


def fusion_sources(task: str) -> list[str] | None:
    """Source tasks of a fused task (``nnfb-features fuse``), or None for a plain task.

    A fused task has no data/processed folder of its own; ``fuse`` records its sources in
    ``outputs/results/<task>/fusion.json``.
    """
    path = RESULTS_DIR / task / "fusion.json"
    return json.loads(path.read_text())["tasks"] if path.exists() else None


def processed_task(task: str) -> str:
    """Task whose data/processed folder holds TASK's labels and splits: TASK itself, or
    the first source of a fused task (all sources share labels and splits)."""
    sources = fusion_sources(task)
    return sources[0] if sources else task


def is_organ_task(task: str) -> bool:
    """Whether TASK (or the source of a fused task) is a multi-class organ task (D38)."""
    from nnfoundation_bench.datasets import TASKS

    registered = TASKS.get(processed_task(task))
    return registered is not None and registered.organ_task


def refuse_organ_tasks(*tasks: str, what: str) -> None:
    """Exit with a clear error if one of TASKS is an organ task: they are probe-only."""
    import typer

    for task in tasks:
        if is_organ_task(task):
            raise typer.Exit(
                f"{task} is a multi-class organ task (D38): {what} is not supported. "
                "Organ tasks are probe-only: nnfb-features extract / probe."
            )

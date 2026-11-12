"""Naming and locations of fine-tuning runs."""

from pathlib import Path

from nnfoundation_bench.config import FINETUNED_DIR, RESULTS_DIR
from nnfoundation_bench.weights import MODELS


def run_name(model: str, lr: float | None = None, scratch: bool = False, tag: str = "") -> str:
    lr = MODELS[model].lr if lr is None else lr
    name = f"{model}{'_scratch' if scratch else ''}_lr{lr:g}"
    return f"{name}_{tag}" if tag else name


def experiment_dir(task: str) -> Path:
    """glovita ``--exp_dir``; runs land in ``<exp_dir>/generic_3d_dataset/<run>/<fold>/``."""
    return FINETUNED_DIR / task


def run_dir(task: str, run: str) -> Path:
    return experiment_dir(task) / "generic_3d_dataset" / run


def results_dir(task: str, run: str, fold: int) -> Path:
    return RESULTS_DIR / task / run / f"fold{fold}"

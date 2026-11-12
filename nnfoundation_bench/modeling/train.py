"""Fine-tune nnFoundation on a prepared task with glovita (``uv run nnfb-train --help``)."""

import os
import subprocess
import sys

import typer
from loguru import logger

from nnfoundation_bench.config import (
    DEFAULT_SEED,
    EFFECTIVE_BATCH_SIZE,
    EPOCHS,
    PROCESSED_DATA_DIR,
    STEPS_PER_EPOCH,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
)
from nnfoundation_bench.modeling.runs import experiment_dir, run_name
from nnfoundation_bench.weights import MODELS, get_checkpoint

app = typer.Typer(add_completion=False)


def build_command(
    task: str,
    model: str,
    fold: int,
    lr: float,
    run: str,
    num_gpus: int,
    batch_size: int,
    epochs: int,
    steps_per_epoch: int,
    scratch: bool,
    seed: int,
    num_workers: int,
) -> list[str]:
    per_step = batch_size * num_gpus
    if EFFECTIVE_BATCH_SIZE % per_step:
        raise ValueError(f"batch_size x num_gpus = {per_step} must divide {EFFECTIVE_BATCH_SIZE}")
    launch = [sys.executable, "-m", "accelerate.commands.launch", "--num_machines", "1"]
    launch += ["--num_processes", str(num_gpus), "--dynamo_backend", "no"]
    launch += ["--multi_gpu"] if num_gpus > 1 else []
    experiment_dir(task).mkdir(parents=True, exist_ok=True)
    # Global options first, then each tyro subcommand followed by its own options.
    return [
        *launch,
        "-m", "nnfoundation_bench.modeling.glovita_entry",
        "--exp_dir", str(experiment_dir(task)),
        "--task.metrics", "auroc", "ap", "balanced_acc", "f1",
        "--optimizer.name", "AdamW",
        "--optimizer.lr", f"{lr:g}",
        "--optimizer.weight_decay", f"{WEIGHT_DECAY:g}",
        "--optimizer.warmup_epochs", str(max(1, round(epochs * WARMUP_FRACTION))),
        "--training.epochs", str(epochs),
        "--training.precision", "bf16",
        "--training.gradient_accumulation_steps", str(EFFECTIVE_BATCH_SIZE // per_step),
        "--training.train_steps_per_epoch", str(steps_per_epoch),
        "--training.best_checkpoint_metric", "val_AUROC",
        "--training.seed", str(seed),
        "--dataloading.batch_size", str(batch_size),
        "--dataloading.num_workers", str(num_workers),
        "data:generic3d_dataset_config",
        "--data.data_root_dir", str(PROCESSED_DATA_DIR / task),
        "--data.fold", str(fold),
        "model.encoder:nnfoundation_encoder_config",
        "--model.encoder.checkpoint_path", str(get_checkpoint(model)),
        "--model.encoder.input_channels", "1",
        *(["--model.encoder.no_pretrained"] if scratch else []),
        "model.head:classification_head_config",
        "--model.head.init_std", "0.01",
        "logger:mlflow_logger_config",
        "--logger.tracking_uri", f"sqlite:///{experiment_dir(task) / 'mlflow.db'}",
        "--logger.experiment_name", task,
        "--logger.group", run,
        "--logger.run_name", f"{run}_fold{fold}",
    ]  # fmt: skip


@app.command()
def train(
    task: str,
    model: str = typer.Option("cnn", help=f"Pretrained model: {', '.join(MODELS)}"),
    fold: int = typer.Option(0, help="Split index in splits.json"),
    lr: float | None = typer.Option(None, help="Peak LR (default: paper value per model)"),
    gpus: str = typer.Option("0", help="Comma-separated GPU ids, e.g. 0,1"),
    batch_size: int = typer.Option(1, help="Per-GPU batch size (accumulation fills up to 32)"),
    epochs: int = typer.Option(EPOCHS, help="Epochs (paper: 100, reduced compute: 20)"),
    steps_per_epoch: int = typer.Option(STEPS_PER_EPOCH),
    scratch: bool = typer.Option(False, help="Random init with the same architecture"),
    seed: int = typer.Option(DEFAULT_SEED),
    num_workers: int = typer.Option(8),
    tag: str = typer.Option("", help="Suffix for the run name"),
) -> None:
    """Fine-tune MODEL on TASK, fold FOLD (full fine-tuning, best checkpoint by val AUROC)."""
    lr = MODELS[model].lr if lr is None else lr
    run = run_name(model, lr, scratch, tag)
    cmd = build_command(
        task, model, fold, lr, run, len(gpus.split(",")), batch_size, epochs, steps_per_epoch,
        scratch, seed, num_workers,
    )  # fmt: skip
    logger.info(f"Run {run}, fold {fold}: {' '.join(cmd)}")
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpus,
        # Samples per epoch across all processes: steps x effective batch (see glovita_entry).
        "NNFB_MIN_TRAIN_SAMPLES": str(steps_per_epoch * EFFECTIVE_BATCH_SIZE),
    }
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    app()

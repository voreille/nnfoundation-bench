"""CLI for downloading and preparing benchmark datasets (``uv run nnfb-data --help``)."""

import typer
from loguru import logger

from nnfoundation_bench.config import DEFAULT_SEED, N_SPLITS
from nnfoundation_bench.datasets import TASKS, get_task

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("list")
def list_tasks() -> None:
    """List registered classification tasks."""
    for task in TASKS.values():
        typer.echo(f"{task.name:20} [{task.source}] {task.description}")


@app.command()
def download(task: str) -> None:
    """Download the raw source data of TASK into data/raw/."""
    get_task(task).download()
    logger.success(f"Downloaded {task}")


@app.command()
def prepare(
    task: str,
    device: str = typer.Option(
        "gpu", help="Device for ROI models such as TotalSegmentator: gpu, gpu:N or cpu"
    ),
    seed: int = typer.Option(DEFAULT_SEED, help="Seed of the first stratified split"),
    n_splits: int = typer.Option(N_SPLITS, help="Number of repeated 40/10/50 splits (folds)"),
    overwrite: bool = typer.Option(False, help="Recompute existing crops"),
) -> None:
    """Build data/processed/TASK for glovita (ROI crop, 1 mm, z-score, masks, splits)."""
    from nnfoundation_bench.preprocessing import prepare_task

    prepare_task(
        get_task(task),
        device=device,
        seed=seed,
        n_splits=n_splits,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    app()

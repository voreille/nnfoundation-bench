"""CLI for downloading and preparing benchmark datasets (``uv run nnfb-data --help``)."""

import typer
from loguru import logger

from nnfoundation_bench.config import DEFAULT_SEED
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
    device: str = typer.Option("gpu", help="TotalSegmentator device: gpu, gpu:N or cpu"),
    fast_roi: bool = typer.Option(True, help="Use TotalSegmentator's 3 mm fast model"),
    seed: int = typer.Option(DEFAULT_SEED, help="Seed for the stratified split"),
    overwrite: bool = typer.Option(False, help="Recompute existing crops"),
) -> None:
    """Build data/processed/TASK (ROI crop, 1 mm resampling, splits)."""
    from nnfoundation_bench.preprocessing import prepare_task

    prepare_task(get_task(task), device=device, fast_roi=fast_roi, seed=seed, overwrite=overwrite)


if __name__ == "__main__":
    app()

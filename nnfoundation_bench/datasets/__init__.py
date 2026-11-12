"""Registry of downstream classification tasks.

To add a dataset: create ``datasets/<source>.py`` exposing one or more
``ClassificationTask`` objects and register them in ``TASKS`` below.
"""

from nnfoundation_bench.datasets.base import ClassificationTask
from nnfoundation_bench.datasets.kits import KITS_LIVER

TASKS: dict[str, ClassificationTask] = {task.name: task for task in (KITS_LIVER,)}


def get_task(name: str) -> ClassificationTask:
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(f"Unknown task {name!r}. Available: {sorted(TASKS)}") from None


__all__ = ["TASKS", "ClassificationTask", "get_task"]

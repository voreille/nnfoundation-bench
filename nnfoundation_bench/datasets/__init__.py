"""Registry of downstream classification tasks.

To add a dataset: create ``datasets/<source>.py`` exposing one or more
``ClassificationTask`` objects and register them in ``TASKS`` below.
"""

from nnfoundation_bench.datasets.base import CaseInputs, ClassificationTask
from nnfoundation_bench.datasets.hecktor import HECKTOR_HPV_PET_GTVP
from nnfoundation_bench.datasets.kits import KITS_LIVER
from nnfoundation_bench.datasets.mrnet import MRNET_TASKS

TASKS: dict[str, ClassificationTask] = {
    task.name: task for task in (KITS_LIVER, HECKTOR_HPV_PET_GTVP, *MRNET_TASKS)
}


def get_task(name: str) -> ClassificationTask:
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(f"Unknown task {name!r}. Available: {sorted(TASKS)}") from None


__all__ = ["TASKS", "CaseInputs", "ClassificationTask", "get_task"]

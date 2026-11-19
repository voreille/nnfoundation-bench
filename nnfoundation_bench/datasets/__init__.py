"""Registry of downstream classification tasks.

To add a dataset: create ``datasets/<source>.py`` exposing one or more
``ClassificationTask`` objects and register them in ``TASKS`` below.
"""

from nnfoundation_bench.datasets.base import CaseInputs, ClassificationTask
from nnfoundation_bench.datasets.brats import BRATS_TASKS
from nnfoundation_bench.datasets.hecktor import HECKTOR_HPV_PET_GTVP
from nnfoundation_bench.datasets.hecktor2022_rfs2y import HECKTOR22_RFS_TASKS
from nnfoundation_bench.datasets.kits import KITS_LIVER
from nnfoundation_bench.datasets.lld_mmri import LLD_MMRI_TASKS
from nnfoundation_bench.datasets.mrnet import MRNET_TASKS
from nnfoundation_bench.datasets.rsna_spine import RSNA_SPINE_FRACTURE
from nnfoundation_bench.datasets.stoic import STOIC_TASKS
from nnfoundation_bench.datasets.upenn_gbm import UPENN_OS_TASKS

TASKS: dict[str, ClassificationTask] = {
    task.name: task
    for task in (
        KITS_LIVER,
        HECKTOR_HPV_PET_GTVP,
        *MRNET_TASKS,
        *LLD_MMRI_TASKS,
        *STOIC_TASKS,
        RSNA_SPINE_FRACTURE,
        *BRATS_TASKS,
        *UPENN_OS_TASKS,
        *HECKTOR22_RFS_TASKS,
    )
}


def get_task(name: str) -> ClassificationTask:
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(f"Unknown task {name!r}. Available: {sorted(TASKS)}") from None


__all__ = ["TASKS", "CaseInputs", "ClassificationTask", "get_task"]

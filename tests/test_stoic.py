import pandas as pd

from nnfoundation_bench.datasets import get_task
from nnfoundation_bench.datasets.stoic import target_labels


def test_target_labels():
    reference = pd.DataFrame({"PatientID": [1003, 2], "probCOVID": [1, 0], "probSevere": [1, 0]})
    assert target_labels(reference, "severe").to_dict("list") == {
        "case_id": ["1003", "2"],
        "label": [1, 0],
    }
    assert target_labels(reference, "covid")["label"].tolist() == [1, 0]


def test_tasks_registered():
    for target in ("severe", "covid"):
        task = get_task(f"stoic_{target}")
        assert task.modality == "CT" and not task.require_roi

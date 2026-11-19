import pandas as pd
import pytest

from nnfoundation_bench.datasets import get_task
from nnfoundation_bench.datasets.upenn_gbm import (
    SEQUENCES,
    WHOLE_TUMOUR,
    study_id,
    survival_labels,
)

OUTCOME = pd.DataFrame(
    {"PatientID": ["UPENN_UPENN-GBM-00002", "UPENN_UPENN-GBM-00001"],
     "Time": [364, 600], "Event": [1, 1]}
)  # fmt: skip
CLINICAL = pd.DataFrame(
    {"PatientID": ["UPENN_UPENN-GBM-00001", "UPENN_UPENN-GBM-00001", "UPENN_UPENN-GBM-00002"],
     "age": [52, 54, 61], "Time_since_baseline_preop": ["0", "521", "0"]}
)  # fmt: skip
SOURCES = {"UPENN-GBM-00001_11": "automatic", "UPENN-GBM-00002_11": "expert"}


def test_study_id_is_the_preoperative_scan():
    assert study_id("UPENN_UPENN-GBM-00002") == "UPENN-GBM-00002_11"


def test_one_year_label_time_event_age_and_mask_source():
    labels = survival_labels(OUTCOME, CLINICAL, SOURCES)
    assert labels["case_id"].tolist() == ["UPENN-GBM-00001_11", "UPENN-GBM-00002_11"]
    assert labels["label"].tolist() == [1, 0]  # 600 d >= 1 year, 364 d < 1 year
    assert labels["time"].tolist() == [600, 364] and labels["event"].tolist() == [1, 1]
    assert labels["age"].tolist() == [52, 61]  # pre-operative row, not the follow-up
    assert labels["mask_source"].tolist() == ["automatic", "expert"]
    assert "group" not in labels  # single centre


def test_missing_mask_raises():
    with pytest.raises(RuntimeError):
        survival_labels(OUTCOME, CLINICAL, {"UPENN-GBM-00001_11": "expert"})


def test_one_task_per_sequence():
    for suffix, _ in SEQUENCES.values():
        task = get_task(f"upenn_os_{suffix}")
        assert task.modality == "MR" and task.require_roi and task.pad_value == 0
    assert WHOLE_TUMOUR == (1, 2, 4)

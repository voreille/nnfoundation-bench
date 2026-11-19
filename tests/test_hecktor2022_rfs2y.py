import pandas as pd

from nnfoundation_bench.datasets import get_task
from nnfoundation_bench.datasets.hecktor2022_rfs2y import (
    GTVP,
    HPV_COLUMN,
    MODALITIES,
    rfs_labels,
)

ENDPOINT = pd.DataFrame(
    {"PatientID": ["CHUM-001", "MDA-100", "USZ-005", "CHB-002", "HGJ-010"],
     "Relapse": [1, 0, 1, 0, 1], "RFS": [300, 2000, 1500, 400, 729],
     "cohort": ["train", "train", "test", "test", "train"]}
)  # fmt: skip
CLINICAL = pd.DataFrame(
    {"PatientID": ["CHUM-001", "MDA-100", "USZ-005", "CHB-002", "HGJ-010"],
     "Age": [60, 55, 70, 48, 66], HPV_COLUMN: [None, 1.0, None, 0.0, 1.0]}
)  # fmt: skip


def test_two_year_label_drops_patients_censored_before_two_years():
    labels = rfs_labels(ENDPOINT, CLINICAL).set_index("case_id")
    assert "CHB-002" not in labels.index  # censored at 400 d: 2-year status unknown
    assert labels["label"].to_dict() == {
        "CHUM-001": 1,  # recurrence at 300 d
        "HGJ-010": 1,  # recurrence at 729 d, just inside 2 years
        "MDA-100": 0,  # recurrence-free at 2000 d
        "USZ-005": 0,  # recurrence after 2 years
    }
    assert labels.loc["USZ-005", "event"] == 1 and labels.loc["USZ-005", "time"] == 1500
    assert labels.loc["MDA-100", "hpv"] == 1 and pd.isna(labels.loc["CHUM-001", "hpv"])
    assert labels.loc["CHUM-001", "age"] == 60
    assert labels.loc["CHUM-001", "group"] == "CHUM"
    assert labels.loc["CHUM-001", "stratum"] == "CHUM_1"


def test_one_task_per_modality_on_gtvp():
    for suffix, _, pad in MODALITIES.values():
        task = get_task(f"hecktor22_rfs_{suffix}")
        assert task.roi == "GTVp" and task.require_roi and task.pad_value == pad
    assert GTVP == 1

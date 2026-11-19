import pandas as pd

from nnfoundation_bench.datasets.hecktor2022 import endpoint_labels


def test_endpoint_labels_event_time_centre():
    endpoints = pd.DataFrame(
        {
            "PatientID": ["CHUM-001", "MDA-202", "USZ-010"],
            "Relapse": [0, 1, 1],
            "RFS": [1092, 365, 0],
            "official_split": ["train", "test", "test"],
        }
    )
    labels = endpoint_labels(endpoints)
    assert labels["label"].tolist() == labels["event"].tolist() == [0, 1, 1]
    assert labels["time"].tolist() == [1092.0, 365.0, 0.0]
    assert labels["group"].tolist() == ["CHUM", "MDA", "USZ"]
    assert labels["stratum"].tolist() == ["CHUM_0", "MDA_1", "USZ_1"]
    assert labels["official_split"].tolist() == ["train", "test", "test"]


def test_endpoint_labels_rejects_duplicates():
    import pytest

    endpoints = pd.DataFrame(
        {"PatientID": ["MDA-1", "MDA-1"], "Relapse": [0, 1], "RFS": [1, 2],
         "official_split": ["train", "test"]}
    )  # fmt: skip
    with pytest.raises(RuntimeError):
        endpoint_labels(endpoints)

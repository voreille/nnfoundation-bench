import pandas as pd

from nnfoundation_bench.datasets import get_task
from nnfoundation_bench.datasets.brats import SEQUENCES, WHOLE_TUMOUR, glioma_labels

METADATA = pd.DataFrame(
    {
        "BraTS Subject ID": ["BraTS-GLI-00001-100", "BraTS-GLI-00001-101",
                             "BraTS-GLI-00002-101", "BraTS-GLI-00002-102",
                             "BraTS-GLI-00003-100", "BraTS-GLI-00004-100",
                             "BraTS-GLI-00005-100", "BraTS-GLI-00006-100"],
        "Site": ["UCSF", "UCSF", "Duke", "Duke", "UCSD", "UCSF", "UCSF", "Indiana"],
        "Glioma type": ["Astrocytoma", "Astrocytoma", "Glioblastoma", "Glioblastoma",
                        "Oligodendroglioma ", "glioma NOS", "Other", "Glioblastoma"],
        "Patient's Age": [30, 31, 60, 61, None, 50, 40, 70],
    }
)  # fmt: skip


def test_earliest_scan_per_patient_and_type_mapping():
    cases = list(METADATA["BraTS Subject ID"])
    labels = glioma_labels(METADATA, cases)
    assert labels["case_id"].tolist() == [
        "BraTS-GLI-00001-100",
        "BraTS-GLI-00002-101",
        "BraTS-GLI-00003-100",
        "BraTS-GLI-00006-100",
    ]  # one scan per patient; NOS and Other dropped
    assert labels["label"].tolist() == [0, 1, 0, 1]
    assert labels["group"].tolist() == ["UCSF", "Duke", "UCSD", "Indiana"]
    assert labels["stratum"].tolist() == ["UCSF_0", "Duke_1", "UCSD_0", "Indiana_1"]
    # age of the chosen (earliest) scan; unknown stays NaN
    assert labels["age"].tolist()[:2] == [30, 60] and labels["age"].isna().tolist()[2]


def test_earliest_among_scans_with_segmentation():
    labels = glioma_labels(METADATA, ["BraTS-GLI-00001-101", "BraTS-GLI-00002-102"])
    assert labels["case_id"].tolist() == ["BraTS-GLI-00001-101", "BraTS-GLI-00002-102"]


def test_scan_missing_from_metadata_raises():
    try:
        glioma_labels(METADATA, ["BraTS-GLI-09999-100"])
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_one_task_per_sequence():
    for sequence in SEQUENCES:
        task = get_task(f"brats_gbm_{sequence}")
        assert task.modality == "MR" and task.require_roi and task.pad_value == 0
    assert WHOLE_TUMOUR == (1, 2, 3)

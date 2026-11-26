from nnfoundation_bench.datasets import get_task
from nnfoundation_bench.datasets.kits import liver_disease_labels


def _case(case_id, mild, severe):
    return {
        "case_id": case_id,
        "comorbidities": {
            "mild_liver_disease": mild,
            "moderate_to_severe_liver_disease": severe,
        },
    }


def test_liver_label_is_union_of_liver_comorbidities():
    meta = [
        _case("case_00000", False, False),
        _case("case_00001", True, False),
        _case("case_00002", False, True),
        _case("case_00003", True, True),
    ]
    assert liver_disease_labels(meta)["label"].tolist() == [0, 1, 1, 1]


def test_task_registered():
    task = get_task("kits_liver")
    assert task.source == "kits23" and not task.require_roi


def test_malignancy_label_drops_unknown_cases():
    from nnfoundation_bench.datasets.kits import malignancy_labels

    meta = [
        {"case_id": "case_00000", "malignant": True},
        {"case_id": "case_00001", "malignant": False},
        {"case_id": "case_00002", "malignant": None},
    ]
    labels = malignancy_labels(meta)
    assert labels["case_id"].tolist() == ["case_00000", "case_00001"]
    assert labels["label"].tolist() == [1, 0]


def test_malignant_task_uses_the_tumour_mask():
    task = get_task("kits_malignant")
    inputs = task.case_inputs("case_00000", "cpu")
    assert task.require_roi and task.modality == "CT"
    assert inputs.roi_mask.parent.name == "segmentations" and inputs.roi_label == 2

from nnfoundation_bench.datasets import get_task
from nnfoundation_bench.datasets.lld_mmri import PHASES, malignancy_labels


def _series(category):
    return {"annotation": {"num_targets": 1, "lesion": {"0": {"category": category}}}}


ANNOTATION = {
    "Category_info": {"Hepatic_cyst": 4, "Hepatocellular_carcinoma": 6,
                      "Benign": [0, 2, 4, 5], "Malignant": [1, 3, 6]},
    "Annotation_info": {"MR-1": [_series(6)] * 8, "MR-2": [_series(4)] * 8},
}  # fmt: skip


def test_malignancy_from_category_grouping():
    labels = malignancy_labels(ANNOTATION, ["MR-1_1", "MR-2_3"])
    assert labels.to_dict("list") == {"case_id": ["MR-1_1", "MR-2_3"], "label": [1, 0]}


def test_inconsistent_categories_raise():
    annotation = {**ANNOTATION, "Annotation_info": {"MR-1": [_series(6), _series(4)]}}
    try:
        malignancy_labels(annotation, ["MR-1_1"])
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_one_task_per_phase():
    for suffix, _ in PHASES.values():
        task = get_task(f"lld_mmri_malignant_{suffix}")
        assert task.modality == "MR" and task.require_roi and task.pad_value == 0

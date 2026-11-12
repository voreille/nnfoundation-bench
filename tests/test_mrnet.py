import numpy as np
import pandas as pd
import SimpleITK as sitk

from nnfoundation_bench.datasets import get_task
from nnfoundation_bench.datasets.mrnet import (
    PLANES,
    TARGETS,
    exam_split,
    npy_to_image,
    read_label_csv,
    target_labels,
)
from nnfoundation_bench.preprocessing import crop_around, image_center


def test_label_csv_first_row_is_not_a_header(tmp_path):
    # Redivis export: no header, values prefixed with "_".
    path = tmp_path / "redivis-train-acl.csv"
    path.write_text("_0000,_1\n1049,0\n_0806,_1\n")
    labels = read_label_csv(path)
    assert labels.to_dict("list") == {"case_id": ["0000", "1049", "0806"], "label": [1, 0, 1]}


def test_target_labels_selects_one_column():
    labels = pd.DataFrame(
        {"case_id": ["0000", "0001"], "abnormal": [1, 1], "acl": [0, 1], "meniscus": [0, 0]}
    )
    assert target_labels(labels, "acl").to_dict("list") == {
        "case_id": ["0000", "0001"],
        "label": [0, 1],
    }


def test_exam_split():
    assert exam_split("0000") == "train"
    assert exam_split("1129") == "train"
    assert exam_split("1130") == "valid"


def test_all_tasks_registered():
    for target in TARGETS:
        for plane in PLANES:
            task = get_task(f"mrnet_{target}_{plane}")
            assert task.modality == "MR" and task.target == target and task.pad_value == 0


def _marker_volume(shape=(30, 256, 256)):
    """Volume with a marker at slice 0 / row 0 / col 0 corners to track the axis mapping."""
    array = np.zeros(shape, np.uint8)
    array[0, 100:156, 100:156] = 50  # first slice
    array[10:20, 0:20, 100:156] = 100  # top rows
    array[10:20, 100:156, 0:20] = 200  # left columns
    return array


def _where(crop: np.ndarray, value: int) -> np.ndarray:
    return np.argwhere(np.isclose(crop, value, atol=1)).mean(0)


def test_plane_geometry_maps_rows_to_inferior_and_fits_crop():
    for plane in PLANES:
        image = npy_to_image(_marker_volume(), plane)
        crop = sitk.GetArrayFromImage(
            crop_around(
                image, image_center(image), interpolator=sitk.sitkNearestNeighbor, pad_value=0
            )
        )
        mid = 95.5
        # The whole series (160 x 160 x 120 mm) fits in the 192^3 crop.
        assert {0, 50, 100, 200} == set(np.unique(crop).astype(int))
        top = _where(crop, 100)  # (z, y, x): z = S-I, y = A-P, x = L-R
        if plane in ("sagittal", "coronal"):
            # array rows run superior -> inferior: row 0 is at the top (high z in LPS)
            assert top[0] > mid + 50
        else:
            # axial rows run anterior -> posterior: row 0 is anterior (low y in LPS)
            assert top[1] < mid - 50
            # axial slice 0 is the most superior
            assert _where(crop, 50)[0] > mid + 40
        left = _where(crop, 200)
        if plane == "sagittal":
            assert left[1] < mid - 50  # col 0 anterior

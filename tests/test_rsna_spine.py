import zipfile

import numpy as np
import pandas as pd
import SimpleITK as sitk

from nnfoundation_bench.datasets import get_task
from nnfoundation_bench.datasets.rsna_spine import (
    build_zip_index,
    fracture_labels,
    read_dicom_series,
)
from nnfoundation_bench.preprocessing import binary_roi


def test_fracture_label_is_patient_overall():
    table = pd.DataFrame(
        {"StudyInstanceUID": ["1.2.3", "1.2.4"], "patient_overall": [1, 0], "C1": [1, 0]}
    )
    assert fracture_labels(table).to_dict("list") == {
        "case_id": ["1.2.3", "1.2.4"],
        "label": [1, 0],
    }


def test_zip_index_groups_train_slices_by_study(tmp_path):
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name in ("train.csv", "train_images/1.2.3/1.dcm", "train_images/1.2.3/2.dcm",
                     "train_images/1.2.4/1.dcm", "test_images/9/1.dcm",
                     "segmentations/1.2.3.nii"):  # fmt: skip
            archive.writestr(name, b"")
    with zipfile.ZipFile(path) as archive:
        index = build_zip_index(archive)
    assert {k: sorted(v) for k, v in index.items()} == {
        "1.2.3": ["train_images/1.2.3/1.dcm", "train_images/1.2.3/2.dcm"],
        "1.2.4": ["train_images/1.2.4/1.dcm"],
    }


def test_read_dicom_series_sorts_slices(tmp_path):
    volume = sitk.GetImageFromArray(np.arange(4, dtype=np.int16)[:, None, None] * np.ones(
        (4, 8, 8), np.int16))  # fmt: skip
    volume.SetSpacing((0.5, 0.5, 2.0))
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()  # else every file gets its own series UID
    for z in range(4):
        image = volume[:, :, z]
        image.SetMetaData("0020|000d", "1.2.826.0.1.2")  # study
        image.SetMetaData("0020|000e", "1.2.826.0.1.1")  # same series
        image.SetMetaData("0008|0018", f"1.2.826.0.1.3.{z}")  # SOP instance
        image.SetMetaData("0020|0032", f"0\\0\\{2.0 * z}")  # image position
        image.SetMetaData("0020|0037", "1\\0\\0\\0\\1\\0")
        image.SetMetaData("0008|0060", "CT")
        # Write in reverse order under shuffled names; GDCM sorts by position.
        writer.SetFileName(str(tmp_path / f"{3 - z}.dcm"))
        writer.Execute(image)
    read = read_dicom_series(tmp_path)
    assert read.GetSize() == (8, 8, 4)
    assert sitk.GetArrayFromImage(read)[:, 0, 0].tolist() == [0, 1, 2, 3]


def test_binary_roi_takes_union_of_labels():
    mask = sitk.GetImageFromArray(np.array([[[0, 1, 2, 3]]], np.uint8))
    assert sitk.GetArrayFromImage(binary_roi(mask, (1, 3))).ravel().tolist() == [0, 1, 0, 1]
    assert sitk.GetArrayFromImage(binary_roi(mask, 2)).ravel().tolist() == [0, 0, 1, 0]


def test_task_registered():
    assert get_task("rsna_spine_fracture").modality == "CT"

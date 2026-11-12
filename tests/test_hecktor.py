import numpy as np
import pandas as pd
import SimpleITK as sitk

from nnfoundation_bench.datasets import get_task
from nnfoundation_bench.datasets.base import CaseInputs
from nnfoundation_bench.datasets.hecktor import (
    LABELS_HPV_COLUMN,
    LABELS_ID_COLUMN,
    hpv_labels,
    patient_id,
)
from nnfoundation_bench.preprocessing import _process_case


def test_patient_id_strips_curation_index():
    assert patient_id("CHUM-029-86") == "CHUM-029"
    assert patient_id("HGJ-089-102") == "HGJ-089"


def test_hpv_labels_drop_unknown_and_unlisted():
    table = pd.DataFrame(
        {LABELS_ID_COLUMN: ["CHUM-029", "MDA-050", "HGJ-010"], LABELS_HPV_COLUMN: [1, 0, None]}
    )
    labels = hpv_labels(table, ["CHUM-029-86", "MDA-050-55", "HGJ-010-83", "CHUP-003-17"])
    assert labels.to_dict("list") == {
        "case_id": ["CHUM-029-86", "MDA-050-55"],
        "label": [1, 0],
        "group": ["CHUM", "MDA"],
        "stratum": ["CHUM_1", "MDA_0"],
    }


def test_task_registered():
    task = get_task("hecktor_hpv_pet_gtvp")
    assert task.modality == "PT" and task.roi == "GTVp" and task.require_roi


def test_pet_crop_centred_on_ct_grid_mask(tmp_path):
    # PET at 4 mm with a hot spot; GTV mask defined on a 1 mm "CT" grid.
    pet = sitk.GetImageFromArray(np.zeros((40, 40, 40), np.float32))
    pet.SetSpacing((4.0, 4.0, 4.0))
    arr = sitk.GetArrayFromImage(pet)
    arr[20, 20, 20] = 10.0
    hot = sitk.GetImageFromArray(arr)
    hot.CopyInformation(pet)
    center = hot.TransformIndexToPhysicalPoint((20, 20, 20))

    ct_mask = sitk.Image((160, 160, 160), sitk.sitkUInt8)
    idx = ct_mask.TransformPhysicalPointToIndex(center)
    ct_mask[idx] = 1
    sitk.WriteImage(hot, tmp_path / "pet.nii.gz")
    sitk.WriteImage(ct_mask, tmp_path / "gtvp.nii.gz")
    for d in ("out/images", "out/masks", "qc/images", "qc/masks"):
        (tmp_path / d).mkdir(parents=True)

    found = _process_case(
        CaseInputs(tmp_path / "pet.nii.gz", tmp_path / "gtvp.nii.gz"),
        0.0, tmp_path / "out", tmp_path / "qc", "c",
    )  # fmt: skip
    assert found
    qc = sitk.GetArrayFromImage(sitk.ReadImage(tmp_path / "qc" / "images" / "c.nii.gz"))
    peak = np.unravel_index(np.argmax(qc), qc.shape)
    assert all(abs(p - 95.5) <= 2 for p in peak)

    no_roi = CaseInputs(tmp_path / "pet.nii.gz", None)
    assert not _process_case(no_roi, 0.0, tmp_path / "out", tmp_path / "qc", "d")

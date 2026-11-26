"""TotalSegmentator CT/MRI organ recognition (issue #4): dataset helpers and an end-to-end
``prepare`` on tiny synthetic scans. No real data, no network."""

import json
import socket
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

CT_TASK = "totalseg_ct_organs"
MRI_TASK = "totalseg_mri_organs"


def _ts():
    from nnfoundation_bench.datasets import totalsegmentator

    return totalsegmentator


# ---------------------------------------------------------------- registration, constants


def test_organ_tasks_are_registered_with_official_variants():
    from nnfoundation_bench.datasets import TASKS, get_task

    for name in (CT_TASK, MRI_TASK):
        task = get_task(name)
        assert task.organ_task is True and task.is_multiclass is True
        assert task.split_of is None
        official = get_task(f"{name}_official")
        assert official.split_of == name
        assert official.organ_task is True and official.is_multiclass is True
    assert {CT_TASK, MRI_TASK, f"{CT_TASK}_official", f"{MRI_TASK}_official"} <= set(TASKS)


def test_ct_and_mri_tasks_have_their_source_and_pad_values():
    from nnfoundation_bench.config import CT_PAD_VALUE

    ts = _ts()
    ct, mri = ts.TOTALSEG_CT_ORGANS, ts.TOTALSEG_MRI_ORGANS
    assert ct.name == CT_TASK and ct.source == "totalsegmentator_ct"
    assert ct.modality == "CT"
    assert ct.pad_value == CT_PAD_VALUE == -1024
    assert ct.intensity_min == CT_PAD_VALUE
    assert mri.name == MRI_TASK and mri.source == "totalsegmentator_mri"
    assert mri.modality in ("MR", "MRI")
    assert ts.MR_PAD_VALUE == 0.0 and mri.pad_value == 0.0


def test_min_organ_volume_constant():
    from nnfoundation_bench.config import MIN_ORGAN_MM3

    assert MIN_ORGAN_MM3 == 1000


def test_existing_tasks_are_not_organ_tasks():
    from nnfoundation_bench.datasets import TASKS

    binary = [t for name, t in TASKS.items() if not name.startswith("totalseg_")]
    assert len(binary) >= 38
    for task in binary:
        assert task.organ_task is False, task.name
        assert task.is_multiclass is False, task.name
        assert task.load_scans is None and task.scan_paths is None
        assert task.organ_classes is None


def test_classification_task_new_fields_default_to_off():
    from nnfoundation_bench.datasets.base import ClassificationTask

    task = ClassificationTask(
        name="t", source="s", description="d", target="y", roi="r", modality="CT",
        download=lambda: None, load_labels=lambda: pd.DataFrame(),
        case_inputs=lambda c, d: None, pad_value=0.0,
    )  # fmt: skip
    assert task.organ_task is False and task.is_multiclass is False
    assert task.load_scans is None and task.scan_paths is None and task.organ_classes is None


def test_official_variant_of_an_organ_task_stays_an_organ_task():
    from nnfoundation_bench.datasets.base import official_variant

    variant = official_variant(_ts().TOTALSEG_MRI_ORGANS)
    assert variant.name == f"{MRI_TASK}_official" and variant.split_of == MRI_TASK
    assert variant.organ_task is True and variant.pad_value == 0.0


# ---------------------------------------------------------------- meta and class sets


def test_read_meta_handles_bom_and_semicolons(tmp_path):
    path = tmp_path / "meta.csv"
    path.write_text(
        "﻿image_id;age;gender;institute;study_type;split\n"
        "s0001;55;m;A;ct abdomen;train\n"
        "s0002;71;f;B;ct thorax;test\n",
        encoding="utf-8",
    )
    meta = _ts().read_meta(path)
    assert meta.columns[0] == "scan_id"
    assert "image_id" not in meta.columns
    assert not any(c.startswith("﻿") for c in meta.columns)
    assert meta["scan_id"].tolist() == ["s0001", "s0002"]
    assert meta["split"].tolist() == ["train", "test"]
    assert meta["institute"].tolist() == ["A", "B"]
    assert meta["age"].astype(int).tolist() == [55, 71]


def test_structure_classes_lists_every_nifti_sorted(tmp_path):
    seg = tmp_path / "segmentations"
    seg.mkdir()
    for name in ("spleen", "aorta", "vertebrae_L1", "heart"):
        (seg / f"{name}.nii.gz").write_bytes(b"")
    (seg / "notes.txt").write_text("not a structure")
    assert _ts().structure_classes(seg) == ("aorta", "heart", "spleen", "vertebrae_L1")


def test_ct_v1_classes_are_the_98_shared_names_plus_heart():
    from totalsegmentator.map_to_binary import class_map

    v1 = set(class_map["total_v1"].values())
    v2 = set(class_map["total"].values())
    classes = _ts().ct_v1_classes(v2)
    assert isinstance(classes, tuple)
    assert len(classes) == 98
    assert list(classes) == sorted(classes)
    assert set(classes) == (v1 & v2) | {"heart"}
    assert "heart" in classes and "liver" in classes
    for absent in ("skull", "face", "heart_atrium_left", "pulmonary_artery", "sternum"):
        assert absent not in classes


def test_ct_v1_classes_on_a_subset_follows_the_formula():
    # skull is v2-only, brain and liver are in both; heart is always added.
    assert _ts().ct_v1_classes({"liver", "skull", "brain"}) == ("brain", "heart", "liver")


# ---------------------------------------------------------------- organ_samples


def _organs():
    return pd.DataFrame(
        {
            "scan_id": ["s2", "s2", "s1", "s1", "s1", "s1", "s2", "s1"],
            "organ": ["liver", "aorta", "liver", "aorta", "skull", "kidney_left",
                      "kidney_left", "heart"],
            "voxels": [5000, 999, 1000, 20000, 90000, 10, 500, 3000],
        }
    )  # fmt: skip


def test_organ_samples_filters_volume_and_class_set_and_renumbers():
    classes = ("aorta", "heart", "kidney_left", "liver")
    samples = _ts().organ_samples(_organs(), classes, min_mm3=1000)
    assert list(samples.columns[:4]) == ["case_id", "scan_id", "organ", "label"]
    # s2 aorta (999 < 1000) is dropped, s1 liver (== 1000) is kept, skull is not in the
    # class set, kidney_left has no sample left and is dropped from the labels.
    assert samples["case_id"].tolist() == ["s1__aorta", "s1__heart", "s1__liver", "s2__liver"]
    assert samples["scan_id"].tolist() == ["s1", "s1", "s1", "s2"]
    assert samples["organ"].tolist() == ["aorta", "heart", "liver", "liver"]
    assert samples["label"].tolist() == [0, 1, 2, 2]  # aorta, heart, liver: 0..K-1
    assert samples["case_id"].is_monotonic_increasing


def test_organ_samples_output_columns_and_official_split():
    organs = _organs().assign(
        official_split=lambda d: d["scan_id"].map({"s1": "train", "s2": "test"})
    )
    samples = _ts().organ_samples(organs, ("aorta", "heart", "liver"), min_mm3=1000)
    assert list(samples.columns) == ["case_id", "scan_id", "organ", "label", "official_split"]
    assert samples["official_split"].tolist() == ["train", "train", "train", "test"]


def test_organ_samples_label_follows_class_order_not_row_order():
    organs = pd.DataFrame(
        {"scan_id": ["a", "a", "b"], "organ": ["spleen", "brain", "spleen"],
         "voxels": [2000, 2000, 2000]}
    )  # fmt: skip
    samples = _ts().organ_samples(organs, ("brain", "liver", "spleen"), min_mm3=1000)
    assert dict(zip(samples["case_id"], samples["label"], strict=True)) == {
        "a__brain": 0, "a__spleen": 1, "b__spleen": 1,
    }  # fmt: skip


# ---------------------------------------------------------------- download / load_labels


def test_download_does_not_use_network_and_points_to_zenodo(tmp_path, monkeypatch, capfd):
    ts = _ts()
    for name in ("RAW_DATA_DIR", "RAW_DIR"):
        monkeypatch.setattr(ts, name, tmp_path / "raw", raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("download() must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    for task, record, md5 in (
        (ts.TOTALSEG_CT_ORGANS, "10047292", "fe250e5718e0a3b5df4c4ea9d58a62fe"),
        (ts.TOTALSEG_MRI_ORGANS, "14710732", "54638f4cb883ce3b34225195358c398f"),
    ):
        text = ""
        try:
            task.download()
        except AssertionError:
            raise
        except Exception as e:  # noqa: BLE001 - raising with the instructions is allowed
            text += str(e)
        out = capfd.readouterr()
        text += out.out + out.err
        assert record in text and md5 in text


def test_load_labels_before_prepare_raises_a_clear_error(tmp_path, monkeypatch):
    ts = _ts()
    monkeypatch.setattr(ts, "PROCESSED_DATA_DIR", tmp_path / "processed", raising=False)
    with pytest.raises(Exception, match=r"(?i)prepare"):
        ts.TOTALSEG_CT_ORGANS.load_labels()


# ---------------------------------------------------------------- end-to-end prepare

# Native grid of the synthetic scans: (x, y, z) size and spacing in mm.
SIZE_XYZ = (24, 24, 16)
SPACING = (1.5, 1.5, 2.0)
STRUCTURES = ("aorta", "heart", "liver", "skull", "spleen", "tiny_gland")  # ids 1..6
CLASSES = ("aorta", "heart", "liver", "spleen", "tiny_gland")  # skull: not a class
VALUES = {"liver": 60.0, "spleen": 45.0, "aorta": 30.0, "heart": 35.0, "skull": 500.0,
          "tiny_gland": 10.0}  # fmt: skip
OVERLAP_SCAN, OVERLAP_VOXELS = "s0003", 6 * 2 * 2
EMPTY_SPLEEN_SCAN = "s0005"
FLIPPED_SCAN = "s0002"
GOOD_SCANS = [f"s{i:04d}" for i in range(1, 11)]
CORRUPT_SCAN, MISMATCHED_SCAN = "s0011", "s0012"
SPLITS = {**{s: "train" for s in GOOD_SCANS[:6]}, **{s: "val" for s in GOOD_SCANS[6:8]},
          **{s: "test" for s in GOOD_SCANS[8:]}, CORRUPT_SCAN: "train",
          MISMATCHED_SCAN: "test"}  # fmt: skip


def _boxes(scan: str) -> dict[str, tuple[slice, slice, slice]]:
    """(z, y, x) boxes of every structure (8 x 8 x 6 voxels = 1728 mm^3)."""
    heart = (
        (slice(9, 15), slice(8, 16), slice(8, 16))  # overlaps aorta in 6 x 2 x 2 voxels
        if scan == OVERLAP_SCAN
        else (slice(9, 15), slice(10, 18), slice(10, 18))
    )
    return {
        "liver": (slice(2, 8), slice(2, 10), slice(2, 10)),
        "spleen": (slice(2, 8), slice(2, 10), slice(14, 22)),
        "aorta": (slice(9, 15), slice(2, 10), slice(2, 10)),
        "heart": heart,
        "skull": (slice(2, 8), slice(14, 22), slice(2, 10)),
        "tiny_gland": (slice(2, 4), slice(14, 16), slice(14, 16)),  # 36 mm^3
    }


def _geometry(image: sitk.Image, scan: str) -> sitk.Image:
    image.SetSpacing(SPACING)
    image.SetOrigin((-20.0, 30.0, 5.0))
    if scan == FLIPPED_SCAN:  # x axis stored right-to-left
        image.SetDirection((-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return image


def _write_scan(root: Path, scan: str) -> None:
    scan_dir = root / scan
    seg_dir = scan_dir / "segmentations"
    seg_dir.mkdir(parents=True)
    shape = SIZE_XYZ[::-1]
    image = np.full(shape, -1000.0, np.float32)
    boxes = _boxes(scan)
    for name in STRUCTURES:
        mask = np.zeros(shape, np.uint8)
        if not (scan == EMPTY_SPLEEN_SCAN and name == "spleen"):
            mask[boxes[name]] = 1
            image[boxes[name]] = VALUES[name]
        if scan == MISMATCHED_SCAN and name == "spleen":
            continue  # a scan without spleen.nii.gz: different structure set
        sitk.WriteImage(_geometry(sitk.GetImageFromArray(mask), scan), seg_dir / f"{name}.nii.gz")
    if scan == OVERLAP_SCAN:  # aorta was written over by heart in the image; keep aorta value
        image[boxes["aorta"]] = VALUES["aorta"]
    if scan == CORRUPT_SCAN:
        (scan_dir / "ct.nii.gz").write_bytes(b"this is not a nifti file")
    else:
        sitk.WriteImage(_geometry(sitk.GetImageFromArray(image), scan), scan_dir / "ct.nii.gz")


def _native_mean(root: Path, scan: str) -> float:
    return float(sitk.GetArrayFromImage(sitk.ReadImage(root / scan / "ct.nii.gz")).mean())


@pytest.fixture
def synthetic_ct(tmp_path, monkeypatch):
    """12 synthetic CT scans (10 good, 1 corrupt, 1 with a different structure set) and
    an organ task whose load_scans / scan_paths / organ_classes point at them."""
    from nnfoundation_bench import preprocessing

    ts = _ts()
    raw = tmp_path / "raw_ct"
    scans = [*GOOD_SCANS, CORRUPT_SCAN, MISMATCHED_SCAN]
    for scan in scans:
        _write_scan(raw, scan)
    meta = pd.DataFrame(
        {
            "scan_id": scans,
            "age": [40 + i for i in range(len(scans))],
            "gender": ["m", "f"] * (len(scans) // 2),
            "institute": ["A", "B", "C"] * (len(scans) // 3),
            "study_type": ["ct abdomen"] * len(scans),
            "split": [SPLITS[s] for s in scans],
        }
    )
    processed, interim = tmp_path / "processed", tmp_path / "interim"
    monkeypatch.setattr(preprocessing, "PROCESSED_DATA_DIR", processed)
    monkeypatch.setattr(preprocessing, "INTERIM_DATA_DIR", interim)
    for name, value in (("PROCESSED_DATA_DIR", processed), ("INTERIM_DATA_DIR", interim),
                        ("RAW_DATA_DIR", tmp_path / "raw")):  # fmt: skip
        monkeypatch.setattr(ts, name, value, raising=False)
    task = replace(
        ts.TOTALSEG_CT_ORGANS,
        load_scans=lambda: meta.copy(),
        scan_paths=lambda scan: (raw / scan / "ct.nii.gz", raw / scan / "segmentations"),
        organ_classes=lambda: CLASSES,
    )
    return task, raw, processed / CT_TASK


def _read_b2nd(path: Path) -> np.ndarray:
    import blosc2

    return blosc2.open(str(path), mode="r")[...]


def _expected_cases() -> set[str]:
    organs = ("aorta", "heart", "liver", "spleen")
    return {
        f"{s}__{o}" for s in GOOD_SCANS for o in organs
        if not (s == EMPTY_SPLEEN_SCAN and o == "spleen")
    }  # fmt: skip


def _classes_map(value) -> dict[int, str]:
    if isinstance(value, dict):
        return {int(k): v for k, v in value.items()}
    return dict(enumerate(value))


def _prepare(task, **kwargs):
    from nnfoundation_bench.preprocessing import prepare_task

    prepare_task(task, device="cpu", seed=0, n_splits=3, **kwargs)


def test_prepare_scan_task_is_exported():
    from nnfoundation_bench import preprocessing

    assert callable(preprocessing.prepare_scan_task)


def test_prepare_writes_one_whole_scan_image_and_label_map_per_scan(synthetic_ct):
    task, _raw, out = synthetic_ct
    _prepare(task)
    images = sorted(p.stem for p in (out / "images").glob("*.b2nd"))
    masks = sorted(p.stem for p in (out / "masks").glob("*.b2nd"))
    assert images == masks == GOOD_SCANS  # per scan, not per sample; failed scans absent
    extent = [(n - 1) * s for n, s in zip(SIZE_XYZ, SPACING, strict=True)][::-1]  # z, y, x
    for scan in GOOD_SCANS:
        image = _read_b2nd(out / "images" / f"{scan}.b2nd")
        mask = _read_b2nd(out / "masks" / f"{scan}.b2nd")
        assert image.dtype == np.float32 and mask.dtype == np.uint8
        assert image.ndim == 4 and image.shape[0] == 1 and image.shape == mask.shape
        for n, e in zip(image.shape[1:], extent, strict=True):
            assert abs(n - (e + 1)) <= 1.5, (image.shape, extent)  # whole extent at 1 mm
        assert image.shape[1] < 192  # not a 192^3 crop
        assert set(np.unique(mask)) <= set(range(len(STRUCTURES) + 1))


def test_label_map_ids_follow_sorted_structure_names(synthetic_ct):
    task, _raw, out = synthetic_ct
    _prepare(task)
    scans = pd.read_csv(out / "scans.csv", dtype={"scan_id": str}).set_index("scan_id")
    for scan in ("s0001", FLIPPED_SCAN):
        image = _read_b2nd(out / "images" / f"{scan}.b2nd")[0]
        mask = _read_b2nd(out / "masks" / f"{scan}.b2nd")[0]
        mean, std = scans.loc[scan, "mean"], scans.loc[scan, "std"]
        for label_id, name in enumerate(STRUCTURES, start=1):
            if name == "tiny_gland":
                continue
            voxels = mask == label_id
            assert 1400 <= voxels.sum() <= 2100, (name, voxels.sum())  # ~1728 mm^3
            # The z-scored image holds the structure's intensity under its label.
            physical = image[voxels] * std + mean
            assert np.median(physical) == pytest.approx(VALUES[name], abs=1.0)
    # skull is not a class but stays in the label map (id 4).
    assert (_read_b2nd(out / "masks" / "s0001.b2nd") == 4).sum() > 1000


def test_prepare_reorients_to_lps_identity(synthetic_ct):
    task, _raw, out = synthetic_ct
    _prepare(task)

    def mean_x(scan, label_id):
        mask = _read_b2nd(out / "masks" / f"{scan}.b2nd")[0]
        return np.nonzero(mask == label_id)[2].mean()

    liver, spleen = STRUCTURES.index("liver") + 1, STRUCTURES.index("spleen") + 1
    # Liver at low native x index in both scans; s0002 stores x right-to-left, so in
    # physical (LPS) space its liver lies at higher x than its spleen.
    assert mean_x("s0001", liver) < mean_x("s0001", spleen)
    assert mean_x(FLIPPED_SCAN, liver) > mean_x(FLIPPED_SCAN, spleen)


def test_scans_csv_has_normalisation_and_meta_columns(synthetic_ct):
    task, raw, out = synthetic_ct
    _prepare(task)
    scans = pd.read_csv(out / "scans.csv", dtype={"scan_id": str}).set_index("scan_id")
    assert {"shape", "mean", "std", "pad_z", "age", "institute", "study_type", "split"} <= set(
        scans.columns
    )
    assert set(GOOD_SCANS) <= set(scans.index)
    for scan in GOOD_SCANS:
        row = scans.loc[scan]
        assert row["pad_z"] == pytest.approx((-1024 - row["mean"]) / row["std"], rel=1e-4)
        assert row["mean"] == pytest.approx(_native_mean(raw, scan), rel=0.05)
        assert row["split"] == SPLITS[scan]


def test_organs_csv_logs_volumes_ids_and_overlap(synthetic_ct):
    task, _raw, out = synthetic_ct
    _prepare(task)
    organs = pd.read_csv(out / "organs.csv", dtype={"scan_id": str})
    assert {"scan_id", "organ", "label_id", "voxels", "overlap_voxels"} <= set(organs.columns)
    ids = organs.drop_duplicates("organ").set_index("organ")["label_id"].to_dict()
    for name, label_id in ids.items():
        assert label_id == STRUCTURES.index(name) + 1
    s1 = organs[organs["scan_id"] == "s0001"].set_index("organ")
    assert "skull" in s1.index  # every structure of the label map is logged
    assert 1400 <= s1.loc["liver", "voxels"] <= 2100
    overlap = organs.groupby("scan_id")["overlap_voxels"].max()
    assert overlap[OVERLAP_SCAN] == OVERLAP_VOXELS
    assert (overlap.drop(OVERLAP_SCAN) == 0).all()


def test_overlapping_voxels_go_to_the_first_class(synthetic_ct):
    task, _raw, out = synthetic_ct
    _prepare(task)
    organs = pd.read_csv(out / "organs.csv", dtype={"scan_id": str})
    vol = organs[organs["scan_id"] == OVERLAP_SCAN].set_index("organ")["voxels"]
    # aorta (first by name) keeps its full volume, heart loses the shared voxels.
    assert vol["aorta"] > vol["heart"]
    assert vol["aorta"] - vol["heart"] == pytest.approx(OVERLAP_VOXELS * 4.5, rel=0.35)


def test_labels_are_per_organ_sample_with_volume_and_class_filters(synthetic_ct):
    task, _raw, out = synthetic_ct
    _prepare(task)
    labels = json.loads((out / "labels.json").read_text())
    assert set(labels) == _expected_cases()
    expected_label = {"aorta": 0, "heart": 1, "liver": 2, "spleen": 3}
    for case, label in labels.items():
        _scan, organ = case.split("__")
        assert label == expected_label[organ]
    table = pd.read_csv(out / "labels.csv", dtype={"case_id": str, "scan_id": str})
    assert {"case_id", "scan_id", "organ", "label"} <= set(table.columns)
    # meta columns go to scans.csv, not labels.csv (no age / centre references).
    assert not {"age", "institute", "group", "stratum", "study_type"} & set(table.columns)
    assert set(table["case_id"]) == set(labels)
    row = table.set_index("case_id").loc["s0004__liver"]
    assert row["scan_id"] == "s0004" and row["organ"] == "liver" and row["label"] == 2


def test_dataset_json_records_scan_layout_classes_and_failures(synthetic_ct):
    task, _raw, out = synthetic_ct
    _prepare(task)
    summary = json.loads((out / "dataset.json").read_text())
    assert summary["layout"] == "scan"
    assert summary["num_classes"] == 4
    assert _classes_map(summary["classes"]) == {0: "aorta", 1: "heart", 2: "liver", 3: "spleen"}
    # tiny_gland (36 mm^3 everywhere) has no sample: dropped and listed.
    assert "tiny_gland" in json.dumps({k: v for k, v in summary.items() if k != "classes"})
    assert "tiny_gland" not in _classes_map(summary["classes"]).values()
    # Corrupt image and different structure set: listed as failed, the run went on.
    assert CORRUPT_SCAN in summary["failed"] and MISMATCHED_SCAN in summary["failed"]
    assert "s0001" not in summary["failed"]
    # Per-class sample counts (keyed by name or id).
    counts = {"aorta": 10, "heart": 10, "liver": 10, "spleen": 9}
    by_id = {str(i): n for i, n in enumerate(counts.values())}
    found = [
        v for v in summary.values()
        if isinstance(v, dict) and {str(k): v2 for k, v2 in v.items()} in (counts, by_id)
    ]  # fmt: skip
    assert found, "dataset.json has no per-class sample counts"


def test_splits_are_drawn_on_scans(synthetic_ct):
    task, _raw, out = synthetic_ct
    _prepare(task)
    labels = json.loads((out / "labels.json").read_text())
    splits = json.loads((out / "splits.json").read_text())
    assert len(splits) == 3
    for split in splits:
        assert set(split) == {"train", "val", "test"}
        cases = [c for v in split.values() for c in v]
        assert len(cases) == len(set(cases)) and set(cases) == set(labels)
        scans_of = {k: {c.split("__")[0] for c in v} for k, v in split.items()}
        assert not scans_of["train"] & scans_of["val"]
        assert not scans_of["train"] & scans_of["test"]
        assert not scans_of["val"] & scans_of["test"]
        assert len(scans_of["test"]) == 5 and len(scans_of["val"]) in (1, 2)
    assert len({tuple(s["test"]) for s in splits}) > 1  # one seed per fold


def test_rerun_skips_finished_scans_and_gives_the_same_outputs(synthetic_ct):
    task, _raw, out = synthetic_ct
    _prepare(task)
    image = out / "images" / "s0001.b2nd"
    mtime = image.stat().st_mtime_ns
    first = {n: (out / n).read_text() for n in ("labels.json", "splits.json")}
    _prepare(task)
    assert image.stat().st_mtime_ns == mtime
    assert {n: (out / n).read_text() for n in first} == first
    summary = json.loads((out / "dataset.json").read_text())
    assert CORRUPT_SCAN in summary["failed"]


def test_load_labels_reads_the_prepared_samples(synthetic_ct):
    task, _raw, out = synthetic_ct
    _prepare(task)
    labels = _ts().TOTALSEG_CT_ORGANS.load_labels()
    assert {"case_id", "label"} <= set(labels.columns)
    prepared = json.loads((out / "labels.json").read_text())
    assert dict(zip(labels["case_id"], labels["label"].astype(int), strict=True)) == prepared


def test_official_variant_uses_meta_split_grouped_by_scan(synthetic_ct):
    from nnfoundation_bench.datasets.base import official_variant

    task, _raw, out = synthetic_ct
    _prepare(task)
    variant = official_variant(task)
    _prepare(variant)
    var_dir = out.parent / variant.name
    (split,) = json.loads((var_dir / "splits.json").read_text())
    scans_of = {k: {c.split("__")[0] for c in v} for k, v in split.items()}
    assert scans_of == {
        "train": set(GOOD_SCANS[:6]), "val": set(GOOD_SCANS[6:8]), "test": set(GOOD_SCANS[8:])
    }  # fmt: skip
    labels = json.loads((out / "labels.json").read_text())
    assert sorted(c for v in split.values() for c in v) == sorted(labels)
    assert json.loads((var_dir / "labels.json").read_text()) == labels
    for name in ("images", "masks", "features"):
        assert (var_dir / name).is_symlink() and (var_dir / name).resolve() == out / name

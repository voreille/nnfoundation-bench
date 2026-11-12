"""HECKTOR head-and-neck PET/CT, curated subset for HPV status (EUVIP 2024).

The data is registration-gated and comes as DICOM: one folder per patient with a CT
series, a PET series and a DICOM SEG holding GTVp (segment 1) and GTVn (segment 2),
defined on the CT. Each patient is converted once with okapy (CT in HU, PET in SUVbw,
SEG -> one mask per segment on the CT grid) and cached under
``data/interim/hecktor_euvip2024_hpv/converted/<case_id>/``.
"""

import json
import shutil

import pandas as pd
from loguru import logger

from nnfoundation_bench.config import INTERIM_DATA_DIR, PET_PAD_VALUE, RAW_DATA_DIR
from nnfoundation_bench.datasets.base import CaseInputs, ClassificationTask

SOURCE = "hecktor_euvip2024_hpv"
RAW_DIR = RAW_DATA_DIR / SOURCE
CONVERTED_DIR = INTERIM_DATA_DIR / SOURCE / "converted"

# ASSUMPTION until the curated label file is available: HECKTOR 2022 clinical-info
# conventions (PatientID like "CHUM-029", HPV coded 0/1, empty = unknown).
LABELS_PATH = RAW_DIR / "hpv_status.csv"
LABELS_ID_COLUMN = "PatientID"
LABELS_HPV_COLUMN = "HPV status (0=-, 1=+)"


def download() -> None:
    """Nothing to download: the curated copy must be placed in data/raw by hand."""
    if not RAW_DIR.is_dir() or not any(RAW_DIR.iterdir()):
        raise FileNotFoundError(
            f"{RAW_DIR} is missing. HECKTOR is registration-gated; copy the curated "
            "EUVIP 2024 HPV subset (one DICOM folder per patient) there."
        )
    logger.info(f"{len(case_ids())} patient folders in {RAW_DIR}")


def case_ids() -> list[str]:
    return sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir())


def patient_id(case_id: str) -> str:
    """Folder "CHUM-029-86" -> DICOM PatientID "CHUM-029" (the suffix is a curation index)."""
    return "-".join(case_id.split("-")[:2])


def convert_case(case_id: str) -> dict:
    """Convert one patient with okapy (cached). Returns paths relative to its folder.

    Manifest: {"CT": path, "PT": path, "masks": {"GTVp": path, "GTVn": path}}.
    """
    out_dir = CONVERTED_DIR / case_id
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())

    from okapy.dicom.conversion.builder import build_dicom_conversion_step

    tmp_dir = out_dir.with_name(f".partial.{case_id}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    step = build_dicom_conversion_step({"conversion": {"extension": "nii.gz"}})
    collection = step.run(RAW_DIR / case_id, output_dir=tmp_dir)

    images, masks = {}, {}
    for study in collection.studies:
        for image_set in study.image_mask_sets:
            modality = image_set.image.modality
            if modality in images:
                raise RuntimeError(f"{case_id}: more than one {modality} series")
            images[modality] = image_set.image.path
            for mask in image_set.masks:
                masks[mask.identity.label.strip()] = mask.path
    if set(images) != {"CT", "PT"}:
        raise RuntimeError(f"{case_id}: expected one CT and one PT series, got {sorted(images)}")

    manifest = {
        "CT": str(images["CT"].relative_to(tmp_dir)),
        "PT": str(images["PT"].relative_to(tmp_dir)),
        "masks": {label: str(path.relative_to(tmp_dir)) for label, path in masks.items()},
    }
    (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.rename(out_dir)
    return manifest


def converted_inputs(case_id: str, image: str, roi: str) -> CaseInputs:
    manifest = convert_case(case_id)
    folder = CONVERTED_DIR / case_id
    mask = manifest["masks"].get(roi)
    return CaseInputs(folder / manifest[image], folder / mask if mask else None)


def hpv_labels(table: pd.DataFrame, cases: list[str]) -> pd.DataFrame:
    """Binary HPV status per case folder; cases with unknown status are dropped."""
    status = table.set_index(LABELS_ID_COLUMN)[LABELS_HPV_COLUMN]
    rows = [
        {"case_id": case, "label": int(status[patient_id(case)])}
        for case in cases
        if patient_id(case) in status.index and pd.notna(status[patient_id(case)])
    ]
    if dropped := len(cases) - len(rows):
        logger.warning(f"{dropped} cases without a known HPV status are excluded")
    return pd.DataFrame(rows, columns=["case_id", "label"])


HECKTOR_HPV_PET_GTVP = ClassificationTask(
    name="hecktor_hpv_pet_gtvp",
    source=SOURCE,
    description="HECKTOR PET (SUVbw), crop around GTVp, target: HPV positive",
    target="hpv_positive",
    roi="GTVp",
    modality="PT",
    download=download,
    load_labels=lambda: hpv_labels(pd.read_csv(LABELS_PATH), case_ids()),
    case_inputs=lambda case_id, device: converted_inputs(case_id, "PT", "GTVp"),
    pad_value=PET_PAD_VALUE,
)

"""Handcrafted radiomics baseline (``uv run nnfb-radiomics --help``).

``extract`` computes pyradiomics features inside the ROI of every prepared case. It reads
the QC crops written by ``prepare`` (``data/interim/<task>/{images,masks}/``): 1 mm
isotropic, physical units (HU / SUV / MR), and the same ROI mask the embedding poolings
use, so radiomics and the frozen-encoder probes see the same voxels. ``probe`` fits the
same logistic-regression probe as ``nnfb-features probe``, on the same splits, and
writes ``reports/results/<task>/radiomics_<features>_logreg/``.
"""

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import typer
from loguru import logger
from tqdm import tqdm

from nnfoundation_bench.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from nnfoundation_bench.datasets import get_task
from nnfoundation_bench.modeling.features import probe_folds, task_labels

app = typer.Typer(no_args_is_help=True, add_completion=False)

# Intensity discretisation per modality. CT and PET have physical units, so a fixed bin
# width keeps bins comparable across patients. MR intensities have no unit (and MRNet /
# LLD-MMRI were rescaled to 0-255 per series), so a fixed bin count is used instead.
DISCRETISATION = {
    "CT": {"binWidth": 25.0},  # HU
    "PT": {"binWidth": 0.5},  # SUVbw
    "MR": {"binCount": 32},
}
# Feature sets: image types given to pyradiomics. All feature classes are enabled.
FEATURE_SETS = {
    "original": {"Original": {}},
    "filtered": {"Original": {}, "LoG": {"sigma": [1.0, 3.0]}, "Wavelet": {}},
}


def radiomics_dir(task: str, feature_set: str) -> Path:
    return PROCESSED_DATA_DIR / task / "radiomics" / feature_set


def run_name(feature_set: str) -> str:
    return f"radiomics_{feature_set}_logreg"


def extractor_settings(modality: str) -> dict:
    """pyradiomics settings: crops are already 1 mm isotropic, so no resampling."""
    if modality not in DISCRETISATION:
        raise ValueError(f"No radiomics discretisation for modality {modality!r}")
    return {**DISCRETISATION[modality], "label": 1, "normalize": False, "force2D": False}


def to_real(name: str, value) -> float:
    """Feature value as float. With NumPy 2, pyradiomics' eigenvalue-based shape features
    (Elongation, Flatness, axis lengths) come back complex with a zero imaginary part."""
    value = complex(np.asarray(value).reshape(-1)[0])
    if abs(value.imag) > 1e-9 * max(1.0, abs(value.real)):
        raise ValueError(f"{name}: complex feature value {value}")
    return value.real


def extract_case(image: Path, mask: Path, settings: dict, image_types: dict) -> dict:
    """All pyradiomics features of one case (diagnostics dropped). Empty ROI -> {}."""
    import SimpleITK as sitk
    from radiomics import featureextractor

    logging.getLogger("radiomics").setLevel(logging.ERROR)
    mask_image = sitk.ReadImage(mask)
    if not sitk.GetArrayViewFromImage(mask_image).any():
        return {}
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableAllFeatures()
    extractor.enableImageTypes(**image_types)
    result = extractor.execute(sitk.ReadImage(image), mask_image)
    return {k: to_real(k, v) for k, v in result.items() if not k.startswith("diagnostics")}


def _extract_to_file(args: tuple) -> str | None:
    """Worker: extract one case and write its JSON. Returns an error message or None."""
    case, image, mask, out, settings, image_types = args
    try:
        features = extract_case(image, mask, settings, image_types)
    except Exception as e:  # noqa: BLE001 - one bad case must not stop the run
        return f"{case}: {type(e).__name__}: {e}"
    tmp = out.with_name(f".partial.{out.name}")
    tmp.write_text(json.dumps(features))
    tmp.rename(out)
    return None


@app.command()
def extract(
    task: str,
    features: str = typer.Option("original", help=f"Feature set: {', '.join(FEATURE_SETS)}"),
    jobs: int = typer.Option(8, help="Parallel worker processes"),
    overwrite: bool = typer.Option(False, help="Recompute existing features"),
) -> None:
    """Extract pyradiomics features inside the ROI of every case of TASK (cached per case)."""
    modality = get_task(task).modality
    settings, image_types = extractor_settings(modality), FEATURE_SETS[features]
    cases, _, _ = task_labels(task)
    qc_dir, out_dir = INTERIM_DATA_DIR / task, radiomics_dir(task, features)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs_args = [
        (c, qc_dir / "images" / f"{c}.nii.gz", qc_dir / "masks" / f"{c}.nii.gz",
         out_dir / f"{c}.json", settings, image_types)
        for c in cases
        if overwrite or not (out_dir / f"{c}.json").exists()
    ]  # fmt: skip
    logger.info(f"{task} / radiomics {features}: {len(jobs_args)} of {len(cases)} to extract")
    errors = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for error in tqdm(pool.map(_extract_to_file, jobs_args), total=len(jobs_args)):
            if error:
                logger.error(error)
                errors.append(error)
    (out_dir / "info.json").write_text(
        json.dumps(
            {
                "modality": modality,
                "settings": settings,
                "image_types": image_types,
                "source": "QC crops in data/interim/<task> (1 mm, physical units)",
                "failed": errors,
            },
            indent=2,
        )
    )
    if errors:
        logger.warning(f"{len(errors)} cases failed; they are imputed in the probe")
    logger.success(f"Radiomics features in {out_dir}")


def load_features(task: str, feature_set: str, cases: list[str]) -> tuple[np.ndarray, list]:
    """(cases x features) matrix; missing (empty ROI, failure) or infinite values are NaN."""
    folder = radiomics_dir(task, feature_set)
    rows = []
    for case in cases:
        path = folder / f"{case}.json"
        rows.append(json.loads(path.read_text()) if path.exists() else {})
    names = sorted({name for row in rows for name in row})
    if not names:
        raise FileNotFoundError(f"No radiomics features in {folder}; run extract first")
    x = np.array([[row.get(n, np.nan) for n in names] for row in rows], dtype=np.float64)
    x[~np.isfinite(x)] = np.nan  # inf (e.g. a ratio on a flat ROI) is imputed like NaN
    return x, names


@app.command()
def probe(
    task: str,
    features: str = typer.Option("original", help=f"Feature set: {', '.join(FEATURE_SETS)}"),
) -> None:
    """Logistic-regression probe on radiomics features, every fold."""
    cases, _, _ = task_labels(task)
    x, names = load_features(task, features, cases)
    missing = int(np.isnan(x).any(axis=1).sum())
    logger.info(f"{len(names)} features, {missing} of {len(cases)} cases with missing values")
    probe_folds(task, run_name(features), x)


if __name__ == "__main__":
    app()

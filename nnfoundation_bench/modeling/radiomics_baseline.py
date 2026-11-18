"""Handcrafted radiomics baseline (``uv run nnfb-radiomics --help``).

``extract`` computes pyradiomics features inside the ROI of every prepared case. It reads
the QC crops written by ``prepare`` (``data/interim/<task>/{images,masks}/``): 1 mm
isotropic, physical units (HU / SUV / MR), and the same ROI mask the embedding poolings
use, so radiomics and the frozen-encoder probes see the same voxels. ``probe`` fits the
same logistic-regression probe as ``nnfb-features probe``, on the same splits, and
writes ``outputs/results/<task>/radiomics_<features>[_corr<thr>]_<logreg|enet>/``.
Radiomics features are highly redundant, so ``probe`` can drop correlated features per
fold (``--corr-threshold``) and/or use an elastic net (``--classifier enet``).
"""

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import typer
from loguru import logger
from tqdm import tqdm

from nnfoundation_bench.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from nnfoundation_bench.datasets import get_task
from nnfoundation_bench.modeling.features import probe_folds, task_labels
from nnfoundation_bench.modeling.survival import probe_kind

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
# ``filtered`` is no longer run by default (D25): it never beat ``original`` except on
# HECKTOR, where the gain was centre signal.
FEATURE_SETS = {
    "original": {"Original": {}},
    "filtered": {"Original": {}, "LoG": {"sigma": [1.0, 3.0]}, "Wavelet": {}},
}


# Elastic-net L1 shares tried on val together with C (0 would be plain L2).
ENET_L1_RATIOS = (0.1, 0.5, 0.9)
CLASSIFIERS = ("logreg", "enet", "cox")


def radiomics_dir(task: str, feature_set: str) -> Path:
    return PROCESSED_DATA_DIR / task / "radiomics" / feature_set


def run_name(
    feature_set: str, corr_threshold: float | None = None, classifier: str = "logreg"
) -> str:
    """``radiomics_<set>[_corr<thr>]_<logreg|enet>``; the defaults keep the R1/R2 names."""
    corr = "" if corr_threshold is None else f"_corr{corr_threshold:g}"
    return f"radiomics_{feature_set}{corr}_{classifier}"


def correlation_filter(x: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """Indices of the features kept after removing redundant ones (fit on train only).

    Constant features are dropped first. Then, as in caret's ``findCorrelation``, while
    some pair has |Spearman rho| > ``threshold``, the most correlated pair is taken and the
    feature with the larger mean |rho| to all kept features is dropped. NaN are replaced
    by the column median before ranking.
    """
    from scipy.stats import rankdata

    x = np.where(np.isnan(x), np.nanmedian(x, axis=0, keepdims=True), x)
    x = np.nan_to_num(x)  # all-NaN columns become constant and are dropped below
    keep = np.flatnonzero(np.ptp(x, axis=0) > 0)
    if len(keep) < 2:
        return keep
    corr = np.abs(np.corrcoef(rankdata(x[:, keep], axis=0), rowvar=False))
    np.fill_diagonal(corr, 0.0)
    alive = np.ones(len(keep), bool)
    while True:
        sub = np.where(alive[:, None] & alive[None, :], corr, 0.0)
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        if sub[i, j] <= threshold:
            break
        mean_corr = sub.sum(axis=1) / (alive.sum() - 1)
        alive[i if mean_corr[i] >= mean_corr[j] else j] = False
    return keep[alive]


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
    """All pyradiomics features of one case (diagnostics dropped). Empty ROI -> {}.

    The ROI is the union of all mask values (masks may keep labels, e.g. GTVp = 1 and
    GTVn = 2, D26), binarised to ``label`` 1.
    """
    import SimpleITK as sitk
    from radiomics import featureextractor

    logging.getLogger("radiomics").setLevel(logging.ERROR)
    mask_image = sitk.Cast(sitk.ReadImage(mask) > 0, sitk.sitkUInt8)
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
    corr_threshold: float | None = typer.Option(
        None, help="Drop redundant features: |Spearman rho| above this, fit per fold on train"
    ),
    classifier: str | None = typer.Option(
        None,
        help="logreg (L2), enet (elastic net, L1 share chosen on val) or cox (ridge Cox); "
        "default: cox for survival tasks, logreg otherwise",
    ),
) -> None:
    """Linear probe on radiomics features, every fold (ridge Cox on survival tasks)."""
    kind = probe_kind(task)
    classifier = classifier or kind
    allowed = ("cox",) if kind == "cox" else ("logreg", "enet")
    if classifier not in allowed:
        raise typer.BadParameter(f"classifier for {task} must be one of {allowed}")
    cases, _, _ = task_labels(task)
    x, names = load_features(task, features, cases)
    missing = int(np.isnan(x).any(axis=1).sum())
    logger.info(f"{len(names)} features, {missing} of {len(cases)} cases with missing values")
    select = (
        None if corr_threshold is None else partial(correlation_filter, threshold=corr_threshold)
    )
    probe_folds(
        task,
        run_name(features, corr_threshold, classifier),
        x,
        select=select,
        l1_ratios=ENET_L1_RATIOS if classifier == "enet" else (0.0,),
    )


if __name__ == "__main__":
    app()

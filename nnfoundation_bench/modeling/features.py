"""Frozen nnFoundation features and linear probes (``uv run nnfb-features --help``).

``extract`` runs the frozen pretrained encoder once per prepared crop and pools its last
feature map in several ways, using the ROI mask exported by ``prepare``. ``probe`` fits
an L2 logistic regression per fold on one pooling and writes results in the same format
as fine-tuned runs, so ``nnfb-predict summarize`` lists both. ``fuse`` combines the
features of several single-modality tasks of the same cases (late fusion, D14, D30).
``combine`` blends an image probe with a covariate probe (e.g. age, centre; D36).

The encoder is built with glovita's ``nnFoundationEncoder`` (same weights and
architecture as fine-tuning). glovita's head sees the mean of this feature map
(``mean`` below); the other poolings are what we compare against it. The 2D baselines
(``dinov2``, ``dinov2_3p``, D38) run slice by slice on the QC crops instead
(``slice_features``) and are pooled and probed the same way.
"""

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

from nnfoundation_bench.config import CROP_SIZE, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR
from nnfoundation_bench.datasets import get_task
from nnfoundation_bench.modeling.predict import binary_metrics
from nnfoundation_bench.modeling.runs import processed_task, results_dir
from nnfoundation_bench.modeling.slice_features import SLICE_ENCODERS
from nnfoundation_bench.modeling.survival import (
    fit_cox_probe,
    probe_kind,
    survival_metrics,
    survival_table,
)
from nnfoundation_bench.weights import MODELS, get_checkpoint

app = typer.Typer(no_args_is_help=True, add_completion=False)

# Frozen encoders: nnFoundation (3D) and the slice-wise 2D baselines (D38).
FEATURE_MODELS = (*MODELS, *SLICE_ENCODERS)

# ROI dilation for ``roi_margin_*`` poolings, in mm (= voxels at 1 mm).
ROI_MARGIN_MM = 10.0
# Weighted means of real embeddings only: a per-channel max mixes coordinates from
# different locations into a vector the encoder never produces (see D13).
POOLINGS = ("mean", "roi_mean", "roi_margin_mean")
# Concatenation of the per-ROI means (tasks with ``roi_names``), built at probe time.
CONCAT_POOLING = "roi_concat_mean"
# Same poolings over L2-normalised tokens (every token weighs the same, only its direction
# counts), for the 2D slice encoders (D39): ``roi_mean_l2`` etc.
L2_SUFFIX = "_l2"
# Inverse L2 strengths tried on the validation split.
PROBE_CS = tuple(10.0**k for k in range(-4, 3))


def features_dir(task: str, model: str) -> Path:
    return PROCESSED_DATA_DIR / task / "features" / model


def probe_run_name(model: str, pooling: str, kind: str = "logreg") -> str:
    """``<model>_frozen_<pooling>_<logreg|cox>``."""
    return f"{model}_frozen_{pooling}_{kind}"


def task_poolings(task: str, model: str | None = None) -> tuple[str, ...]:
    """POOLINGS, plus ``<roi>_mean`` per named ROI and their concatenation if the task has
    named ROIs (e.g. GTVp and GTVn). For a 2D slice encoder (``model``), also every pooling
    but the concatenation over L2-normalised tokens (D39)."""
    names = [name for _, name in get_task(task).roi_names]
    base = (*POOLINGS, *(f"{n}_mean" for n in names))
    l2 = tuple(p + L2_SUFFIX for p in base) if model in SLICE_ENCODERS else ()
    return (*base, *((CONCAT_POOLING,) if names else ()), *l2)


def cell_weights(mask: np.ndarray, grid: tuple[int, ...]) -> np.ndarray:
    """Fraction of each feature-grid cell covered by ``mask`` (z, y, x) -> ``grid``."""
    factors = [s // g for s, g in zip(mask.shape, grid, strict=True)]
    if any(s % g for s, g in zip(mask.shape, grid, strict=True)):
        raise ValueError(f"Mask {mask.shape} does not tile the feature grid {grid}")
    blocks = mask.astype(np.float32).reshape(
        grid[0], factors[0], grid[1], factors[1], grid[2], factors[2]
    )
    return blocks.mean(axis=(1, 3, 5))


def dilate(mask: np.ndarray, margin: float) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    if not mask.any():
        return mask.astype(bool)
    return distance_transform_edt(~mask.astype(bool)) <= margin


def pool_features(
    feature_map, mask: np.ndarray, roi_names: tuple[tuple[int, str], ...] = (), l2: bool = False
):
    """Pool a (C, gz, gy, gx) map in every ``POOLINGS`` way; ``mask`` is (z, y, x) uint8.

    ROI poolings weight each cell by the fraction of ROI voxels it holds; ``roi`` is the
    union of all mask values. Without a ROI they fall back to the whole-crop mean. Each
    ``(value, name)`` of ``roi_names`` adds ``<name>_mean`` over ``mask == value``, NaN
    when that ROI is absent (imputed in the probe like any missing value).

    ``l2`` adds every pooling again over tokens scaled to unit L2 norm (``<pooling>_l2``,
    D39). ``feature_map`` is a numpy array or a torch tensor; a tensor is pooled on its
    device (the axial 2D maps are ~450 MB, D38), and the pooled vectors come back as numpy.
    """
    grid = tuple(feature_map.shape[1:])
    n_cells = int(np.prod(grid))
    # Pooling name -> cell weights summing to 1; None: ROI absent.
    weights = {"mean": np.full(n_cells, 1 / n_cells, np.float32)}
    for name, roi in (("roi", mask > 0), ("roi_margin", dilate(mask > 0, ROI_MARGIN_MM))):
        w = cell_weights(roi, grid).reshape(-1)
        weights[f"{name}_mean"] = w / w.sum() if w.sum() > 0 else weights["mean"]
    for value, name in roi_names:
        w = cell_weights(mask == value, grid).reshape(-1)
        weights[f"{name}_mean"] = w / w.sum() if w.sum() > 0 else None

    if isinstance(feature_map, np.ndarray):
        flat = feature_map.reshape(feature_map.shape[0], -1).astype(np.float32)
        maps = {"": flat}
        if l2:
            maps[L2_SUFFIX] = flat / np.linalg.norm(flat, axis=0, keepdims=True).clip(1e-12)

        def matvec(x, w):
            return x @ w
    else:
        import torch

        flat = feature_map.reshape(feature_map.shape[0], -1).float()
        maps = {"": flat}
        if l2:
            maps[L2_SUFFIX] = flat / flat.norm(dim=0, keepdim=True).clamp_min(1e-12)

        def matvec(x, w):
            return (x @ torch.from_numpy(w).to(x.device)).cpu().numpy()

    nan = np.full(len(flat), np.nan, np.float32)
    return {
        name + suffix: nan if w is None else matvec(x, w)
        for suffix, x in maps.items()
        for name, w in weights.items()
    }


def build_encoder(model: str, device: str):
    """Frozen pretrained encoder, loaded exactly as glovita does for fine-tuning."""
    import torch
    from glovita.models.img_encoder.nnfoundation import nnFoundationEncoder

    encoder = nnFoundationEncoder(
        get_checkpoint(model),
        pretrained=True,
        input_channels=1,
        drop_path_rate=None,
        input_shape=CROP_SIZE,
    )
    encoder.eval().requires_grad_(False)
    return encoder.to(torch.device(device))


def feature_map(encoder, x):
    """Last feature map (B, C, gz, gy, gx) before glovita's global average pooling."""
    import torch
    from dynamic_network_architectures.architectures.primus import Primus

    net = encoder.model
    if not isinstance(net, Primus):
        return net.encoder(x)  # ResEnc-L: (B, 320, 6, 6, 6) for 192^3
    # Same steps as glovita's primus_forward_features, minus the token mean.
    tokens = net.down_projection(x)
    grid = tokens.shape[2:]
    tokens = tokens.flatten(2).transpose(1, 2)
    n_reg = 0 if net.register_tokens is None else net.register_tokens.shape[1]
    if n_reg:
        tokens = torch.cat((net.register_tokens.expand(len(tokens), -1, -1), tokens), dim=1)
    tokens, _ = net.eva(tokens)
    tokens = tokens[:, n_reg:]
    return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, *grid)  # (B, 1056, 24, 24, 24)


def _missing_keys(path: Path, keys: list[str]) -> bool:
    """True if the features file is absent or lacks a pooling (e.g. added later, D39)."""
    if not path.exists():
        return True
    with np.load(path) as f:
        return not set(keys) <= set(f.files)


def _read_b2nd(path: Path) -> np.ndarray:
    import blosc2

    return blosc2.open(str(path), mode="r")[...]


def _slice_map(task: str, case: str, model, encoder, device: str, shape):
    """Slice-wise 2D feature map of one case (a tensor on ``device``), from its QC crop in
    HU / SUV / MR units (the z-scored arrays lose the absolute scale the CT and PET windows
    need)."""
    import SimpleITK as sitk

    from nnfoundation_bench.modeling.slice_features import prepare, slice_feature_map

    path = INTERIM_DATA_DIR / processed_task(task) / "images" / f"{case}.nii.gz"
    crop = sitk.GetArrayFromImage(sitk.ReadImage(path))
    if crop.shape != tuple(shape):
        raise ValueError(f"{case}: QC crop {crop.shape} is not on the mask grid {shape}")
    volume = prepare(crop, get_task(task).modality, encoder)
    return slice_feature_map(
        model, volume, encoder.planes, device, to_numpy=False, encoder=encoder
    )


@app.command()
def extract(
    task: str,
    model: str = typer.Option("cnn", help=f"Pretrained model: {', '.join(FEATURE_MODELS)}"),
    device: str = typer.Option("cuda:0", help="Torch device"),
    save_maps: bool = typer.Option(
        False, help="Also save the full feature map (float16; ViT ~29 MB per case)"
    ),
    overwrite: bool = typer.Option(False, help="Recompute existing features"),
) -> None:
    """Pool frozen-encoder features of every case of TASK (one forward pass each)."""
    import torch

    root = PROCESSED_DATA_DIR / task
    cases = sorted(json.loads((root / "labels.json").read_text()))
    out_dir = features_dir(task, model)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = [p for p in task_poolings(task, model) if p != CONCAT_POOLING]
    todo = [c for c in cases if overwrite or _missing_keys(out_dir / f"{c}.npz", keys)]
    logger.info(f"{task} / {model}: {len(todo)} of {len(cases)} cases to extract")
    if not todo:
        return

    roi_names = get_task(task).roi_names
    slice_encoder = SLICE_ENCODERS.get(model)
    if slice_encoder:
        from nnfoundation_bench.modeling import slice_features

        encoder = slice_features.build_slice_encoder(model, device)
        checkpoint = slice_features.checkpoint(model)
    else:
        encoder = build_encoder(model, device)
        checkpoint = MODELS[model].repo + "@" + MODELS[model].revision
    for case in tqdm(todo, desc=f"{task} {model}"):
        mask = _read_b2nd(root / "masks" / f"{case}.b2nd")[0]
        if slice_encoder:
            fmap = _slice_map(task, case, encoder, slice_encoder, device, mask.shape)
        else:
            image = torch.from_numpy(_read_b2nd(root / "images" / f"{case}.b2nd"))[None]
            with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                fmap = feature_map(encoder, image.to(device)).float()[0].cpu().numpy()
        arrays = pool_features(fmap, mask, roi_names, l2=bool(slice_encoder))
        if save_maps:
            arrays["map"] = np.asarray(fmap.cpu() if slice_encoder else fmap, np.float16)
        tmp = out_dir / f".partial.{case}.npz"
        np.savez(tmp, **arrays)
        tmp.rename(out_dir / f"{case}.npz")
    (out_dir / "info.json").write_text(
        json.dumps(
            {
                "model": model,
                "checkpoint": checkpoint,
                "feature_dim": int(fmap.shape[0]),
                "grid": list(fmap.shape[1:]),
                "poolings": keys,
                "roi_margin_mm": ROI_MARGIN_MM,
                "precision": "bf16 autocast",
            },
            indent=2,
        )
    )
    logger.success(f"Features in {out_dir} (grid {fmap.shape[1:]}, dim {fmap.shape[0]})")


def load_pooled(task: str, model: str, pooling: str, cases: list[str]) -> np.ndarray:
    """(cases x features) matrix of one pooling; ``roi_concat_mean`` stacks the per-ROI
    means side by side."""
    if pooling == CONCAT_POOLING:
        parts = [p for p in task_poolings(task) if p not in (*POOLINGS, CONCAT_POOLING)]
        return np.hstack([load_pooled(task, model, p, cases) for p in parts])
    folder = features_dir(task, model)
    return np.stack([np.load(folder / f"{c}.npz")[pooling] for c in cases])


def fit_probe(
    x: np.ndarray,
    y: np.ndarray,
    split: dict[str, np.ndarray],
    cs=PROBE_CS,
    l1_ratios=(0.0,),
) -> tuple[np.ndarray, dict, float]:
    """Logistic regression on train, hyperparameters picked by val AUROC.

    Returns test scores, the chosen ``{"C", "l1_ratio"}`` and the val AUROC. ``l1_ratio``
    0 is plain L2 (lbfgs); values in (0, 1] give an elastic net (saga), whose L1 part sets
    the weights of redundant features to zero. Missing values (NaN, e.g. a radiomics
    feature on an empty ROI) get the train median; features are then standardised with
    train statistics. The model is fitted on train only (val is used for selection), like
    glovita's best-checkpoint selection.
    """
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr, va, te = split["train"], split["val"], split["test"]
    best = (-np.inf, None, None)
    for l1_ratio in l1_ratios:
        for c in cs:
            clf = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
                LogisticRegression(
                    C=c,
                    l1_ratio=l1_ratio,
                    solver="lbfgs" if l1_ratio == 0 else "saga",
                    class_weight="balanced",
                    max_iter=5000,
                ),
            )
            with warnings.catch_warnings():
                # saga at the weakest penalties may stop before full convergence; the
                # val AUROC decides whether that fit is used.
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf.fit(x[tr], y[tr])
            val_scores = clf.predict_proba(x[va])[:, 1]
            auc = roc_auc_score(y[va], val_scores) if len(np.unique(y[va])) > 1 else np.nan
            if best[1] is None or auc > best[0]:
                best = (auc, {"C": c, "l1_ratio": l1_ratio}, clf)
    val_auc, params, clf = best
    return clf.predict_proba(x[te])[:, 1], params, float(val_auc)


def task_labels(task: str) -> tuple[list[str], np.ndarray, list[dict]]:
    """Sorted case IDs, their labels and the splits of a prepared (or fused) task."""
    root = PROCESSED_DATA_DIR / processed_task(task)
    labels = json.loads((root / "labels.json").read_text())
    splits = json.loads((root / "splits.json").read_text())
    cases = sorted(labels)
    return cases, np.array([labels[c] for c in cases]), splits


def probe_folds(
    task: str,
    run: str,
    x: np.ndarray,
    select=None,
    l1_ratios=(0.0,),
) -> list[float]:
    """Fit ``fit_probe`` (or ``fit_cox_probe`` on survival tasks) on every fold; ``x``
    rows follow ``task_labels`` order.

    ``select(x_train) -> column indices`` (optional) picks features on each fold's train
    cases only, e.g. ``correlation_filter``. Writes ``outputs/results/<task>/<run>/
    fold<k>/`` (metrics.json, predictions.csv) like fine-tuned runs and returns the test
    AUROC (C-index for survival tasks) per fold.
    """
    cases, y, splits = task_labels(task)
    survival = survival_table(task)
    if survival is not None:
        if l1_ratios != (0.0,):
            raise ValueError("Survival tasks only have the ridge Cox probe")
        time = survival.loc[cases, "time"].to_numpy(float)
        event = survival.loc[cases, "event"].to_numpy(int)
    index = {c: i for i, c in enumerate(cases)}
    scores_per_fold = []
    for fold, split in enumerate(splits):
        idx = {k: np.array([index[c] for c in v]) for k, v in split.items()}
        cols = np.arange(x.shape[1]) if select is None else select(x[idx["train"]])
        test = pd.DataFrame({"case_id": split["test"], "label": y[idx["test"]]})
        metrics = {"task": task, "run": run, "fold": fold, "num_features": len(cols)}
        if survival is None:
            scores, params, val = fit_probe(x[:, cols], y, idx, l1_ratios=l1_ratios)
            test["score"] = scores
            metrics |= {"probe_C": params["C"], "probe_l1_ratio": params["l1_ratio"]}
            metrics |= {"val_auroc": val}
            metrics |= binary_metrics(test["label"].to_numpy(), scores)
            key = "auroc"
        else:
            scores, params, val = fit_cox_probe(x[:, cols], time, event, idx)
            test = test.assign(time=time[idx["test"]], event=event[idx["test"]], score=scores)
            metrics |= {"probe_alpha": params["alpha"], "val_cindex": val}
            metrics |= survival_metrics(test["event"].to_numpy(), test["time"].to_numpy(), scores)
            key = "cindex"
        out_dir = results_dir(task, run, fold)
        out_dir.mkdir(parents=True, exist_ok=True)
        test.to_csv(out_dir / "predictions.csv", index=False)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        scores_per_fold.append(metrics[key])
    mean, std = np.nanmean(scores_per_fold), np.nanstd(scores_per_fold, ddof=1)
    logger.info(f"{run}: test {key} {mean:.4f} ± {std:.4f}")
    return scores_per_fold


def save_fold(task, run, fold, cases, labels, scores, info: dict) -> float:
    """Write one fold's predictions.csv and metrics.json; returns the test AUROC."""
    test = pd.DataFrame({"case_id": cases, "label": labels, "score": scores})
    out_dir = results_dir(task, run, fold)
    out_dir.mkdir(parents=True, exist_ok=True)
    test.to_csv(out_dir / "predictions.csv", index=False)
    metrics = {"task": task, "run": run, "fold": fold} | info
    metrics |= binary_metrics(test["label"].to_numpy(), test["score"].to_numpy())
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics["auroc"]


@app.command()
def probe(
    task: str,
    model: str = typer.Option("cnn", help=f"Pretrained model: {', '.join(FEATURE_MODELS)}"),
    pooling: Annotated[
        list[str] | None, typer.Option(help="Poolings to probe (default: all)")
    ] = None,
) -> None:
    """Linear probe on frozen features, every fold; writes outputs/results/TASK/<run>/.

    Logistic regression, or ridge Cox on survival tasks (runs ``..._cox``).
    """
    cases, _, _ = task_labels(task)
    kind = probe_kind(task)
    for pool in pooling or task_poolings(task, model):
        x = load_pooled(task, model, pool, cases)
        probe_folds(task, probe_run_name(model, pool, kind), x)


FUSIONS = ("concat", "mean", "stack")


def fused_run_name(model: str, method: str, pooling: str | None) -> str:
    """``vit_fused_concat_roi_mean_logreg``; radiomics: ``radiomics_original_fused_concat_logreg``."""
    if model == "radiomics":
        return f"radiomics_original_fused_{method}_logreg"
    return f"{model}_fused_{method}_{pooling}_logreg"


def check_same_cases(tasks: list[str]) -> None:
    """Fusion needs identical labels and splits in every source task (D14)."""
    files = [
        [(PROCESSED_DATA_DIR / t / name).read_text() for name in ("labels.json", "splits.json")]
        for t in tasks
    ]
    if any(f != files[0] for f in files[1:]):
        raise typer.Exit(
            f"labels.json / splits.json differ between {tasks}: fuse needs both equal"
        )


def stack_scores(val_scores: np.ndarray, y_val: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """Score-level fusion (M4): a logistic regression on the per-modality probe scores
    (logits, one column per modality), fitted on val and applied to test."""
    from scipy.special import logit
    from sklearn.linear_model import LogisticRegression

    def to_logit(s):
        return logit(np.clip(s, 1e-6, 1 - 1e-6))

    stacker = LogisticRegression(class_weight="balanced", max_iter=5000)
    stacker.fit(to_logit(val_scores), y_val)
    return stacker.predict_proba(to_logit(test_scores))[:, 1]


def stack_folds(task: str, run: str, xs: list[np.ndarray]) -> list[float]:
    """M4 on every fold: one probe per modality (``fit_probe``: train, C by val), then
    ``stack_scores`` on their val scores. Val picks each probe's C and fits the stacker."""
    cases, y, splits = task_labels(task)
    index = {c: i for i, c in enumerate(cases)}
    aucs = []
    for fold, split in enumerate(splits):
        idx = {k: np.array([index[c] for c in v]) for k, v in split.items()}
        n_val = len(idx["val"])
        both = {**idx, "test": np.concatenate([idx["val"], idx["test"]])}
        scores = np.column_stack([fit_probe(x, y, both)[0] for x in xs])
        test = stack_scores(scores[:n_val], y[idx["val"]], scores[n_val:])
        info = {"num_modalities": len(xs)}
        aucs.append(save_fold(task, run, fold, split["test"], y[idx["test"]], test, info))
    logger.info(f"{run}: test AUROC {np.nanmean(aucs):.4f} ± {np.nanstd(aucs, ddof=1):.4f}")
    return aucs


@app.command()
def fuse(
    tasks: Annotated[list[str], typer.Argument(help="Single-modality tasks of the same cases")],
    name: str = typer.Option(..., help="Name of the fused task in outputs/results/"),
    model: str = typer.Option(
        "vit", help=f"{', '.join(FEATURE_MODELS)} or radiomics (original set)"
    ),
    pooling: Annotated[
        list[str] | None, typer.Option(help="Poolings to fuse (default: all; not radiomics)")
    ] = None,
    method: Annotated[
        list[str] | None, typer.Option(help=f"Fusion methods (default: {', '.join(FUSIONS)})")
    ] = None,
) -> None:
    """Late fusion across modalities (D14, D30): M2 concatenation, M3 mean of the
    embeddings (one encoder, one space), M4 score stacking. Results go to
    ``outputs/results/NAME/`` with ``fusion.json`` naming the source tasks."""
    check_same_cases(tasks)
    (RESULTS_DIR / name).mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / name / "fusion.json").write_text(json.dumps({"tasks": tasks}, indent=2))
    cases, _, _ = task_labels(name)
    methods = method or list(FUSIONS)
    if model == "radiomics":
        from nnfoundation_bench.modeling.radiomics_baseline import load_features

        groups = {None: [load_features(t, "original", cases)[0] for t in tasks]}
        methods = [m for m in methods if m != "mean"]  # different features per modality
    else:
        groups = {p: [load_pooled(t, model, p, cases) for t in tasks] for p in pooling or POOLINGS}
    for pool, xs in groups.items():
        for m in methods:
            run = fused_run_name(model, m, pool)
            if m == "concat":
                probe_folds(name, run, np.hstack(xs))
            elif m == "mean":
                probe_folds(name, run, np.mean(xs, axis=0))
            elif m == "stack":
                stack_folds(name, run, xs)
            else:
                raise typer.Exit(f"Unknown fusion method {m!r}: {FUSIONS}")


# Image + covariates (D36): weight of the image score, tried on val.
BLEND_WEIGHTS = tuple(np.round(np.linspace(0, 1, 11), 2))


def plus_run_name(base: str, columns: list[str]) -> str:
    """``vit_frozen_roi_mean_logreg`` + [age] -> ``vit_frozen_roi_mean_plus_age_logreg``."""
    stem, kind = base.rsplit("_", 1)
    return f"{stem}_plus_{'_'.join(columns)}_{kind}"


def blend(val_scores: np.ndarray, test_scores: np.ndarray, val_metric) -> tuple[np.ndarray, float]:
    """Weighted sum of two scores (columns: image, covariates), each standardised with val
    statistics. The image weight ``w`` (0 = covariates alone, 1 = image alone) maximises
    ``val_metric`` over ``BLEND_WEIGHTS``; ties keep the smaller ``w``. Returns the test
    scores and ``w``."""
    mean, std = val_scores.mean(0), val_scores.std(0)
    std = np.where(std > 0, std, 1.0)
    zv, zt = (val_scores - mean) / std, (test_scores - mean) / std
    w = max(BLEND_WEIGHTS, key=lambda w: val_metric(w * zv[:, 0] + (1 - w) * zv[:, 1]))
    return w * zt[:, 0] + (1 - w) * zt[:, 1], float(w)


def image_features(task: str, model: str, pooling: str | None, cases: list[str]):
    """(base run name, features) of an image probe: frozen features or radiomics."""
    if model == "radiomics":
        from nnfoundation_bench.modeling.radiomics_baseline import load_features

        return "radiomics_original_logreg", load_features(task, "original", cases)[0]
    return probe_run_name(model, pooling), load_pooled(task, model, pooling, cases)


def task_covariates(task: str, cases: list[str], columns: list[str]) -> np.ndarray:
    from nnfoundation_bench.modeling.predict import covariate_matrix

    labels = pd.read_csv(
        PROCESSED_DATA_DIR / processed_task(task) / "labels.csv", dtype={"case_id": str}
    ).set_index("case_id")
    if missing := [c for c in columns if c not in labels]:
        raise typer.Exit(f"{task}: no column {missing} in labels.csv")
    return covariate_matrix(labels.loc[cases], columns)


def combine_folds(task: str, run: str, x_image: np.ndarray, x_cov: np.ndarray) -> list[float]:
    """Per fold: an image probe and a covariate probe (``fit_probe``: train, C by val), then
    ``blend`` of their logits with the image weight chosen by val AUROC (D36)."""
    from scipy.special import expit, logit
    from sklearn.metrics import roc_auc_score

    cases, y, splits = task_labels(task)
    index = {c: i for i, c in enumerate(cases)}
    aucs = []
    for fold, split in enumerate(splits):
        idx = {k: np.array([index[c] for c in v]) for k, v in split.items()}
        n_val = len(idx["val"])
        both = {**idx, "test": np.concatenate([idx["val"], idx["test"]])}
        scores = np.column_stack(
            [logit(np.clip(fit_probe(x, y, both)[0], 1e-6, 1 - 1e-6)) for x in (x_image, x_cov)]
        )
        y_val = y[idx["val"]]
        test, w = blend(
            scores[:n_val], scores[n_val:], lambda s, y_val=y_val: roc_auc_score(y_val, s)
        )
        info = {"blend_weight_image": w}
        aucs.append(save_fold(task, run, fold, split["test"], y[idx["test"]], expit(test), info))
    logger.info(f"{run}: test AUROC {np.nanmean(aucs):.4f} ± {np.nanstd(aucs, ddof=1):.4f}")
    return aucs


@app.command()
def combine(
    task: str,
    model: str = typer.Option(
        "vit", help=f"{', '.join(FEATURE_MODELS)} or radiomics (original set)"
    ),
    pooling: Annotated[
        list[str] | None, typer.Option(help="Poolings (default: all; not radiomics)")
    ] = None,
    column: Annotated[
        list[str] | None, typer.Option(help="labels.csv covariates (default: age)")
    ] = None,
) -> None:
    """Image probe + covariate probe blended on val (D36). Run
    ``<image run>_plus_<columns>_logreg``; compare with the image run and
    ``covariates_<columns>_logreg`` (``<!-- generated: adjusted TASK -->``)."""
    cases, _, _ = task_labels(task)
    columns = column or ["age"]
    x_cov = task_covariates(task, cases, columns)
    for pool in [None] if model == "radiomics" else pooling or POOLINGS:
        base, x_image = image_features(task, model, pool, cases)
        combine_folds(task, plus_run_name(base, columns), x_image, x_cov)


if __name__ == "__main__":
    app()

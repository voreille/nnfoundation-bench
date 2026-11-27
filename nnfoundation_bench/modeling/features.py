"""Frozen nnFoundation features and linear probes (``uv run nnfb-features --help``).

``extract`` runs the frozen pretrained encoder once per prepared crop and pools its last
feature map in several ways, using the ROI mask exported by ``prepare``. ``probe`` fits
an L2 logistic regression per fold on one pooling and writes results in the same format
as fine-tuned runs, so ``nnfb-predict summarize`` lists both. ``fuse`` combines the
features of several single-modality tasks of the same cases (late fusion, D14, D30).
``combine`` blends an image probe with a covariate probe (e.g. age, centre; D36).

The encoder is built with glovita's ``nnFoundationEncoder`` (same weights and
architecture as fine-tuning). glovita's head sees the mean of this feature map
(``mean`` below); the other poolings are what we compare against it.

Organ tasks (D38) store whole scans: ``extract`` cuts each organ's 192^3 crop from the scan
at extract time (``organ_crop``) and ``probe`` fits a multinomial logistic regression,
scored by balanced accuracy, accuracy and macro one-vs-rest AUROC.
"""

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

from nnfoundation_bench.config import CROP_SIZE, PROCESSED_DATA_DIR, RESULTS_DIR
from nnfoundation_bench.datasets import get_task
from nnfoundation_bench.modeling.predict import (
    balanced_accuracy,
    binary_metrics,
    multiclass_metrics,
)
from nnfoundation_bench.modeling.runs import (
    is_organ_task,
    processed_task,
    refuse_organ_tasks,
    results_dir,
)
from nnfoundation_bench.modeling.survival import (
    fit_cox_probe,
    probe_kind,
    survival_metrics,
    survival_table,
)
from nnfoundation_bench.weights import MODELS, get_checkpoint

app = typer.Typer(no_args_is_help=True, add_completion=False)

# ROI dilation for ``roi_margin_*`` poolings, in mm (= voxels at 1 mm).
ROI_MARGIN_MM = 10.0
# Weighted means of real embeddings only: a per-channel max mixes coordinates from
# different locations into a vector the encoder never produces (see D13).
POOLINGS = ("mean", "roi_mean", "roi_margin_mean")
# Concatenation of the per-ROI means (tasks with ``roi_names``), built at probe time.
CONCAT_POOLING = "roi_concat_mean"
# Inverse L2 strengths tried on the validation split.
PROBE_CS = tuple(10.0**k for k in range(-4, 3))


def features_dir(task: str, model: str) -> Path:
    return PROCESSED_DATA_DIR / task / "features" / model


def probe_run_name(model: str, pooling: str, kind: str = "logreg") -> str:
    """``<model>_frozen_<pooling>_<logreg|cox>``."""
    return f"{model}_frozen_{pooling}_{kind}"


def task_poolings(task: str) -> tuple[str, ...]:
    """POOLINGS, plus ``<roi>_mean`` per named ROI and their concatenation if the task has
    named ROIs (e.g. GTVp and GTVn)."""
    names = [name for _, name in get_task(task).roi_names]
    if not names:
        return POOLINGS
    return (*POOLINGS, *(f"{n}_mean" for n in names), CONCAT_POOLING)


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
    """Voxels within ``margin`` voxels (Euclidean) of ``mask``.

    The distance transform runs on the mask's bounding box widened by the margin only:
    every voxel within the margin lies inside it, so the result equals a full-volume
    transform (about 0.5 s per 192^3 crop) at a fraction of the cost.
    """
    from scipy.ndimage import distance_transform_edt

    mask = mask.astype(bool)
    if not mask.any():
        return mask
    pad = int(np.ceil(margin)) + 1
    box = []
    for axis in range(mask.ndim):
        idx = np.flatnonzero(mask.any(axis=tuple(a for a in range(mask.ndim) if a != axis)))
        box.append(slice(max(int(idx[0]) - pad, 0), min(int(idx[-1]) + 1 + pad, mask.shape[axis])))
    box = tuple(box)
    out = np.zeros_like(mask)
    out[box] = distance_transform_edt(~mask[box]) <= margin
    return out


def pool_features(
    feature_map: np.ndarray, mask: np.ndarray, roi_names: tuple[tuple[int, str], ...] = ()
) -> dict[str, np.ndarray]:
    """Pool a (C, gz, gy, gx) map in every ``POOLINGS`` way; ``mask`` is (z, y, x) uint8.

    ROI poolings weight each cell by the fraction of ROI voxels it holds; ``roi`` is the
    union of all mask values. Without a ROI they fall back to the whole-crop mean. Each
    ``(value, name)`` of ``roi_names`` adds ``<name>_mean`` over ``mask == value``, NaN
    when that ROI is absent (imputed in the probe like any missing value).
    """
    flat = feature_map.reshape(feature_map.shape[0], -1).astype(np.float32)
    pooled = {"mean": flat.mean(1)}
    grid = feature_map.shape[1:]
    for name, roi in (("roi", mask > 0), ("roi_margin", dilate(mask > 0, ROI_MARGIN_MM))):
        w = cell_weights(roi, grid).reshape(-1)
        pooled[f"{name}_mean"] = flat @ w / w.sum() if w.sum() > 0 else pooled["mean"]
    for value, name in roi_names:
        w = cell_weights(mask == value, grid).reshape(-1)
        pooled[f"{name}_mean"] = (
            flat @ w / w.sum() if w.sum() > 0 else np.full(len(flat), np.nan, np.float32)
        )
    return pooled


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


def _read_b2nd(path: Path) -> np.ndarray:
    import blosc2

    return blosc2.open(str(path), mode="r")[...]


class LazyVolume:
    """(z, y, x) view of a channel-first b2nd file; slicing reads only the chunks needed."""

    def __init__(self, path: Path):
        import blosc2

        self.array = blosc2.open(str(path), mode="r")
        self.shape = tuple(self.array.shape[1:])

    def __getitem__(self, key: tuple[slice, ...]) -> np.ndarray:
        return self.array[(0, *key)]


def organ_bbox(mask: np.ndarray, organ: int) -> tuple[slice, slice, slice]:
    """Bounding box (slices) of ``mask == organ``; ValueError if the organ is absent."""
    inside = mask == organ
    if not inside.any():
        raise ValueError(f"Label {organ} is not in the mask")
    box = []
    for axis in range(3):
        idx = np.flatnonzero(inside.any(axis=tuple(a for a in range(3) if a != axis)))
        box.append(slice(int(idx[0]), int(idx[-1]) + 1))
    return tuple(box)


def organ_crop(
    image,
    mask: np.ndarray,
    organ: int,
    size: int = CROP_SIZE[0],
    pad: float = 0.0,
    bbox: tuple[slice, slice, slice] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """``size``^3 crop of ``image`` and of the binary mask ``mask == organ``, centred on the
    organ's bounding-box centre (as ``roi_center``); ``pad`` outside the scan.

    ``image`` is a (z, y, x) array or anything sliceable like one (``LazyVolume``).
    ``bbox`` (from ``organ_bbox`` or ``scipy.ndimage.find_objects``) skips the search.
    Returns (image crop float32, organ mask crop uint8, fraction of the organ inside).
    """
    bbox = organ_bbox(mask, organ) if bbox is None else bbox
    total = int((mask[bbox] == organ).sum())
    starts = [int(np.floor((b.start + b.stop - 1) / 2 - (size - 1) / 2 + 0.5)) for b in bbox]
    src = tuple(
        slice(max(a, 0), min(a + size, n)) for a, n in zip(starts, mask.shape, strict=True)
    )
    dst = tuple(slice(s.start - a, s.stop - a) for s, a in zip(src, starts, strict=True))
    crop = np.full((size,) * 3, pad, np.float32)
    organ_mask = np.zeros((size,) * 3, np.uint8)
    crop[dst] = image[src]
    organ_mask[dst] = mask[src] == organ
    return crop, organ_mask, float(organ_mask.sum() / total)


def _crop_cases(root: Path, cases: list[str]):
    """(case, image (1, z, y, x), mask (z, y, x), {}) of the prepared 192^3 crops."""
    for case in cases:
        image = _read_b2nd(root / "images" / f"{case}.b2nd")
        yield case, image, _read_b2nd(root / "masks" / f"{case}.b2nd")[0], {}


def _organ_cases(root: Path, cases: list[str]):
    """(case, organ crop (1, 192^3), organ mask, {roi_inside}) of an organ task: one scan
    at a time, its image read lazily, crops padded with the scan's z-scored pad value."""
    from scipy.ndimage import find_objects

    labels = pd.read_csv(root / "labels.csv", dtype={"case_id": str, "scan_id": str})
    labels = labels.set_index("case_id").loc[cases]
    organs = pd.read_csv(root / "organs.csv", dtype={"scan_id": str})
    label_ids = organs.set_index(["scan_id", "organ"])["label_id"]
    pad_z = pd.read_csv(root / "scans.csv", dtype={"scan_id": str}).set_index("scan_id")["pad_z"]
    for scan, samples in labels.groupby("scan_id", sort=True):
        image = LazyVolume(root / "images" / f"{scan}.b2nd")
        mask = _read_b2nd(root / "masks" / f"{scan}.b2nd")[0]
        boxes = find_objects(mask)
        for case, organ in samples["organ"].items():
            label_id = int(label_ids[(scan, organ)])
            crop, organ_mask, inside = organ_crop(
                image, mask, label_id, pad=float(pad_z[scan]), bbox=boxes[label_id - 1]
            )
            yield case, crop[None], organ_mask, {"roi_inside": np.float32(inside)}


@app.command()
def extract(
    task: str,
    model: str = typer.Option("cnn", help=f"Pretrained model: {', '.join(MODELS)}"),
    device: str = typer.Option("cuda:0", help="Torch device"),
    save_maps: bool = typer.Option(
        False, help="Also save the full feature map (float16; ViT ~29 MB per case)"
    ),
    overwrite: bool = typer.Option(False, help="Recompute existing features"),
) -> None:
    """Pool frozen-encoder features of every case of TASK (one forward pass each).

    Organ tasks: one forward pass per (scan, organ) sample on its organ-centred crop, plus
    ``roi_inside`` (fraction of the organ inside the crop).
    """
    import torch

    root = PROCESSED_DATA_DIR / task
    cases = sorted(json.loads((root / "labels.json").read_text()))
    out_dir = features_dir(task, model)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cases if overwrite or not (out_dir / f"{c}.npz").exists()]
    logger.info(f"{task} / {model}: {len(todo)} of {len(cases)} cases to extract")
    if not todo:
        return

    roi_names = get_task(task).roi_names
    cases_of = _organ_cases if is_organ_task(task) else _crop_cases
    encoder = build_encoder(model, device)
    for case, array, mask, extra in tqdm(cases_of(root, todo), total=len(todo), desc=task):
        image = torch.from_numpy(array)[None]
        with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            fmap = feature_map(encoder, image.to(device)).float()[0].cpu().numpy()
        arrays = pool_features(fmap, mask, roi_names) | extra
        if save_maps:
            arrays["map"] = fmap.astype(np.float16)
        tmp = out_dir / f".partial.{case}.npz"
        np.savez(tmp, **arrays)
        tmp.rename(out_dir / f"{case}.npz")
    (out_dir / "info.json").write_text(
        json.dumps(
            {
                "model": model,
                "checkpoint": MODELS[model].repo + "@" + MODELS[model].revision,
                "feature_dim": int(fmap.shape[0]),
                "grid": list(fmap.shape[1:]),
                "poolings": [p for p in task_poolings(task) if p != CONCAT_POOLING],
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
    multiclass: bool = False,
):
    """Logistic regression on train, hyperparameters picked by val AUROC.

    Returns test scores, the chosen ``{"C", "l1_ratio"}`` and the val AUROC. With
    ``multiclass`` (organ tasks, D38) it is a multinomial regression picked by val balanced
    accuracy, and returns ``(proba, C, val balanced accuracy)``: ``proba`` is (n_test, K)
    with one column per class of ``y`` (``np.unique(y)``; 0 for a class missing from
    train). ``l1_ratio``
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
            if multiclass:
                score = balanced_accuracy(y[va], clf.predict(x[va]))
            else:
                val_scores = clf.predict_proba(x[va])[:, 1]
                score = roc_auc_score(y[va], val_scores) if len(np.unique(y[va])) > 1 else np.nan
            if best[1] is None or score > best[0]:
                best = (score, {"C": c, "l1_ratio": l1_ratio}, clf)
    val_score, params, clf = best
    if not multiclass:
        return clf.predict_proba(x[te])[:, 1], params, float(val_score)
    classes = np.unique(y)
    proba = np.zeros((len(te), len(classes)))
    proba[:, np.searchsorted(classes, clf.classes_)] = clf.predict_proba(x[te])
    return proba, params["C"], float(val_score)


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
    if is_organ_task(task):
        return organ_probe_folds(task, run, x, cases, y, splits, select, l1_ratios)
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


def organ_probe_folds(task, run, x, cases, y, splits, select=None, l1_ratios=(0.0,)) -> list:
    """``probe_folds`` of an organ task: multinomial probe per fold (D38). Writes
    metrics.json (balanced accuracy, accuracy, macro AUROC), predictions.csv
    (``case_id,scan_id,organ,label,pred,p_pred``) and probs.npz (``case_id``, ``classes``,
    ``probs``); returns the test balanced accuracy per fold."""
    samples = pd.read_csv(
        PROCESSED_DATA_DIR / processed_task(task) / "labels.csv",
        dtype={"case_id": str, "scan_id": str},
    ).set_index("case_id")
    classes = np.unique(y)
    index = {c: i for i, c in enumerate(cases)}
    scores_per_fold = []
    for fold, split in enumerate(splits):
        if not split["val"]:
            raise typer.Exit(f"{task} fold {fold}: empty val split, C cannot be chosen")
        idx = {k: np.array([index[c] for c in v], dtype=int) for k, v in split.items()}
        cols = np.arange(x.shape[1]) if select is None else select(x[idx["train"]])
        proba, c, val = fit_probe(x[:, cols], y, idx, l1_ratios=l1_ratios, multiclass=True)
        proba = proba.astype(np.float32)
        test = samples.loc[split["test"], ["scan_id", "organ"]].reset_index()
        test["label"] = y[idx["test"]]
        test["pred"] = classes[proba.argmax(1)]
        test["p_pred"] = proba.max(1)
        metrics = {"task": task, "run": run, "fold": fold, "num_features": len(cols)}
        metrics |= {"probe_C": c, "val_balanced_accuracy": val}
        metrics |= multiclass_metrics(test["label"].to_numpy(), proba, classes)
        out_dir = results_dir(task, run, fold)
        out_dir.mkdir(parents=True, exist_ok=True)
        test.to_csv(out_dir / "predictions.csv", index=False)
        np.savez(
            out_dir / "probs.npz", case_id=np.array(split["test"]), classes=classes, probs=proba
        )
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        scores_per_fold.append(metrics["balanced_accuracy"])
    mean, std = np.nanmean(scores_per_fold), np.nanstd(scores_per_fold, ddof=1)
    logger.info(f"{run}: test balanced accuracy {mean:.4f} ± {std:.4f}")
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
    model: str = typer.Option("cnn", help=f"Pretrained model: {', '.join(MODELS)}"),
    pooling: Annotated[
        list[str] | None, typer.Option(help="Poolings to probe (default: all)")
    ] = None,
) -> None:
    """Linear probe on frozen features, every fold; writes outputs/results/TASK/<run>/.

    Logistic regression, or ridge Cox on survival tasks (runs ``..._cox``).
    """
    cases, _, _ = task_labels(task)
    kind = probe_kind(task)
    for pool in pooling or task_poolings(task):
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
    model: str = typer.Option("vit", help=f"{', '.join(MODELS)} or radiomics (original set)"),
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
    refuse_organ_tasks(*tasks, what="late fusion")
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
    model: str = typer.Option("vit", help=f"{', '.join(MODELS)} or radiomics (original set)"),
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
    refuse_organ_tasks(task, what="image + covariate blends")
    cases, _, _ = task_labels(task)
    columns = column or ["age"]
    x_cov = task_covariates(task, cases, columns)
    for pool in [None] if model == "radiomics" else pooling or POOLINGS:
        base, x_image = image_features(task, model, pool, cases)
        combine_folds(task, plus_run_name(base, columns), x_image, x_cov)


if __name__ == "__main__":
    app()

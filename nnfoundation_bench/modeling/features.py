"""Frozen nnFoundation features and linear probes (``uv run nnfb-features --help``).

``extract`` runs the frozen pretrained encoder once per prepared crop and pools its last
feature map in several ways, using the ROI mask exported by ``prepare``. ``probe`` fits
an L2 logistic regression per fold on one pooling and writes results in the same format
as fine-tuned runs, so ``nnfb-predict summarize`` lists both.

The encoder is built with glovita's ``nnFoundationEncoder`` (same weights and
architecture as fine-tuning). glovita's head sees the mean of this feature map
(``mean`` below); the other poolings are what we compare against it.
"""

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

from nnfoundation_bench.config import CROP_SIZE, PROCESSED_DATA_DIR
from nnfoundation_bench.modeling.predict import binary_metrics
from nnfoundation_bench.modeling.runs import results_dir
from nnfoundation_bench.weights import MODELS, get_checkpoint

app = typer.Typer(no_args_is_help=True, add_completion=False)

# ROI dilation for ``roi_margin_*`` poolings, in mm (= voxels at 1 mm).
ROI_MARGIN_MM = 10.0
# Weighted means of real embeddings only: a per-channel max mixes coordinates from
# different locations into a vector the encoder never produces (see D13).
POOLINGS = ("mean", "roi_mean", "roi_margin_mean")
# Inverse L2 strengths tried on the validation split.
PROBE_CS = tuple(10.0**k for k in range(-4, 3))


def features_dir(task: str, model: str) -> Path:
    return PROCESSED_DATA_DIR / task / "features" / model


def probe_run_name(model: str, pooling: str) -> str:
    return f"{model}_frozen_{pooling}_logreg"


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


def pool_features(feature_map: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Pool a (C, gz, gy, gx) map in every ``POOLINGS`` way; ``mask`` is (z, y, x) uint8.

    ROI poolings weight each cell by the fraction of ROI voxels it holds. Without a ROI
    they fall back to the whole-crop mean.
    """
    flat = feature_map.reshape(feature_map.shape[0], -1).astype(np.float32)
    pooled = {"mean": flat.mean(1)}
    grid = feature_map.shape[1:]
    for name, roi in (("roi", mask > 0), ("roi_margin", dilate(mask > 0, ROI_MARGIN_MM))):
        w = cell_weights(roi, grid).reshape(-1)
        pooled[f"{name}_mean"] = flat @ w / w.sum() if w.sum() > 0 else pooled["mean"]
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
    """Pool frozen-encoder features of every case of TASK (one forward pass each)."""
    import torch

    root = PROCESSED_DATA_DIR / task
    cases = sorted(json.loads((root / "labels.json").read_text()))
    out_dir = features_dir(task, model)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cases if overwrite or not (out_dir / f"{c}.npz").exists()]
    logger.info(f"{task} / {model}: {len(todo)} of {len(cases)} cases to extract")
    if not todo:
        return

    encoder = build_encoder(model, device)
    for case in tqdm(todo, desc=f"{task} {model}"):
        image = torch.from_numpy(_read_b2nd(root / "images" / f"{case}.b2nd"))[None]
        mask = _read_b2nd(root / "masks" / f"{case}.b2nd")[0]
        with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            fmap = feature_map(encoder, image.to(device)).float()[0].cpu().numpy()
        arrays = pool_features(fmap, mask)
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
                "poolings": list(POOLINGS),
                "roi_margin_mm": ROI_MARGIN_MM,
                "precision": "bf16 autocast",
            },
            indent=2,
        )
    )
    logger.success(f"Features in {out_dir} (grid {fmap.shape[1:]}, dim {fmap.shape[0]})")


def load_pooled(task: str, model: str, pooling: str, cases: list[str]) -> np.ndarray:
    folder = features_dir(task, model)
    return np.stack([np.load(folder / f"{c}.npz")[pooling] for c in cases])


def fit_probe(
    x: np.ndarray, y: np.ndarray, split: dict[str, np.ndarray], cs=PROBE_CS
) -> tuple[np.ndarray, float, float]:
    """Logistic regression on train, C picked by val AUROC. Returns test scores, C, val AUROC.

    Features are standardised with train statistics. The model is fitted on train only
    (val is used for selection), like glovita's best-checkpoint selection.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr, va, te = split["train"], split["val"], split["test"]
    best = (-np.inf, None, None)
    for c in cs:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", max_iter=5000),
        )
        clf.fit(x[tr], y[tr])
        val_scores = clf.predict_proba(x[va])[:, 1]
        auc = roc_auc_score(y[va], val_scores) if len(np.unique(y[va])) > 1 else np.nan
        if best[1] is None or auc > best[0]:
            best = (auc, c, clf)
    val_auc, c, clf = best
    return clf.predict_proba(x[te])[:, 1], c, float(val_auc)


@app.command()
def probe(
    task: str,
    model: str = typer.Option("cnn", help=f"Pretrained model: {', '.join(MODELS)}"),
    pooling: Annotated[
        list[str] | None, typer.Option(help="Poolings to probe (default: all)")
    ] = None,
) -> None:
    """Linear probe on frozen features, every fold; writes reports/results/TASK/<run>/."""
    root = PROCESSED_DATA_DIR / task
    labels = json.loads((root / "labels.json").read_text())
    splits = json.loads((root / "splits.json").read_text())
    cases = sorted(labels)
    index = {c: i for i, c in enumerate(cases)}
    y = np.array([labels[c] for c in cases])

    for pool in pooling or POOLINGS:
        x = load_pooled(task, model, pool, cases)
        run = probe_run_name(model, pool)
        aucs = []
        for fold, split in enumerate(splits):
            idx = {k: np.array([index[c] for c in v]) for k, v in split.items()}
            scores, c, val_auc = fit_probe(x, y, idx)
            test = pd.DataFrame(
                {"case_id": split["test"], "label": y[idx["test"]], "score": scores}
            )
            out_dir = results_dir(task, run, fold)
            out_dir.mkdir(parents=True, exist_ok=True)
            test.to_csv(out_dir / "predictions.csv", index=False)
            metrics = {"task": task, "run": run, "fold": fold, "probe_C": c, "val_auroc": val_auc}
            metrics |= binary_metrics(test["label"].to_numpy(), test["score"].to_numpy())
            (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            aucs.append(metrics["auroc"])
        logger.info(f"{run}: test AUROC {np.nanmean(aucs):.4f} ± {np.nanstd(aucs, ddof=1):.4f}")


if __name__ == "__main__":
    app()

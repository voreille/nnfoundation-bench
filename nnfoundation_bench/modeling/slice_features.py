"""2D encoders run slice by slice on the 3D crops: DINOv2 (natural images, baseline D38)
and Curia / Curia-2 (CT and MR, D40).

Each slice of the 192^3 crop (1 mm) is resized to the encoder's ``size`` and normalised the
way the encoder was trained (``SliceEncoder.inputs``):

- ``imagenet``: mapped to [0, 1] with a fixed window (CT, PET) or per-volume percentiles
  (MR), copied to the three RGB channels and ImageNet-normalised (DINOv2).
- ``zscore``: raw HU / SUV / MR values, bicubic resize, z-scored per slice, one channel
  (Curia's ``CuriaImageProcessor``).

The encoder's last-layer patch tokens (CLS and register tokens dropped) are stacked along
the slice axis into a (C, gz, gy, gx) feature map, so ``features.pool_features`` and every
probe apply unchanged.

- ``<model>``: axial slices only; the map keeps every slice, e.g. (1024, 192, 24, 24).
- ``<model>_3p``: axial, coronal and sagittal slices. Each plane's map is averaged over
  blocks of slices to a g^3 grid and the three are averaged per cell, e.g. (1024, 24, 24,
  24) for DINOv2, the grid of the nnFoundation ViT.

Heavy imports (torch, transformers) stay inside the functions.
"""

from dataclasses import dataclass

import numpy as np

from nnfoundation_bench.weights import HUB_MODELS, get_hub_model

# Slice side after resizing: 24 patches of 14 px (DINOv2), one token per 8 mm.
SLICE_SIZE = 336
# Curia was trained on whole slices (typically 512 px at ~0.7 mm); 256 px over the 192 mm
# crop keeps that pixel size (0.75 mm) instead of zooming in 2.7x. ASSUMPTION, D40.
CURIA_SLICE_SIZE = 256
# Fixed intensity windows mapped to [0, 1]. CT: soft tissue (W400 / L40 HU). PET: SUV 0-20
# (tumour SUVmax in HECKTOR 2022 is 10-19). ASSUMPTION, D38.
WINDOWS = {"CT": (-160.0, 240.0), "PT": (0.0, 20.0)}
# MR has no absolute scale: per-volume percentiles instead.
MR_PERCENTILES = (0.5, 99.5)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Volume axis (z, y, x) that each plane slices along.
PLANE_AXIS = {"axial": 0, "coronal": 1, "sagittal": 2}


@dataclass(frozen=True)
class SliceEncoder:
    weights: str  # key of weights.HUB_MODELS
    planes: tuple[str, ...]
    size: int = SLICE_SIZE
    inputs: str = "imagenet"  # or "zscore" (see module doc)
    # Coronal / sagittal slices with superior at the top (Curia: IL / IP; the crops' z
    # axis runs inferior to superior). Axial slices are PL as they come (LPS crops).
    superior_up: bool = False
    # Lower clip after resizing, before the z-score (Curia-2: -1000 HU, ``clip_below_air``).
    clip_min: float | None = None


THREE_PLANES = ("axial", "coronal", "sagittal")
_CURIA = {"size": CURIA_SLICE_SIZE, "inputs": "zscore", "superior_up": True}
SLICE_ENCODERS = {
    "dinov2": SliceEncoder("dinov2", ("axial",)),
    "dinov2_3p": SliceEncoder("dinov2", THREE_PLANES),
    "curia": SliceEncoder("curia", ("axial",), **_CURIA),
    "curia_3p": SliceEncoder("curia", THREE_PLANES, **_CURIA),
    "curia2": SliceEncoder("curia2", ("axial",), **_CURIA, clip_min=-1000.0),
    "curia2_3p": SliceEncoder("curia2", THREE_PLANES, **_CURIA, clip_min=-1000.0),
}


def checkpoint(name: str) -> str:
    model = HUB_MODELS[SLICE_ENCODERS[name].weights]
    return f"{model.repo}@{model.revision}"


def prepare(volume: np.ndarray, modality: str, encoder: SliceEncoder) -> np.ndarray:
    """QC crop (HU, SUV, MR) to the encoder's input range: [0, 1] for ``imagenet``, raw
    float32 for ``zscore`` (normalised per slice after resizing)."""
    if encoder.inputs == "imagenet":
        return to_unit(volume, modality)
    return volume.astype(np.float32)


def to_unit(volume: np.ndarray, modality: str) -> np.ndarray:
    """Intensities (HU, SUV, MR) to [0, 1]: fixed window for CT/PET, percentiles for MR."""
    if modality in WINDOWS:
        lo, hi = WINDOWS[modality]
    else:
        lo, hi = np.percentile(volume, MR_PERCENTILES)
    return np.clip((volume.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def build_slice_encoder(name: str, device: str):
    import torch
    from transformers import AutoModel

    encoder = SLICE_ENCODERS[name]
    model = AutoModel.from_pretrained(get_hub_model(encoder.weights))
    if encoder.size % model.config.patch_size:
        raise ValueError(f"{name}: slice size {encoder.size} is not a multiple of the patch")
    model.eval().requires_grad_(False)
    return model.to(torch.device(device))


def _normalise(x, inputs: str):
    """(B, 1, H, W) resized slices to model inputs (see module doc)."""
    import torch

    if inputs == "imagenet":
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        return (x.expand(-1, 3, -1, -1) - mean) / std
    # Per-slice z-score; a constant slice is only centred (as CuriaImageProcessor).
    mean = x.mean((1, 2, 3), keepdim=True)
    std = x.std((1, 2, 3), keepdim=True)
    return (x - mean) / torch.where(std < 1e-6, 1.0, std)


def plane_tokens(
    model,
    volume,
    axis: int,
    batch_size: int,
    size: int = SLICE_SIZE,
    inputs: str = "imagenet",
    superior_up: bool = False,
    clip_min: float | None = None,
):
    """Patch tokens of every slice along ``axis``: (n_slices, g, g, C) on the volume's
    device, in (z, y, x) order. ``volume`` is a (z, y, x) tensor from ``prepare``."""
    import torch
    import torch.nn.functional as F

    slices = volume.movedim(axis, 0)[:, None]  # (n, 1, a, b), a/b in (z, y, x) order
    flip = superior_up and axis != 0  # rows are z for coronal and sagittal slices
    if flip:
        slices = slices.flip(2)
    n_skip = 1 + getattr(model.config, "num_register_tokens", 0)  # CLS + registers
    grid = size // model.config.patch_size
    resize = (
        {"mode": "bilinear"} if inputs == "imagenet" else {"mode": "bicubic", "antialias": True}
    )
    out = []
    for start in range(0, len(slices), batch_size):
        x = F.interpolate(slices[start : start + batch_size], size=(size, size), **resize)
        if clip_min is not None:
            x = x.clamp_min(clip_min)
        x = _normalise(x, inputs)
        tokens = model(pixel_values=x).last_hidden_state[:, n_skip:]
        out.append(tokens.reshape(len(x), grid, grid, -1).float())
    tokens = torch.cat(out)
    return tokens.flip(1) if flip else tokens


def slice_feature_map(
    model,
    volume: np.ndarray,
    planes: tuple[str, ...],
    device: str,
    batch_size: int = 32,
    to_numpy: bool = True,
    encoder: SliceEncoder | None = None,
):
    """(C, gz, gy, gx) feature map of a (z, y, x) volume from ``prepare`` (see module doc):
    a numpy array, or with ``to_numpy=False`` a float32 tensor left on ``device``.
    ``encoder`` gives the input settings (default: DINOv2's)."""
    import torch

    enc = encoder or SliceEncoder("", planes)
    opts = {
        "size": enc.size,
        "inputs": enc.inputs,
        "superior_up": enc.superior_up,
        "clip_min": enc.clip_min,
    }
    vol = torch.from_numpy(volume).to(device)
    with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        if planes == ("axial",):
            fmap = plane_tokens(model, vol, 0, batch_size, **opts).permute(3, 0, 1, 2)
            return fmap.cpu().numpy() if to_numpy else fmap.contiguous()  # (C, z, gy, gx)
        maps = []
        for plane in planes:
            axis = PLANE_AXIS[plane]
            tokens = plane_tokens(model, vol, axis, batch_size, **opts)  # (n, ga, gb, C)
            g = tokens.shape[1]
            # Average blocks of slices so the slice axis matches the in-plane grid.
            tokens = tokens.reshape(g, len(tokens) // g, g, g, -1).mean(1)
            # Back to (z, y, x, C): the slice axis was moved to the front.
            maps.append(tokens.movedim(0, axis))
        fmap = torch.stack(maps).mean(0).permute(3, 0, 1, 2)
        return fmap.cpu().numpy() if to_numpy else fmap.contiguous()

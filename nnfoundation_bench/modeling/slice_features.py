"""2D natural-image encoders run slice by slice on the 3D crops (baseline, D38).

Each slice of the 192^3 crop (1 mm) is mapped to [0, 1] with a fixed window (CT, PET) or
per-volume percentiles (MR), resized to ``SLICE_SIZE`` pixels, copied to the three RGB
channels and ImageNet-normalised. The encoder's last-layer patch tokens (CLS and
register tokens dropped) are stacked along the slice axis into a (C, gz, gy, gx) feature
map, so ``features.pool_features`` and every probe apply unchanged.

- ``dinov2``: axial slices only; the map keeps every slice: (1024, 192, 24, 24).
- ``dinov2_3p``: axial, coronal and sagittal slices. Each plane's map is averaged over
  blocks of slices to a 24^3 grid and the three are averaged per cell: (1024, 24, 24, 24),
  the grid of the nnFoundation ViT.

Heavy imports (torch, transformers) stay inside the functions.
"""

from dataclasses import dataclass

import numpy as np

from nnfoundation_bench.weights import HUB_MODELS, get_hub_model

# Slice side after resizing: 24 patches of 14 px (DINOv2), one token per 8 mm.
SLICE_SIZE = 336
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


SLICE_ENCODERS = {
    "dinov2": SliceEncoder("dinov2", ("axial",)),
    "dinov2_3p": SliceEncoder("dinov2", ("axial", "coronal", "sagittal")),
}


def checkpoint(name: str) -> str:
    model = HUB_MODELS[SLICE_ENCODERS[name].weights]
    return f"{model.repo}@{model.revision}"


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

    model = AutoModel.from_pretrained(get_hub_model(SLICE_ENCODERS[name].weights))
    model.eval().requires_grad_(False)
    return model.to(torch.device(device))


def plane_tokens(model, volume, axis: int, batch_size: int):
    """Patch tokens of every slice along ``axis``: (n_slices, g, g, C) on the volume's
    device. ``volume`` is a (z, y, x) tensor in [0, 1]."""
    import torch
    import torch.nn.functional as F

    slices = volume.movedim(axis, 0)[:, None]  # (n, 1, a, b), a/b in (z, y, x) order
    mean = torch.tensor(IMAGENET_MEAN, device=volume.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=volume.device).view(1, 3, 1, 1)
    n_skip = 1 + model.config.num_register_tokens  # CLS + registers
    grid = SLICE_SIZE // model.config.patch_size
    out = []
    for start in range(0, len(slices), batch_size):
        x = F.interpolate(
            slices[start : start + batch_size], size=(SLICE_SIZE, SLICE_SIZE), mode="bilinear"
        )
        x = (x.expand(-1, 3, -1, -1) - mean) / std
        tokens = model(pixel_values=x).last_hidden_state[:, n_skip:]
        out.append(tokens.reshape(len(x), grid, grid, -1).float())
    return torch.cat(out)


def slice_feature_map(
    model,
    volume: np.ndarray,
    planes: tuple[str, ...],
    device: str,
    batch_size: int = 32,
    to_numpy: bool = True,
):
    """(C, gz, gy, gx) feature map of a (z, y, x) volume in [0, 1] (see module doc): a
    numpy array, or with ``to_numpy=False`` a float32 tensor left on ``device``."""
    import torch

    vol = torch.from_numpy(volume).to(device)
    with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
        if planes == ("axial",):
            fmap = plane_tokens(model, vol, 0, batch_size).permute(3, 0, 1, 2)  # (C, z, gy, gx)
            return fmap.cpu().numpy() if to_numpy else fmap.contiguous()
        maps = []
        for plane in planes:
            axis = PLANE_AXIS[plane]
            tokens = plane_tokens(model, vol, axis, batch_size)  # (n, ga, gb, C)
            g = tokens.shape[1]
            # Average blocks of slices so the slice axis matches the in-plane grid.
            tokens = tokens.reshape(g, len(tokens) // g, g, g, -1).mean(1)
            # Back to (z, y, x, C): the slice axis was moved to the front.
            maps.append(tokens.movedim(0, axis))
        fmap = torch.stack(maps).mean(0).permute(3, 0, 1, 2)
        return fmap.cpu().numpy() if to_numpy else fmap.contiguous()

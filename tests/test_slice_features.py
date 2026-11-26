from types import SimpleNamespace

import numpy as np
import pytest

from nnfoundation_bench.modeling.slice_features import (
    SLICE_ENCODERS,
    SLICE_SIZE,
    plane_tokens,
    slice_feature_map,
    to_unit,
)


class PatchMeanModel:
    """Stand-in for a 2D ViT: one CLS token, then each patch's mean intensity as a
    1-channel token (row-major), so feature maps can be checked against the input."""

    def __init__(self, patch_size=14):
        # No num_register_tokens: Curia's plain Dinov2Config has none.
        self.config = SimpleNamespace(patch_size=patch_size)
        self.inputs = []

    def __call__(self, pixel_values):
        import torch
        import torch.nn.functional as F

        self.inputs.append(pixel_values)
        patches = F.avg_pool2d(pixel_values[:, :1], self.config.patch_size)
        tokens = patches.flatten(2).transpose(1, 2)  # (B, g*g, 1)
        cls = torch.zeros(len(tokens), 1, 1)
        return SimpleNamespace(last_hidden_state=torch.cat([cls, tokens], dim=1))


@pytest.mark.parametrize(
    "name", ["dinov2", "dinov2_3p", "curia", "curia_3p"]
)  # fmt: skip
def test_slice_map_keeps_volume_orientation(name):
    encoder = SLICE_ENCODERS[name]
    planes = encoder.planes
    patch = 14 if encoder.size == SLICE_SIZE else 16
    volume = np.zeros((48, 48, 48), np.float32)
    volume[4:8, 20:24, 40:44] = 1.0  # bright cube, distinct position on each axis
    model = PatchMeanModel(patch)
    fmap = slice_feature_map(model, volume, planes, "cpu", batch_size=16, encoder=encoder)
    g = encoder.size // patch
    assert fmap.shape == ((1, 48, g, g) if planes == ("axial",) else (1, g, g, g))
    peak = np.array(np.unravel_index(np.argmax(fmap[0]), fmap.shape[1:]))
    # Cube centre (voxels 5.5, 21.5, 41.5) in feature-cell coordinates.
    scale = np.array(fmap.shape[1:]) / 48
    cube_centre = (np.array([5.5, 21.5, 41.5]) + 0.5) * scale - 0.5
    assert np.all(np.abs(peak - cube_centre) <= 1.5)


def test_curia_slices_superior_up_and_zscored():
    import torch

    volume = torch.arange(32.0).view(32, 1, 1).expand(32, 32, 32).contiguous()  # value = z
    model = PatchMeanModel(16)
    plane_tokens(model, volume, 1, 64, size=32, inputs="zscore", superior_up=True)
    x = model.inputs[0]  # coronal slices (n, 1, z rows, x columns)
    assert x.shape == (32, 1, 32, 32)
    assert x[0, 0, 0].mean() > x[0, 0, -1].mean()  # superior (high z) at the top
    assert torch.allclose(x.mean((1, 2, 3)), torch.zeros(32), atol=1e-5)
    assert torch.allclose(x.std((1, 2, 3)), torch.ones(32), atol=1e-4)


def test_to_unit_windows():
    ct = np.array([-1024.0, -160.0, 40.0, 240.0, 3000.0])
    assert np.allclose(to_unit(ct, "CT"), [0, 0, 0.5, 1, 1])
    assert np.allclose(to_unit(np.array([0.0, 10.0, 50.0]), "PT"), [0, 0.5, 1])
    mr = to_unit(np.arange(1000.0), "MR")
    assert mr.min() == 0 and mr.max() == 1


def test_design_rows_for_the_2d_baseline():
    import pandas as pd

    from nnfoundation_bench.modeling.predict import design_ids, design_table

    ids = design_ids()
    assert ids["dinov2_frozen_roi_mean_logreg"] == "B2"
    assert ids["dinov2_3p_frozen_roi_margin_mean_logreg"] == "B6"
    assert ids["vit_frozen_mean_logreg"] == "F1"
    assert ids["curia_frozen_roi_mean_logreg"] == "C2"
    assert ids["curia2_3p_frozen_roi_margin_mean_logreg"] == "C12"
    grid = design_table({"t": pd.Series({"dinov2_3p_frozen_mean_cox": "0.6"})}, kinds={"t": "cox"})
    assert "| B4 | 0.6 |" in grid


def test_l2_poolings_weigh_every_token_the_same():
    import torch

    from nnfoundation_bench.modeling.features import pool_features

    rng = np.random.default_rng(0)
    fmap = rng.normal(size=(8, 4, 4, 4)).astype(np.float32)
    fmap[:, 0, 0, 0] *= 100.0  # one high-norm token
    mask = np.zeros((8, 8, 8), np.uint8)
    mask[:2, :2, :2] = 1  # covers cell (0, 0, 0) only
    mask[2:4, :2, :2] = 2  # cell (1, 0, 0)
    pooled = pool_features(fmap, mask, ((1, "gtvp"), (2, "gtvn")), l2=True)
    unit = fmap / np.linalg.norm(fmap, axis=0, keepdims=True)
    assert np.allclose(pooled["mean_l2"], unit.reshape(8, -1).mean(1), atol=1e-6)
    assert np.allclose(pooled["gtvp_mean_l2"], unit[:, 0, 0, 0], atol=1e-6)
    assert np.allclose(pooled["roi_mean_l2"], unit[:, :2, 0, 0].mean(1), atol=1e-6)
    assert np.allclose(pooled["roi_mean"], fmap[:, :2, 0, 0].mean(1), atol=1e-5)
    # A tensor gives the same vectors.
    on_device = pool_features(torch.from_numpy(fmap), mask, ((1, "gtvp"), (2, "gtvn")), l2=True)
    assert on_device.keys() == pooled.keys()
    assert all(np.allclose(on_device[k], pooled[k], atol=1e-5) for k in pooled)
    assert "roi_mean_l2" not in pool_features(fmap, mask)


def test_slice_encoders_also_probe_l2_poolings(tmp_path):
    from nnfoundation_bench.modeling.features import _missing_keys, task_poolings

    assert task_poolings("lld_mmri_malignant_delay", "vit") == (
        "mean", "roi_mean", "roi_margin_mean",
    )  # fmt: skip
    assert "roi_mean_l2" in task_poolings("lld_mmri_malignant_delay", "dinov2")
    hecktor = task_poolings("hecktor2022_rfs_pet", "dinov2_3p")
    assert "gtvn_mean_l2" in hecktor and "roi_concat_mean_l2" not in hecktor
    path = tmp_path / "c.npz"
    np.savez(path, roi_mean=np.zeros(2))
    assert _missing_keys(path, ["roi_mean", "roi_mean_l2"])
    assert not _missing_keys(path, ["roi_mean"])
    assert _missing_keys(tmp_path / "absent.npz", ["roi_mean"])

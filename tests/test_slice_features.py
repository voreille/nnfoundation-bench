from types import SimpleNamespace

import numpy as np
import pytest

from nnfoundation_bench.modeling.slice_features import SLICE_SIZE, slice_feature_map, to_unit


class PatchMeanModel:
    """Stand-in for a 2D ViT: one CLS token, then each patch's mean intensity as a
    1-channel token (row-major), so feature maps can be checked against the input."""

    config = SimpleNamespace(patch_size=14, num_register_tokens=0)

    def __call__(self, pixel_values):
        import torch
        import torch.nn.functional as F

        patches = F.avg_pool2d(pixel_values[:, :1], self.config.patch_size)
        tokens = patches.flatten(2).transpose(1, 2)  # (B, g*g, 1)
        cls = torch.zeros(len(tokens), 1, 1)
        return SimpleNamespace(last_hidden_state=torch.cat([cls, tokens], dim=1))


@pytest.mark.parametrize("planes", [("axial",), ("axial", "coronal", "sagittal")])
def test_slice_map_keeps_volume_orientation(planes):
    volume = np.zeros((48, 48, 48), np.float32)
    volume[4:8, 20:24, 40:44] = 1.0  # bright cube, distinct position on each axis
    fmap = slice_feature_map(PatchMeanModel(), volume, planes, "cpu", batch_size=16)
    g = SLICE_SIZE // 14
    assert fmap.shape == ((1, 48, g, g) if planes == ("axial",) else (1, g, g, g))
    peak = np.array(np.unravel_index(np.argmax(fmap[0]), fmap.shape[1:]))
    # Cube centre (voxels 5.5, 21.5, 41.5) in feature-cell coordinates.
    scale = np.array(fmap.shape[1:]) / 48
    cube_centre = (np.array([5.5, 21.5, 41.5]) + 0.5) * scale - 0.5
    assert np.all(np.abs(peak - cube_centre) <= 1.5)


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
    grid = design_table({"t": pd.Series({"dinov2_3p_frozen_mean_cox": "0.6"})}, kinds={"t": "cox"})
    assert "| B4 | 0.6 |" in grid

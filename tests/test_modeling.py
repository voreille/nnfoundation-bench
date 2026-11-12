import math

import numpy as np

from nnfoundation_bench.modeling.predict import binary_metrics
from nnfoundation_bench.modeling.runs import run_name
from nnfoundation_bench.modeling.train import build_command


def test_run_name_defaults_to_paper_lr():
    assert run_name("cnn") == "cnn_lr0.0003"
    assert run_name("vit", scratch=True, tag="x") == "vit_scratch_lr0.0001_x"


def test_binary_metrics():
    m = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
    assert m["auroc"] == 0.75 and m["n_positive"] == 2
    assert math.isnan(binary_metrics(np.array([0, 0]), np.array([0.1, 0.2]))["auroc"])


def test_gradient_accumulation_reaches_effective_batch(monkeypatch):
    monkeypatch.setattr("nnfoundation_bench.modeling.train.get_checkpoint", lambda m: "ckpt.pth")
    cmd = build_command("t", "cnn", 0, 3e-4, "r", 2, 1, 100, 50, False, 0, 4)
    assert cmd[cmd.index("--training.gradient_accumulation_steps") + 1] == "16"
    assert cmd[cmd.index("--optimizer.warmup_epochs") + 1] == "10"
    assert "--multi_gpu" in cmd and "--model.encoder.no_pretrained" not in cmd


def test_train_split_repeats_to_fixed_epoch_length(tmp_path, monkeypatch):
    import json

    from glovita.datasets.generic_3d_dataset import Generic3DDataset

    from nnfoundation_bench.modeling.glovita_entry import _patch_dataset

    # Restore the original methods after the test.
    monkeypatch.setattr(Generic3DDataset, "__len__", Generic3DDataset.__len__)
    monkeypatch.setattr(Generic3DDataset, "__getitem__", Generic3DDataset.__getitem__)
    _patch_dataset(min_train_samples=10)

    (tmp_path / "labels.json").write_text(json.dumps({"a": 0, "b": 1, "c": 0}))
    split = {"train": ["a", "b"], "val": ["c"], "test": []}
    (tmp_path / "splits.json").write_text(json.dumps(split))
    assert len(Generic3DDataset(tmp_path, "train")) == 10
    assert len(Generic3DDataset(tmp_path, "val")) == 1

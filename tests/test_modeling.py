import math

import numpy as np

from nnfoundation_bench.modeling.predict import binary_metrics, group_prior_scores
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


def test_group_prior_scores_use_train_val_rates_only():
    import pandas as pd

    labels = pd.DataFrame(
        {"case_id": list("abcdef"), "label": [1, 1, 0, 0, 1, 0], "group": list("AABBAC")}
    )
    split = {"train": ["a", "c"], "val": ["b", "d"], "test": ["e", "f"]}
    scores = group_prior_scores(labels, split).set_index("case_id")["score"]
    assert scores["e"] == 1.0  # group A is all positive in train+val
    assert scores["f"] == 0.5  # unseen group C -> overall train+val rate


def test_design_table_cells_are_cnn_slash_vit():
    import pandas as pd

    from nnfoundation_bench.modeling.predict import design_table

    summary = pd.Series(
        {"cnn_frozen_mean_logreg": "0.51 ± 0.06", "vit_frozen_mean_logreg": "0.77 ± 0.05",
         "group_prior": "0.84 ± 0.01"}
    )  # fmt: skip
    lines = design_table({"hecktor": summary}).splitlines()
    assert lines[0] == "| ID | hecktor |"
    assert "| F1 | 0.51 ± 0.06 / 0.77 ± 0.05 |" in lines
    assert "| F2 | - / - |" in lines
    assert lines[-1] == "| ref | 0.84 ± 0.01 |"


def test_last_checkpoint_gets_its_own_results_name():
    import pytest

    from nnfoundation_bench.modeling.runs import eval_run_name

    assert eval_run_name("vit_lr0.0001") == "vit_lr0.0001"
    assert eval_run_name("vit_lr0.0001", "last") == "vit_lr0.0001_last"
    with pytest.raises(ValueError):
        eval_run_name("vit_lr0.0001", "epoch3")


def test_update_generated_rewrites_only_marked_blocks():
    from nnfoundation_bench.modeling.predict import update_generated

    text = (
        "intro\n<!-- generated: runs t -->\nstale\n<!-- end generated -->\n"
        "kept\n<!-- generated: design -->\n<!-- end generated -->\n"
    )
    out = update_generated(text, render=lambda spec: f"[{spec}]")
    assert out == (
        "intro\n<!-- generated: runs t -->\n[runs t]\n<!-- end generated -->\n"
        "kept\n<!-- generated: design -->\n[design]\n<!-- end generated -->\n"
    )


def test_runs_table_sorted_with_design_ids():
    import pandas as pd

    from nnfoundation_bench.modeling.predict import runs_table

    stats = pd.DataFrame(
        {"mean": [0.6, 0.8, 0.7], "std": [0.01, 0.02, 0.03], "count": [5, 5, 3]},
        index=["cnn_frozen_mean_logreg", "group_prior", "other_run"],
    )
    lines = runs_table(stats).splitlines()
    assert lines[2] == "| `group_prior` | ref | 0.800 ± 0.020 |"
    assert lines[3] == "| `other_run` |  | 0.700 ± 0.030 (n=3) |"
    assert lines[4] == "| `cnn_frozen_mean_logreg` | F1 | 0.600 ± 0.010 |"

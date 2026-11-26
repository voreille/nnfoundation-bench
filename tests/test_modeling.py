import math

import numpy as np
import pytest

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


def test_within_group_auroc_ignores_cross_group_pairs():
    import numpy as np

    from nnfoundation_bench.modeling.predict import within_group_auroc

    groups = np.array(["a", "a", "a", "b", "b", "b"])
    labels = np.array([1, 0, 0, 1, 1, 0])
    # Group prior alone (constant within each group): exactly 0.5.
    assert within_group_auroc(labels, np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1]), groups) == 0.5
    # Perfect within groups although group b scores are all below group a's.
    scores = np.array([0.9, 0.5, 0.6, 0.3, 0.2, 0.1])
    assert within_group_auroc(labels, scores, groups) == 1.0
    # 4 pairs (a: 1x2, b: 2x1); one discordant pair in b.
    scores = np.array([0.9, 0.5, 0.6, 0.3, 0.05, 0.1])
    assert within_group_auroc(labels, scores, groups) == 0.75
    # No group with both classes.
    assert np.isnan(
        within_group_auroc(np.array([1, 0]), np.array([0.1, 0.2]), np.array(["a", "b"]))
    )


def test_covariate_matrix_keeps_numeric_nan_and_one_hot_encodes_groups():
    import numpy as np
    import pandas as pd

    from nnfoundation_bench.modeling.predict import covariate_matrix, covariate_run_name

    labels = pd.DataFrame({"age": [50.0, None, 70.0], "group": ["A", "B", "A"]})
    x = covariate_matrix(labels, ["age", "group"])
    assert x.shape == (3, 3) and np.isnan(x[1, 0])
    np.testing.assert_array_equal(x[:, 1:], [[1, 0], [0, 1], [1, 0]])
    assert covariate_run_name(["age", "group"]) == "covariates_age_group_logreg"


def test_design_table_has_fusion_and_covariate_reference_rows():
    import pandas as pd

    from nnfoundation_bench.modeling.predict import design_table

    summary = pd.Series(
        {"vit_fused_stack_roi_mean_logreg": "0.90 ± 0.01", "covariates_age_logreg": "0.80 ± 0.02"}
    )
    lines = design_table({"t": summary}).splitlines()
    assert "| M4 | - / 0.90 ± 0.01 |" in lines
    assert "| ref-age | 0.80 ± 0.02 |" in lines
    assert lines[-1] == "| ref | n/a |"


def test_fused_task_reads_labels_of_its_first_source(tmp_path, monkeypatch):
    import json

    from nnfoundation_bench.modeling import runs

    monkeypatch.setattr(runs, "RESULTS_DIR", tmp_path)
    assert runs.processed_task("plain") == "plain"
    (tmp_path / "fused").mkdir()
    (tmp_path / "fused" / "fusion.json").write_text(json.dumps({"tasks": ["a", "b"]}))
    assert runs.fusion_sources("fused") == ["a", "b"]
    assert runs.processed_task("fused") == "a"


def test_single_fold_cell_has_no_std():
    import pandas as pd

    from nnfoundation_bench.modeling.predict import format_auroc

    row = pd.Series({"mean": 0.6543, "std": float("nan"), "count": 1})
    assert format_auroc(row) == "0.654 (n=1)"


# ------------------------------------------------------------ organ tasks (issue #4)

ORGAN_TASK = "totalseg_ct_organs"
ORGAN_TASKS = (ORGAN_TASK, f"{ORGAN_TASK}_official", "totalseg_mri_organs",
               "totalseg_mri_organs_official")  # fmt: skip


def test_multiclass_metrics_perfect_predictions():
    from nnfoundation_bench.modeling.predict import multiclass_metrics

    labels = np.array([0, 1, 2, 0, 1, 2])
    proba = np.eye(3)[labels] * 0.8 + 0.2 / 3
    m = multiclass_metrics(labels, proba, [0, 1, 2])
    assert m["balanced_accuracy"] == 1.0 and m["accuracy"] == 1.0
    assert m["auroc_macro"] == 1.0 and m["n"] == 6


def test_multiclass_metrics_with_a_class_missing_from_test():
    from nnfoundation_bench.modeling.predict import multiclass_metrics

    labels = np.array([0, 0, 1, 1])
    p0 = np.array([0.8, 0.3, 0.4, 0.1])
    p2 = np.full(4, 0.05)
    proba = np.column_stack([p0, 1 - p0 - p2, p2])
    m = multiclass_metrics(labels, proba, [0, 1, 2])
    # One-vs-rest AUROC over the present classes 0 and 1: 0.75 each.
    assert m["auroc_macro"] == 0.75
    assert m["accuracy"] == 0.75  # predictions 0, 1, 1, 1
    assert m["balanced_accuracy"] == 0.75  # recall 1/2 (class 0), 2/2 (class 1)
    assert m["n"] == 4


def test_multiclass_metrics_balanced_accuracy_weights_classes_equally():
    from nnfoundation_bench.modeling.predict import multiclass_metrics

    labels = np.array([0] * 8 + [1] * 2)
    proba = np.tile([0.9, 0.1], (10, 1))  # always predicts class 0
    m = multiclass_metrics(labels, proba, [0, 1])
    assert m["accuracy"] == 0.8 and m["balanced_accuracy"] == 0.5


def _patch_dirs(tmp_path, monkeypatch):
    from nnfoundation_bench.modeling import features, predict, runs, survival

    for module in (predict, survival, features):
        monkeypatch.setattr(module, "PROCESSED_DATA_DIR", tmp_path / "processed")
    for module in (predict, runs, features):
        monkeypatch.setattr(module, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(runs, "FINETUNED_DIR", tmp_path / "finetuned")
    (tmp_path / "results").mkdir(exist_ok=True)


def test_metric_key_is_balanced_accuracy_for_organ_tasks(tmp_path, monkeypatch):
    from nnfoundation_bench.modeling.predict import metric_key

    _patch_dirs(tmp_path, monkeypatch)
    for task in ORGAN_TASKS:
        assert metric_key(task) == "balanced_accuracy"
    assert metric_key("kits_liver") == "auroc"


FOLD_VALUES = {
    # run -> balanced_accuracy per fold (accuracy = +0.05, auroc_macro = +0.1)
    "cnn_frozen_mean_logreg": [0.60, 0.62, 0.64, 0.66, 0.68],
    "vit_frozen_mean_logreg": [0.80, 0.81, 0.82, 0.83, 0.84],
    "cnn_frozen_roi_mean_logreg": [0.70, 0.70, 0.71, 0.71, 0.72],
    "vit_frozen_roi_mean_logreg": [0.90, 0.92, 0.94, 0.96, 0.98],
    "cnn_frozen_roi_margin_mean_logreg": [0.50, 0.55, 0.60, 0.65, 0.70],
    "vit_frozen_roi_margin_mean_logreg": [0.75, 0.75, 0.75, 0.75, 0.76],
}


def _write_organ_results(tmp_path, task=ORGAN_TASK):
    import json

    for run, values in FOLD_VALUES.items():
        for fold, v in enumerate(values):
            out = tmp_path / "results" / task / run / f"fold{fold}"
            out.mkdir(parents=True)
            metrics = {"task": task, "run": run, "fold": fold, "balanced_accuracy": v,
                       "accuracy": v + 0.05, "auroc_macro": min(v + 0.1, 1.0), "n": 100,
                       "n_classes": 98, "val_balanced_accuracy": v, "probe_C": 1.0}  # fmt: skip
            (out / "metrics.json").write_text(json.dumps(metrics))


def _write_binary_results(tmp_path, task="kits_liver"):
    import json

    for run, v in (("cnn_frozen_mean_logreg", 0.71), ("vit_frozen_mean_logreg", 0.83)):
        for fold in range(5):
            out = tmp_path / "results" / task / run / f"fold{fold}"
            out.mkdir(parents=True)
            metrics = {"task": task, "run": run, "fold": fold, "auroc": v + fold * 0.01,
                       "ap": 0.5, "n": 40, "n_positive": 20}  # fmt: skip
            (out / "metrics.json").write_text(json.dumps(metrics))


def _cell(metric, offset=0.0):
    import pandas as pd

    from nnfoundation_bench.modeling.predict import format_auroc

    def fmt(run):
        v = np.minimum(np.array(FOLD_VALUES[run]) + offset, 1.0)
        return format_auroc(pd.Series({"mean": v.mean(), "std": v.std(ddof=1), "count": 5}))

    return {pool: f"{fmt(f'cnn_frozen_{pool}_logreg')} / {fmt(f'vit_frozen_{pool}_logreg')}"
            for pool in ("mean", "roi_mean", "roi_margin_mean")}  # fmt: skip


def test_summarize_organ_task_uses_balanced_accuracy(tmp_path, monkeypatch):
    import pandas as pd

    from nnfoundation_bench.modeling.predict import summarize

    _patch_dirs(tmp_path, monkeypatch)
    _write_organ_results(tmp_path)
    summarize(ORGAN_TASK)
    summary = pd.read_csv(tmp_path / "results" / ORGAN_TASK / "summary.csv", index_col=0)
    assert summary.loc["vit_frozen_roi_mean_logreg", "mean"] == pytest.approx(0.94)
    assert summary.loc["cnn_frozen_mean_logreg", "mean"] == pytest.approx(0.64)
    assert summary.loc["cnn_frozen_mean_logreg", "count"] == 5


def test_multiclass_block_renders_three_metrics_rows_f1_to_f3(tmp_path, monkeypatch):
    from nnfoundation_bench.modeling.predict import render_block, update_generated

    _patch_dirs(tmp_path, monkeypatch)
    _write_organ_results(tmp_path)
    _write_binary_results(tmp_path)
    text = render_block(f"multiclass {ORGAN_TASK}")
    lines = text.splitlines()
    rows = {
        pool: next(line for line in lines if line.startswith(f"| {row} |"))
        for pool, row in (("mean", "F1"), ("roi_mean", "F2"), ("roi_margin_mean", "F3"))
    }
    for offset in (0.0, 0.05, 0.1):  # balanced accuracy, accuracy, macro AUROC
        for pool, cell in _cell("x", offset).items():
            assert cell in rows[pool], (pool, offset, rows[pool])
    assert "kits_liver" not in text
    # Default pattern: every organ task with results, never binary tasks.
    everything = render_block("multiclass")
    assert ORGAN_TASK in everything and "kits_liver" not in everything
    doc = f"<!-- generated: multiclass {ORGAN_TASK} -->\nstale\n<!-- end generated -->\n"
    assert update_generated(doc) == (
        f"<!-- generated: multiclass {ORGAN_TASK} -->\n{text}\n<!-- end generated -->\n"
    )


def test_design_and_runs_blocks_skip_organ_tasks(tmp_path, monkeypatch):
    from nnfoundation_bench.modeling.predict import render_block

    _patch_dirs(tmp_path, monkeypatch)
    _write_organ_results(tmp_path)
    _write_binary_results(tmp_path)
    design = render_block("design")
    assert "kits_liver" in design.splitlines()[0]
    assert ORGAN_TASK not in design
    assert "| F1 | 0.730 ± 0.016 / 0.850 ± 0.016 |" in design.splitlines()
    runs_block = render_block(f"runs {ORGAN_TASK}")
    assert not any(run in runs_block for run in FOLD_VALUES)
    assert "cnn_frozen_mean_logreg" in render_block("runs kits_liver")  # binary unchanged


# Guards: organ tasks are probe-only (issue #4 non-goals).


@pytest.fixture
def organ_cli(tmp_path, monkeypatch):
    """Prepared-looking organ tasks (labels, splits, features) and every data root in
    tmp_path, so that only a guard stops the commands below."""
    import json
    import subprocess

    import pandas as pd

    from nnfoundation_bench.modeling import cox_probe, radiomics_baseline, train

    _patch_dirs(tmp_path, monkeypatch)
    for module in (train, cox_probe, radiomics_baseline):
        monkeypatch.setattr(module, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(radiomics_baseline, "INTERIM_DATA_DIR", tmp_path / "interim")
    monkeypatch.setattr(train, "get_checkpoint", lambda model: "ckpt.pth")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    rng = np.random.default_rng(0)
    rows = [
        {"case_id": f"s{i:04d}__{o}", "scan_id": f"s{i:04d}", "organ": o, "label": k,
         "official_split": "train" if i < 15 else "test"}
        for i in range(20) for k, o in enumerate(("aorta", "liver"))
    ]  # fmt: skip
    table = pd.DataFrame(rows)
    cases = table["case_id"].tolist()
    split = {"train": cases[:16], "val": cases[16:20], "test": cases[20:]}
    for task in ORGAN_TASKS:
        root = tmp_path / "processed" / task
        for model in ("cnn", "vit"):
            (root / "features" / model).mkdir(parents=True)
            for c, k in zip(cases, table["label"], strict=True):
                vec = rng.normal(size=4) + 3 * k
                np.savez(root / "features" / model / f"{c}.npz", mean=vec, roi_mean=vec,
                         roi_margin_mean=vec, roi_inside=np.float32(1.0))  # fmt: skip
        table.to_csv(root / "labels.csv", index=False)
        (root / "labels.json").write_text(
            json.dumps(dict(zip(cases, table["label"].tolist(), strict=True)))
        )
        (root / "splits.json").write_text(json.dumps([split]))
        (root / "dataset.json").write_text(json.dumps({"layout": "scan", "num_classes": 2}))
    return tmp_path, calls


def _assert_refused(result, tmp_path):
    import re

    assert result.exit_code != 0, result.output
    message = f"{result.output} {result.exception}"
    for task in sorted(ORGAN_TASKS, key=len, reverse=True):
        message = message.replace(task, "<task>")
    message = message.replace(str(tmp_path), "<tmp>")
    assert not isinstance(result.exception, (FileNotFoundError, KeyError)), message
    # A clear error says why: organ / multi-class tasks are not supported here.
    assert re.search(r"organ|multi-?class", message, re.IGNORECASE), message


GUARDED = [
    ("train", [ORGAN_TASK, "--gpus", "0"]),
    ("predict", ["predict", ORGAN_TASK, "--model", "cnn"]),
    ("predict", ["group-baseline", ORGAN_TASK]),
    ("predict", ["covariate-baseline", ORGAN_TASK]),
    ("predict", ["table", ORGAN_TASK, "--within"]),
    (
        "features",
        ["fuse", ORGAN_TASK, f"{ORGAN_TASK}_official", "--name", "fx", "--model", "vit"],
    ),
    ("features", ["combine", ORGAN_TASK, "--model", "vit"]),
    ("cox", ["probe", ORGAN_TASK, "--model", "vit"]),
    ("cox", ["combine", ORGAN_TASK, "--model", "vit"]),
    ("radiomics", ["extract", ORGAN_TASK, "--jobs", "1"]),
    ("radiomics", ["probe", ORGAN_TASK]),
    ("predict", ["group-baseline", "totalseg_mri_organs_official"]),
]


@pytest.mark.parametrize(
    ("cli", "args"),
    GUARDED,
    ids=[
        f"{cli}-{a[0] if cli != 'train' else 'train'}-{a[1] if cli != 'train' else a[0]}"
        for cli, a in GUARDED
    ],
)
def test_unsupported_commands_refuse_organ_tasks(organ_cli, cli, args):
    from typer.testing import CliRunner

    from nnfoundation_bench.modeling import cox_probe, features, predict, radiomics_baseline, train

    tmp_path, calls = organ_cli
    app = {"train": train.app, "predict": predict.app, "features": features.app,
           "cox": cox_probe.app, "radiomics": radiomics_baseline.app}[cli]  # fmt: skip
    result = CliRunner().invoke(app, args)
    _assert_refused(result, tmp_path)
    assert calls == []  # nothing launched (glovita, accelerate)
    assert not (tmp_path / "results" / "fx").exists()
    written = list((tmp_path / "results").glob("*/*/fold*/metrics.json"))
    assert written == []


def test_features_probe_still_works_on_organ_tasks(organ_cli):
    import json

    from typer.testing import CliRunner

    from nnfoundation_bench.modeling import features

    tmp_path, _ = organ_cli
    result = CliRunner().invoke(features.app, ["probe", ORGAN_TASK, "--pooling", "mean"])
    assert result.exit_code == 0, result.output
    metrics = json.loads(
        (
            tmp_path / "results" / ORGAN_TASK / "cnn_frozen_mean_logreg/fold0/metrics.json"
        ).read_text()
    )
    assert metrics["balanced_accuracy"] == 1.0

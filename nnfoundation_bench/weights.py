"""Pretrained checkpoints (Hugging Face, pinned revisions): nnFoundation (3D) and the 2D
natural-image encoders used slice-wise as baselines (D38)."""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from nnfoundation_bench.config import PRETRAINED_DIR


@dataclass(frozen=True)
class Pretrained:
    repo: str
    revision: str
    # Peak fine-tuning learning rate for classification (paper, Table 1).
    lr: float

    @property
    def path(self) -> Path:
        return PRETRAINED_DIR / self.repo.split("/")[-1] / "checkpoint_final.pth"


MODELS = {
    "cnn": Pretrained(
        "MIC-DKFZ/nnFoundationCNN", "f443c09319988e833b3d9589bf95909e4443d7b2", lr=3e-4
    ),
    "vit": Pretrained(
        "MIC-DKFZ/nnFoundationViT", "c691f2115140458ec64017712eb8ebbc8dde6ae4", lr=1e-4
    ),
}


def get_checkpoint(name: str) -> Path:
    """Local path of the ``name`` checkpoint, downloading it on first use."""
    model = MODELS[name]
    if not model.path.exists():
        from huggingface_hub import hf_hub_download

        logger.info(f"Downloading {model.repo}@{model.revision[:8]}")
        for filename in ("checkpoint_final.pth", "adaptation_plan.json"):
            hf_hub_download(
                model.repo, filename, revision=model.revision, local_dir=model.path.parent
            )
    return model.path


@dataclass(frozen=True)
class HubModel:
    """A Hugging Face ``transformers`` model, stored as a snapshot of its repo."""

    repo: str
    revision: str

    @property
    def path(self) -> Path:
        return PRETRAINED_DIR / self.repo.split("/")[-1]


# 2D encoders (natural images), run slice by slice on the crops (D38). DINOv3 is gated
# (access pending), so DINOv2 ViT-L/14 with registers comes first.
HUB_MODELS = {
    "dinov2": HubModel(
        "facebook/dinov2-with-registers-large", "e4c89a4e05589de9b3e188688a303d0f3c04d0f3"
    ),
    # Radiology 2D encoders (Raidium, DINOv2 ViT-B on CT and MR slices, D40). Gated with
    # automatic approval: accept the terms on the model page, then log in (`hf auth login`).
    "curia": HubModel("raidium/curia", "9657dc56276bc6c9503ef6f8d060879c8bee482f"),
    "curia2": HubModel("raidium/curia-2", "645f566dd9e002505691178917cee265b491c7f7"),
}


def get_hub_model(name: str) -> Path:
    """Local snapshot of the ``name`` 2D encoder, downloading it on first use."""
    model = HUB_MODELS[name]
    if not (model.path / "model.safetensors").exists():
        from huggingface_hub import snapshot_download

        logger.info(f"Downloading {model.repo}@{model.revision[:8]}")
        snapshot_download(
            model.repo,
            revision=model.revision,
            local_dir=model.path,
            allow_patterns=["*.json", "model.safetensors", "LICENSE"],
        )
    return model.path

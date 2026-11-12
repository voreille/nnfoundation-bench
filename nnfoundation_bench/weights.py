"""Pretrained nnFoundation checkpoints (Hugging Face, pinned revisions)."""

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

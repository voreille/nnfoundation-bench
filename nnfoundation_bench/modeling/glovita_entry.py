"""glovita's training entry point with two adjustments for small datasets.

Launched by ``train.py`` via ``accelerate launch -m``. glovita is pinned in
pyproject.toml, so the patched internals below are stable.

1. Fixed-length epochs. glovita's ``train_steps_per_epoch`` only truncates an epoch,
   so with ~200 training cases an "epoch" is ~6 optimizer steps instead of the
   paper's 50. We repeat the training cases (reshuffled each epoch) until an epoch
   holds ``NNFB_MIN_TRAIN_SAMPLES`` samples, as nnU-Net-style iteration epochs do.
2. Only ``last.pt`` and ``best.pt`` are kept (glovita also writes one file per epoch,
   which is ~40 GB per CNN run).
"""

import os

import torch


def _patch_dataset(min_train_samples: int) -> None:
    from glovita.datasets.generic_3d_dataset import Generic3DDataset

    original_getitem = Generic3DDataset.__getitem__

    def __len__(self):
        n = len(self.case_ids)
        return max(n, min_train_samples) if self.split == "train" and n else n

    def __getitem__(self, idx):
        return original_getitem(self, idx % len(self.case_ids))

    Generic3DDataset.__len__ = __len__
    Generic3DDataset.__getitem__ = __getitem__


def _patch_checkpointing() -> None:
    from glovita.training.trainer import Trainer

    def _save_checkpoint(self, model, optimizer, epoch, is_best=False):
        ckpt_dir = self.log_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(model)
        state = {"epoch": epoch, "model": unwrapped.state_dict()}
        plan = getattr(getattr(unwrapped, "encoder", None), "adaptation_plan", None)
        if plan is not None:
            state["nnssl_adaptation_plan"] = plan
        torch.save(state, ckpt_dir / ("best.pt" if is_best else "last.pt"))

    Trainer._save_checkpoint = _save_checkpoint


def main() -> None:
    _patch_dataset(int(os.environ.get("NNFB_MIN_TRAIN_SAMPLES", "0")))
    _patch_checkpointing()
    from glovita.cli.train import main as glovita_train

    glovita_train()


if __name__ == "__main__":
    main()

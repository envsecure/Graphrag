"""
accelerator.py — one facade over TPU (torch_xla) / CUDA / CPU.

XLA rules baked in:
  * grads all-reduce inside xm.optimizer_step, then xm.mark_step
  * checkpoints written by rank 0 only
  * dataloader workers stay 0 (xla + forked workers don't mix)
  * bf16 on TPU: launch with XLA_USE_BF16=1 (torch_xla env var)
Multi-core:  torchrun --nproc_per_node=8 train_ner.py --backend tpu
Single:      python train_ner.py --backend tpu     (1 core; cuda/cpu same shape)
"""
from __future__ import annotations

import os

import torch

try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False


class Accelerator:
    def __init__(self, backend: str = "auto", seed: int = 42):
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.device = self._pick(backend)
        torch.manual_seed(seed)

    def _pick(self, backend: str):
        if backend in ("auto", "tpu") and HAS_XLA:
            try:
                if self.world > 1 and not torch.distributed.is_initialized():
                    import torch_xla.distributed.xla_backend  # noqa: F401
                    torch.distributed.init_process_group("xla")
                return xm.xla_device()
            except Exception:
                if backend == "tpu":
                    raise
        if backend in ("auto", "cuda") and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @property
    def is_tpu(self) -> bool:
        return HAS_XLA and self.device.type == "xla"

    def is_master(self) -> bool:
        if self.is_tpu:
            try:
                return bool(xm.is_master_ordinal())
            except Exception:
                pass
        return int(os.environ.get("RANK", "0")) == 0

    def loader(self, dl):
        """Per-core shard when data-parallel; the same loader otherwise.
        ponytail: eval runs on rank 0 only (it has no collectives, so the
        other ranks just wait at the next backward all-reduce)."""
        if self.is_tpu and self.world > 1:
            from torch_xla.distributed.parallel_loader import ParallelLoader
            return ParallelLoader(dl, [self.device]).per_device_loader(self.device)
        return dl

    def optimizer_step(self, optimizer, params, clip: float = 1.0):
        if clip:
            # clipped pre-all-reduce: identical at world=1, close enough at 8
            torch.nn.utils.clip_grad_norm_(params, clip)
        if self.is_tpu:
            xm.optimizer_step(optimizer)       # all-reduce grads, then step
            xm.mark_step()
        else:
            optimizer.step()
        optimizer.zero_grad()

    def to(self, batch: dict) -> dict:
        return {k: v.to(self.device) for k, v in batch.items()}

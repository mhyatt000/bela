from __future__ import annotations

from contextlib import contextmanager, nullcontext
import logging
import time
from typing import Any

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import MetricsTracker
from lerobot.common.utils.utils import has_method
import torch
from torch.amp import GradScaler
from torch.optim import Optimizer

from ..train_tools import cleanup_distributed, setup_distributed


class Trainer:
    """Simple helper class that manages a policy and its optimizer."""

    def __init__(
        self,
        policy: PreTrainedPolicy,
        optimizer: Optimizer,
        grad_scaler: GradScaler,
        lr_scheduler: Optimizer | None = None,
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler
        self.lr_scheduler = lr_scheduler

    def _compute_loss(self, batch: Any, use_amp: bool) -> tuple[torch.Tensor, dict]:
        device = get_device_from_parameters(self.policy)
        self.policy.train()
        with torch.autocast(device_type=device.type) if use_amp else nullcontext():
            return self.policy.forward(batch)

    def _step_optimizer(self, lock=None) -> None:
        ctx = lock if lock is not None else nullcontext()
        with ctx:
            self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.optimizer.zero_grad()

    def update(
        self,
        metrics: MetricsTracker,
        batch: Any,
        clip_norm: float,
        use_amp: bool = False,
        lock=None,
    ) -> tuple[MetricsTracker, dict]:
        start = time.perf_counter()
        loss, out = self._compute_loss(batch, use_amp)
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), clip_norm, error_if_nonfinite=False)
        self._step_optimizer(lock)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if has_method(self.policy, "update"):
            self.policy.update()

        metrics.loss = loss.item()
        metrics.grad_norm = grad_norm.item()
        metrics.lr = self.optimizer.param_groups[0]["lr"]
        metrics.update_s = time.perf_counter() - start
        return metrics, out

    def update_multi(
        self,
        metrics: MetricsTracker,
        batches: dict[str, Any],
        clip_norm: float,
        use_amp: bool = False,
        lock=None,
    ) -> tuple[MetricsTracker, dict]:
        start = time.perf_counter()
        losses, out_dict = [], {}
        for head, batch in batches.items():
            loss, out = self._compute_loss(batch, use_amp)
            losses.append(loss)
            out_dict[head] = out
        loss = torch.stack(losses).mean()
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), clip_norm, error_if_nonfinite=False)
        self._step_optimizer(lock)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if has_method(self.policy, "update"):
            self.policy.update()

        metrics.loss = loss.item()
        metrics.grad_norm = grad_norm.item()
        metrics.lr = self.optimizer.param_groups[0]["lr"]
        metrics.update_s = time.perf_counter() - start
        return metrics, out_dict


@contextmanager
def distributed_context():
    """Setup and cleanup torch distributed mode."""

    is_dist, rank, world_size, device = setup_distributed()
    try:
        yield is_dist, rank, world_size, device
    finally:
        if is_dist:
            cleanup_distributed()
            logging.info("Process group destroyed")

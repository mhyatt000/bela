from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import logging
import time
from typing import Any

from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.scripts.eval import eval_policy
import torch
from torch.amp import GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from bela import train_tools as tt
from bela.common.dataset import make_dataset
from bela.common.datasets.util import postprocess
from bela.common.policies.make import make_policy


@dataclass
class Trainer:
    """Handle distributed offline training."""

    cfg: Any
    rank: int = 0
    world_size: int = 1
    device: torch.device | None = None
    is_dist: bool = False
    wandb_logger: Any | None = None

    def run(self) -> None:
        self._setup()
        self._loop()
        self._cleanup()

    # ---- setup -----------------------------------------------------------------
    def _setup(self) -> None:
        self._setup_dist()
        self._setup_misc()
        self._setup_data()
        self._setup_policy()
        self._setup_opt()
        self._setup_env()

    def _setup_dist(self) -> None:
        self.is_dist, self.rank, self.world_size, self.device = tt.setup_distributed()
        if not self.is_dist:
            self.device = get_safe_torch_device(self.cfg.policy.device, log=True)
        if self.rank != 0:
            logging.getLogger().setLevel(logging.WARNING)

    def _setup_misc(self) -> None:
        if self.cfg.seed is not None:
            set_seed(self.cfg.seed + self.rank)
        if self.rank == 0 and self.cfg.wandb.enable and self.cfg.wandb.project:
            from lerobot.common.utils.wandb_utils import WandBLogger

            self.wandb_logger = WandBLogger(self.cfg)

    def _setup_data(self) -> None:
        self.dataset = make_dataset(self.cfg)
        if self.is_dist:
            base = (
                EpisodeAwareSampler(
                    self.dataset.episode_data_index,
                    drop_n_last_frames=self.cfg.policy.drop_n_last_frames,
                    shuffle=True,
                )
                if hasattr(self.cfg.policy, "drop_n_last_frames")
                else self.dataset
            )
            self.sampler = DistributedSampler(base, num_replicas=self.world_size, rank=self.rank, shuffle=True, seed=self.cfg.seed or 0)
            shuffle = False
        else:
            self.sampler = (
                EpisodeAwareSampler(
                    self.dataset.episode_data_index,
                    drop_n_last_frames=self.cfg.policy.drop_n_last_frames,
                    shuffle=True,
                )
                if hasattr(self.cfg.policy, "drop_n_last_frames")
                else None
            )
            shuffle = self.sampler is None
        bs = self.cfg.batch_size // self.world_size if self.is_dist else self.cfg.batch_size
        self.loader = DataLoader(
            self.dataset,
            num_workers=self.cfg.num_workers,
            prefetch_factor=2,
            batch_size=bs,
            shuffle=shuffle,
            sampler=self.sampler,
            pin_memory=self.device.type != "cpu",
            drop_last=False,
        )
        self.loader_iter = cycle(self.loader)

    def _setup_policy(self) -> None:
        self.policy = make_policy().to(self.device)
        if self.is_dist:
            self.policy = DDP(self.policy, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True)

    def _setup_opt(self) -> None:
        self.optimizer, self.scheduler = make_optimizer_and_scheduler(self.cfg, self.policy)
        self.scaler = GradScaler(self.device.type, enabled=self.cfg.policy.use_amp)

    def _setup_env(self) -> None:
        self.eval_env = None
        if self.rank == 0 and self.cfg.eval_freq > 0 and self.cfg.env is not None:
            self.eval_env = tt.create_distributed_envs(self.cfg.env, self.world_size, self.rank)

    # ---- loop ------------------------------------------------------------------
    def _loop(self) -> None:
        self.step = 0
        self._init_metrics()
        start = time.time()
        last_log = self.step
        for _ in range(self.cfg.steps):
            self._barrier()
            if self.is_dist:
                self.sampler.set_epoch(self.step)
            self._train_step()
            self.step += 1
            if self._should_log():
                self._log(start, last_log)
                last_log = self.step
                start = time.time()
            if self._should_save():
                self._save()
            if self._should_eval():
                self._eval()

    def _init_metrics(self) -> None:
        meters = {
            "loss": AverageMeter("loss", ":.3f"),
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }
        self.tracker = MetricsTracker(
            self.loader.batch_size,
            self.dataset.num_frames,
            self.dataset.num_episodes,
            meters,
            initial_step=self.step,
        )

    def _barrier(self) -> None:
        if self.is_dist:
            dist.barrier()

    def _load_batch(self) -> float:
        start = time.perf_counter()
        self.batch = next(self.loader_iter)
        self.batch = postprocess(self.batch, h="human")
        for k, v in self.batch.items():
            if isinstance(v, torch.Tensor):
                self.batch[k] = v.to(self.device, non_blocking=True)
        return time.perf_counter() - start

    def _train_step(self) -> None:
        self.tracker.dataloading_s = self._load_batch()
        self.tracker, self.out_dict = tt.update_policy(
            self.tracker,
            self.policy,
            self.batch,
            self.optimizer,
            self.cfg.optimizer.grad_clip_norm,
            grad_scaler=self.scaler,
            lr_scheduler=self.scheduler,
            use_amp=self.cfg.policy.use_amp,
        )
        self.tracker.step()

    # ---- conditions ------------------------------------------------------------
    def _should_log(self) -> bool:
        return self.cfg.log_freq > 0 and self.step % self.cfg.log_freq == 0

    def _should_save(self) -> bool:
        return self.cfg.save_checkpoint and (self.step % self.cfg.save_freq == 0 or self.step == self.cfg.steps)

    def _should_eval(self) -> bool:
        return self.rank == 0 and self.eval_env and self.cfg.eval_freq > 0 and self.step % self.cfg.eval_freq == 0

    # ---- actions ---------------------------------------------------------------
    def _log(self, start: float, last_step: int) -> None:
        interval = time.time() - start
        local_steps = self.step - last_step
        total_batches = local_steps * self.world_size
        total_samples = total_batches * self.loader.batch_size
        sps = total_samples / interval
        bps = total_batches / interval
        metrics = self.tracker.to_dict()
        if self.is_dist:
            metrics = tt.all_gather_metrics(metrics, self.device)
        if self.rank == 0:
            for k in self.tracker.meters:
                self.tracker.meters[k].avg = metrics[k]
            gpu = f"{self.world_size} GPU{'s' if self.world_size > 1 else ''}"
            info = f"[{gpu} | {sps:.1f} samples/s | {interval:.2f}s] {self.tracker}"
            logging.info(info)
            if self.wandb_logger:
                self.wandb_logger.log_dict(metrics, self.step)
        self.tracker.reset_averages()

    def _save(self) -> None:
        if self.rank != 0:
            return
        ckpt_dir = get_step_checkpoint_dir(self.cfg.output_dir, self.cfg.steps, self.step)
        model = self.policy.module if self.is_dist else self.policy
        save_checkpoint(ckpt_dir, self.step, self.cfg, model, self.optimizer, self.scheduler)
        update_last_checkpoint(ckpt_dir)
        if self.wandb_logger:
            self.wandb_logger.log_policy(ckpt_dir)

    def _eval(self) -> None:
        step_id = get_step_identifier(self.step, self.cfg.steps)
        with torch.no_grad(), torch.autocast(device_type=self.device.type) if self.cfg.policy.use_amp else nullcontext():
            model = self.policy.module if self.is_dist else self.policy
            info = eval_policy(
                self.eval_env,
                model,
                self.cfg.eval.n_episodes,
                videos_dir=self.cfg.output_dir / "eval" / f"videos_step_{step_id}",
                max_episodes_rendered=4,
                start_seed=self.cfg.seed,
            )
        if self.wandb_logger:
            self.wandb_logger.log_dict(info, self.step, mode="eval")
            self.wandb_logger.log_video(info["video_paths"][0], self.step, mode="eval")

    def _cleanup(self) -> None:
        if self.eval_env:
            self.eval_env.close()
        if self.is_dist:
            tt.cleanup_distributed()

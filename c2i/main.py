# Modified from https://github.com/MCG-NJU/PixNerd

import os
import sys
import torch
import time
import glob
from pathlib import Path
from typing import Any, Union
import random
import hashlib
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lightning import Trainer, LightningModule

from src.data import DataModule
from src.lightning import LightningModel
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback
import lightning.pytorch as pl

import logging
logger = logging.getLogger("lightning.pytorch")

def _seed_from_token(token: str) -> int:
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % (2 ** 31 - 1)


def _shared_run_token() -> Union[str, None]:
    """Return a launch identifier shared by all distributed ranks."""
    explicit = os.environ.get("PIXELDIT2_RUN_TOKEN")
    if explicit:
        return f"explicit:{explicit}"
    elastic = os.environ.get("TORCHELASTIC_RUN_ID")
    if elastic and elastic.lower() != "none":
        return f"torchelastic:{elastic}"
    job = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if job:
        restart = os.environ.get("SLURM_RESTART_COUNT", "0")
        started = os.environ.get("SLURM_JOB_START_TIME", "0")
        return f"slurm:{job}:{restart}:{started}"
    return None


def _wait_for_file(path: str, timeout_s: float = 120.0, interval_s: float = 0.1) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        if os.path.exists(path):
            return True
        time.sleep(interval_s)
    return os.path.exists(path)


def _derive_run_seed(rank: int, seed_file: str):
    """Return a shared per-launch seed and its source."""
    explicit_seed = os.environ.get("PIXELDIT2_RUN_SEED")
    if explicit_seed:
        try:
            return int(explicit_seed) % (2 ** 31 - 1), "PIXELDIT2_RUN_SEED"
        except ValueError:
            logger.warning(f"Ignoring non-integer PIXELDIT2_RUN_SEED={explicit_seed!r}")

    token = _shared_run_token()
    if token is not None:
        return _seed_from_token(token), token

    def _fresh_seed() -> int:
        try:
            t = int(datetime.utcnow().timestamp() * 1e9)
            mix = (t ^ (os.getpid() << 16) ^ int.from_bytes(os.urandom(16), 'big'))
            return _seed_from_token(str(mix & ((1 << 128) - 1)))
        except Exception:
            return int(time.time()) % (2 ** 31 - 1)

    if rank == 0:
        return _fresh_seed(), "wall-clock"

    logger.warning(
        f"Rank {rank}: no shared launch token (TORCHELASTIC_RUN_ID / SLURM_JOB_ID), "
        "falling back to run_seed.txt. On a resume that file may still hold the "
        "previous run's seed; set PIXELDIT2_RUN_TOKEN to a value shared by all "
        "ranks of this launch to make the per-run seed reliable."
    )
    if _wait_for_file(seed_file):
        try:
            with open(seed_file, 'r') as handle:
                return int(handle.read().strip()), f"{seed_file} (may be stale)"
        except Exception:
            pass
    return _fresh_seed(), "wall-clock (rank-local, ranks will disagree)"


class ReWriteRootSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        stamp = time.strftime('%y%m%d%H%M')
        file_path = os.path.join(trainer.default_root_dir, f"config-{stage}-{stamp}.yaml")
        self.parser.save(
            self.config, file_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
        )


class ReWriteRootDirCli(LightningCLI):
    def __init__(self, *args, **kwargs):
        self.auto_resume_ckpt_path = None
        self.run_seed = None
        super().__init__(*args, **kwargs)
    
    def before_instantiate_classes(self) -> None:
        super().before_instantiate_classes()
        config_trainer = self._get(self.config, "trainer", default={})

        if self.subcommand == "predict":
            config_trainer.logger = False
            config_model = self._get(self.config, "model", default={})
            config_model.diffusion_trainer = None
            config_model.ema_tracker = None
            config_data = self._get(self.config, "data", default={})
            config_data.train_dataset = None

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        class TagsClass:
            def __init__(self, exp:str):
                ...
        parser.add_class_arguments(TagsClass, nested_key="tags")

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_default_arguments_to_parser(parser)
        parser.add_argument("--torch_hub_dir", type=str, default=None, help=("torch hub dir"),)
        parser.add_argument("--huggingface_cache_dir", type=str, default=None, help=("huggingface hub dir"),)
        parser.add_argument("--auto_resume", type=bool, default=False, help=("Automatically resume from latest checkpoint if available"),)
        parser.add_argument("--per_run_seed", type=bool, default=True, help=("Enable per-run random seed (synced across ranks)"),)

    def find_latest_checkpoint(self, checkpoint_dir: str) -> Union[str, None]:
        """Find the latest checkpoint in the given directory"""
        if not os.path.exists(checkpoint_dir):
            return None
        
        import re
        
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
        if not checkpoint_files:
            return None
        
        step_pattern = re.compile(r'epoch=\d+-step=(\d+)')
        ckpts_with_steps = []
        
        for ckpt in checkpoint_files:
            match = step_pattern.search(os.path.basename(ckpt))
            if match:
                ckpts_with_steps.append((int(match.group(1)), ckpt))
        
        if not ckpts_with_steps:
            checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        else:
            checkpoint_files = [ckpt for _, ckpt in sorted(ckpts_with_steps, key=lambda x: x[0], reverse=True)]
        
        for ckpt in checkpoint_files:
            try:
                torch.load(ckpt, map_location='cpu', weights_only=False)
                logger.info(f"Found valid checkpoint: {ckpt}")
                return ckpt
            except:
                logger.warning(f"Corrupted checkpoint: {ckpt}")
                continue
        
        return None

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        config_trainer = self._get(self.config_init, "trainer", default={})
        default_root_dir = config_trainer.get("default_root_dir", None)

        if default_root_dir is None:
            default_root_dir = os.path.join(os.getcwd(), "workdirs")

        dirname = ""
        for v, k in self._get(self.config, "tags", default={}).items():
            dirname += f"{v}_{k}"
        default_root_dir = os.path.join(default_root_dir, dirname)
        
        is_resume = self._get(self.config_init, "ckpt_path", default=None)
        auto_resume = self._get(self.config_init, "auto_resume", default=False)
        
        if auto_resume and not is_resume and self.subcommand == "fit":
            latest_ckpt = self.find_latest_checkpoint(default_root_dir)
            if latest_ckpt:
                logger.info(f"Auto-resuming from checkpoint: {latest_ckpt}")
                self.auto_resume_ckpt_path = latest_ckpt
                self.config_init["ckpt_path"] = latest_ckpt
                is_resume = latest_ckpt
            else:
                if os.path.exists(default_root_dir) and os.listdir(default_root_dir):
                    logger.warning(f"Directory {default_root_dir} exists but no checkpoint found. Starting fresh training anyway due to auto_resume=True")
                else:
                    logger.info("No checkpoint found for auto-resume, starting fresh training")
        
        if os.path.exists(default_root_dir) and "debug" not in default_root_dir:
            if os.listdir(default_root_dir) and self.subcommand != "predict" and not is_resume and not auto_resume:
                raise FileExistsError(f"{default_root_dir} already exists. Use --auto_resume=true to automatically resume, or specify --ckpt_path to resume from a specific checkpoint")

        config_trainer.default_root_dir = default_root_dir

        enable_per_run_seed = self._get(self.config_init, "per_run_seed", default=True)
        self.run_seed = None
        rank_int = 0
        if enable_per_run_seed:
            # Use a shared seed for this launch and record it on rank zero.
            seed_file = os.path.join(default_root_dir, "run_seed.txt")

            rank_env = os.environ.get('RANK') or os.environ.get('SLURM_PROCID') or '0'
            try:
                rank_int = int(rank_env)
            except Exception:
                rank_int = 0

            self.run_seed, seed_origin = _derive_run_seed(rank_int, seed_file)
            logger.info(
                f"Rank {rank_int} run seed {self.run_seed} (source: {seed_origin})"
            )
            if rank_int == 0:
                try:
                    os.makedirs(default_root_dir, exist_ok=True)
                    with open(seed_file, 'w') as f:
                        f.write(str(self.run_seed))
                    logger.info(f"Saved run seed {self.run_seed} to {seed_file}")
                except Exception as e:
                    logger.warning(f"Failed to save run seed: {e}")
        else:
            logger.info("per_run_seed disabled; skipping per-run seed generation/seeding")
        
        trainer = super().instantiate_trainer(**kwargs)
        if trainer.is_global_zero:
            os.makedirs(default_root_dir, exist_ok=True)
        # Apply seeding globally including dataloader workers
        if enable_per_run_seed and self.run_seed is not None:
            self._apply_run_seed_to_datamodule()
            rank_seed = int(self.run_seed) + int(rank_int)
            pl.seed_everything(rank_seed, workers=True)
            logger.info(
                f"Applied seed_everything with seed={rank_seed} (base={self.run_seed}, rank={rank_int}, workers=True)"
            )
        return trainer

    def _apply_run_seed_to_datamodule(self) -> None:
        """Set the distributed sampler seed unless the configuration specifies one."""
        datamodule = getattr(self, "datamodule", None)
        if datamodule is None or not hasattr(datamodule, "seed"):
            return

        configured = self._get(self.config, "data", default=None)
        if isinstance(configured, dict):
            configured_seed = configured.get("init_args", {}).get("seed")
        else:
            configured_seed = getattr(configured, "seed", None)
        if configured_seed is not None:
            return  # an explicit seed in the config wins

        datamodule.seed = int(self.run_seed)
        logger.info(f"DataModule sampler seed set to run seed {self.run_seed}")

    def instantiate_classes(self) -> None:
        torch_hub_dir = self._get(self.config, "torch_hub_dir")
        huggingface_cache_dir = self._get(self.config, "huggingface_cache_dir")
        if huggingface_cache_dir is not None:
            os.environ["HUGGINGFACE_HUB_CACHE"] = huggingface_cache_dir
        if torch_hub_dir is not None:
            os.environ["TORCH_HOME"] = torch_hub_dir
            torch.hub.set_dir(torch_hub_dir)
        # The datamodule seed is set after the trainer is instantiated.
        super().instantiate_classes()
    
    def fit(self, **kwargs) -> None:
        ckpt_path = getattr(self, 'auto_resume_ckpt_path', None) or self._get(self.config_init, "ckpt_path", default=None)
        self.trainer.fit(self.model, datamodule=self.datamodule, ckpt_path=ckpt_path)

if __name__ == "__main__":
    from tools.download import resolve_checkpoint
    for i, a in enumerate(sys.argv):
        if a.startswith("--ckpt_path="):
            sys.argv[i] = f"--ckpt_path={resolve_checkpoint(a.split('=', 1)[1])}"

    cli = ReWriteRootDirCli(LightningModel, DataModule,
                            auto_configure_optimizers=False,
                            save_config_callback=ReWriteRootSaveConfigCallback,
                            save_config_kwargs={"overwrite": True})
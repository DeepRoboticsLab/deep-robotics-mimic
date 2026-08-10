import glob
import math
import os
import re
import sys
import time

import torch
import torch.distributed as dist
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

# File path where the NaN-restart checkpoint is written.
# Set via the NAN_RESTART_CHECKPOINT_FILE environment variable by the launcher.
_NAN_RESTART_CHECKPOINT_FILE_ENV = "NAN_RESTART_CHECKPOINT_FILE"


# ---------------------------------------------------------------------------
# Safety patch: torch.distributions.Normal.sample
# ---------------------------------------------------------------------------
# Clamp scale (std) to a small positive value and replace NaN in loc/scale
# to prevent RuntimeError from negative std or NaN propagation.
# ---------------------------------------------------------------------------
_orig_normal_sample = torch.distributions.Normal.sample

def _safe_normal_sample(self, sample_shape=torch.Size()):
    self.loc = torch.nan_to_num(self.loc, nan=0.0, posinf=0.0, neginf=0.0)
    self.scale = torch.clamp(torch.nan_to_num(self.scale, nan=1e-6, posinf=1e-6, neginf=1e-6), min=1e-6)
    return _orig_normal_sample(self, sample_shape)

torch.distributions.Normal.sample = _safe_normal_sample


# ---------------------------------------------------------------------------
# Gloo ↔ CUDA compatibility shims
# ---------------------------------------------------------------------------
# NCCL's broadcast_object_list crashes with "illegal memory access" on some
# consumer GPUs (e.g. RTX 5090) when PhysX GPU simulation is active.
# We use the gloo backend instead but gloo only operates on CPU tensors.
# The helpers below transparently stage CUDA tensors through host memory.
# ---------------------------------------------------------------------------

_orig_broadcast = dist.broadcast
_orig_all_reduce = dist.all_reduce
_orig_broadcast_object_list = dist.broadcast_object_list
_patched = False


def _patch_dist_for_gloo():
    """Wrap torch.distributed ops so gloo works with CUDA tensors."""
    global _patched
    if _patched:
        return
    _patched = True

    def _broadcast(tensor, src=0, group=None, async_op=False):
        if tensor.is_cuda:
            cpu_t = tensor.cpu()
            _orig_broadcast(cpu_t, src=src, group=group, async_op=False)
            tensor.copy_(cpu_t)
            return None
        return _orig_broadcast(tensor, src=src, group=group, async_op=async_op)

    def _all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if tensor.is_cuda:
            cpu_t = tensor.cpu()
            _orig_all_reduce(cpu_t, op=op, group=group, async_op=False)
            tensor.copy_(cpu_t)
            return None
        return _orig_all_reduce(tensor, op=op, group=group, async_op=async_op)

    def _to_cpu(obj):
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            return obj.cpu()
        if isinstance(obj, dict):
            return {k: _to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_to_cpu(v) for v in obj)
        return obj

    def _broadcast_object_list(object_list, src=0, group=None, device=None):
        cpu_list = [_to_cpu(o) for o in object_list]
        _orig_broadcast_object_list(cpu_list, src=src, group=group)
        for i in range(len(object_list)):
            object_list[i] = cpu_list[i]

    dist.broadcast = _broadcast
    dist.all_reduce = _all_reduce
    dist.broadcast_object_list = _broadcast_object_list


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if getattr(self, "logger_type", None) in ["wandb"]:
            policy = getattr(self.alg, "policy", getattr(self.alg, "actor", None))
            normalizer = getattr(self, "obs_normalizer", None)
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(policy, normalizer=normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
    def _configure_multi_gpu(self):
        """Use gloo backend with CPU staging shims to avoid NCCL crashes."""
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.cfg["multi_gpu"] = None
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))
        self.cfg["multi_gpu"] = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }

        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )

        _patch_dist_for_gloo()
        dist.init_process_group(backend="gloo", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        torch.cuda.set_device(self.gpu_local_rank)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def load(self, path: str, load_cfg=None, strict: bool = True, map_location=None):
        """Load checkpoint and sync env.common_step_counter so the curriculum resumes correctly.

        Supports both rsl_rl 3.x (load_optimizer: bool) and 5.x (load_cfg: dict) signatures.
        """
        # Handle callers still using the old 3.x positional arg (bool)
        if isinstance(load_cfg, bool):
            load_cfg = None
        infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        num_steps = getattr(self, "num_steps_per_env", None) or self.cfg.get("num_steps_per_env", 1)
        self.env.unwrapped.common_step_counter = (
            self.current_learning_iteration * num_steps
        )
        return infos

    # ------------------------------------------------------------------
    # NaN detection
    # ------------------------------------------------------------------

    def _find_restart_checkpoint(self, iters_back: int = 1000) -> str | None:
        # rsl_rl 5.x stores log_dir on the logger, not on the runner
        log_dir = getattr(self, "log_dir", None) or getattr(getattr(self, "logger", None), "log_dir", None)
        if not log_dir:
            return None

        cur_iter = self.current_learning_iteration
        target_iter = cur_iter - iters_back  # may be negative, that's fine
        ckpts = glob.glob(os.path.join(log_dir, "model_*.pt"))

        # Parse all checkpoint iterations strictly before current
        parsed = []
        for c in ckpts:
            m = re.search(r"model_(\d+)\.pt$", c)
            if m:
                it = int(m.group(1))
                if it < cur_iter:
                    parsed.append((it, c))

        if not parsed:
            return None

        # Pick the checkpoint closest to (but not exceeding) target_iter.
        # If target_iter is negative (i.e. fewer iters than iters_back),
        # no checkpoint satisfies <= target_iter, so we fall through to
        # selecting the closest checkpoint before the current iteration.
        best_path, best_iter = None, -1
        for it, c in parsed:
            if it <= target_iter and it > best_iter:
                best_iter, best_path = it, c

        # Fallback: pick the most recent checkpoint before current iteration
        if best_path is None:
            parsed.sort(key=lambda x: x[0], reverse=True)
            best_iter, best_path = parsed[0]

        return best_path

    def _signal_nan_restart(self):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(
            f"[NaN] Rank {local_rank}: NaN detected in losses at iteration "
            f"{self.current_learning_iteration}. Triggering restart.",
            flush=True,
        )

        if local_rank == 0:
            ckpt = self._find_restart_checkpoint(iters_back=1000)
            ckpt_file = os.environ.get(_NAN_RESTART_CHECKPOINT_FILE_ENV, "")
            if ckpt_file:
                with open(ckpt_file, "w") as f:
                    f.write(ckpt or "")
            if ckpt:
                print(f"[NaN] Will restart from: {ckpt}", flush=True)
            else:
                print("[NaN] No suitable checkpoint found for restart.", flush=True)

        if local_rank != 0:
            time.sleep(15)
        sys.exit(1)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        original_update = self.alg.update

        def _nan_checked_update():
            try:
                loss_dict = original_update()
            except RuntimeError as e:
                # Catch "normal expects all elements of std >= 0.0" and similar
                print(f"[NaN] RuntimeError during update: {e}", flush=True)
                self._signal_nan_restart()
                return {}  # unreachable (sys.exit above), but keeps type checker happy
            for key, val in loss_dict.items():
                is_nan = (
                    math.isnan(val)
                    if isinstance(val, float)
                    else bool(torch.isnan(torch.as_tensor(val)).any())
                )
                if is_nan:
                    print(f"[NaN] Loss '{key}' = {val}", flush=True)
                    self._signal_nan_restart()
            return loss_dict

        self.alg.update = _nan_checked_update
        try:
            super().learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            self.alg.update = original_update

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, path: str, infos=None):
        super().save(path, infos)
        if getattr(self, "logger_type", None) in ["wandb"]:
            policy = getattr(self.alg, "policy", getattr(self.alg, "actor", None))
            normalizer = getattr(self, "obs_normalizer", None)
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, policy, normalizer=normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None

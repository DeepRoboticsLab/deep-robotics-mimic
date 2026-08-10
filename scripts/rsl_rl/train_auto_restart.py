"""Auto-restart launcher for train.py.

Runs training as a subprocess and automatically restarts from a checkpoint
~1000 iterations earlier when NaN is detected in the training losses.

The signal between train.py and this script is a file path written by
MotionOnPolicyRunner to the path given by NAN_RESTART_CHECKPOINT_FILE:
  - File has content  → NaN was detected, restart from that checkpoint.
  - File is empty     → training crashed for another reason, stop.

Usage
-----
Single GPU:
    python scripts/rsl_rl/train_auto_restart.py [same args as train.py]

Multi-GPU (2 GPUs):
    python scripts/rsl_rl/train_auto_restart.py --nproc_per_node 2 [same args as train.py]

Example:
    python scripts/rsl_rl/train_auto_restart.py --nproc_per_node 2 \\
        --task=Tracking-Flat-DR02_PRO \\
        --logger tensorboard --log_project_name logs/ \\
        --run_name motion_name --headless --max_iterations 200000
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

MAX_RESTARTS = 10
_RESUME_STRIP_ARGS = {"--resume", "--load_run", "--checkpoint"}
_MAX_ITER_ARGS = {"--max_iterations", "--max_iteration"}  # handle typo variants


def _build_cmd(script: str, nproc: int, train_args: list[str]) -> list[str]:
    cmd = [sys.executable]
    if nproc > 1:
        cmd += ["-m", "torch.distributed.run", "--nnodes=1", f"--nproc_per_node={nproc}"]
    cmd += [script] + train_args
    return cmd


def _strip_resume_args(args: list[str]) -> list[str]:
    """Remove --resume, --load_run, --checkpoint and their values."""
    result, skip = [], False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg in _RESUME_STRIP_ARGS:
            skip = True
            continue
        result.append(arg)
    return result


def _parse_checkpoint_iter(checkpoint: str) -> int:
    """Extract iteration number from checkpoint filename (e.g. model_3000.pt -> 3000)."""
    basename = os.path.basename(checkpoint)
    m = re.search(r'(\d+)', basename)
    return int(m.group(1)) if m else 0


def _adjust_max_iterations(args: list[str], completed_iters: int) -> list[str]:
    """Reduce --max_iterations by the number of already-completed iterations."""
    if completed_iters <= 0:
        return args

    result, skip_next = [], False
    found = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        # Handle --max_iterations=10000 style
        for max_arg in _MAX_ITER_ARGS:
            if arg.startswith(f"{max_arg}="):
                old_val = int(arg.split("=", 1)[1])
                new_val = max(old_val - completed_iters, 0)
                result.append(f"{max_arg}={new_val}")
                found = True
                break
        else:
            if arg in _MAX_ITER_ARGS and i + 1 < len(args):
                old_val = int(args[i + 1])
                new_val = max(old_val - completed_iters, 0)
                result.append(arg)
                result.append(str(new_val))
                skip_next = True
                found = True
            else:
                result.append(arg)

    if not found:
        print("[Auto-restart] Warning: --max_iterations not found in args, cannot adjust remaining iterations.")

    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--nproc_per_node", type=int, default=1,
        help="Number of GPUs to use (>1 enables torch.distributed.run).",
    )
    launcher_args, train_args = parser.parse_known_args()
    nproc = launcher_args.nproc_per_node

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")

    # Temporary file used to pass the restart checkpoint path from the runner.
    ckpt_fd, ckpt_file = tempfile.mkstemp(suffix=".txt", prefix="nan_restart_")
    os.close(ckpt_fd)

    env = os.environ.copy()
    env["NAN_RESTART_CHECKPOINT_FILE"] = ckpt_file

    current_args = list(train_args)

    try:
        for attempt in range(MAX_RESTARTS + 1):
            # Clear signal file before each run.
            open(ckpt_file, "w").close()

            cmd = _build_cmd(script, nproc, current_args)
            print(f"\n[Auto-restart] Attempt {attempt + 1}/{MAX_RESTARTS + 1}: {' '.join(cmd)}\n", flush=True)

            result = subprocess.run(cmd, env=env)

            if result.returncode == 0:
                print("[Auto-restart] Training completed successfully.")
                sys.exit(0)

            # Read the NaN signal file.
            with open(ckpt_file) as f:
                restart_ckpt = f.read().strip()

            if not restart_ckpt:
                print(
                    f"[Auto-restart] Training failed (exit {result.returncode}) "
                    "without a NaN restart signal. Stopping."
                )
                sys.exit(result.returncode)

            if attempt >= MAX_RESTARTS:
                print(f"[Auto-restart] Max restarts ({MAX_RESTARTS}) reached. Stopping.")
                sys.exit(1)

            # Build resume args pointing at the earlier checkpoint.
            load_run = os.path.basename(os.path.dirname(restart_ckpt))
            checkpoint = os.path.basename(restart_ckpt)
            completed_iters = _parse_checkpoint_iter(checkpoint)
            print(f"[Auto-restart] Restarting from run='{load_run}' checkpoint='{checkpoint}' "
                  f"(completed iterations: {completed_iters})")

            current_args = _strip_resume_args(current_args)
            current_args = _adjust_max_iterations(current_args, completed_iters)
            current_args += ["--resume", "True", "--load_run", load_run, "--checkpoint", checkpoint]

    finally:
        if os.path.exists(ckpt_file):
            os.unlink(ckpt_file)


if __name__ == "__main__":
    main()

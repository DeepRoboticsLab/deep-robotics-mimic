# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL.

Note: Prefer using train_auto_restart.py which wraps this script
with automatic NaN recovery.

Example:
    python scripts/rsl_rl/train.py --task=Tracking-Flat-DR02_PRO \
        --logger tensorboard --log_project_name logs/ \
        --run_name motion_name --headless --max_iterations=200000
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--registry_name", type=str, default=None, help="The path of the reference motion file (not needed for multi-motion tasks).")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

     # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed
    # --- Load motion file from local path (not needed for multi-motion tasks) ---
    registry_name = args_cli.registry_name
    if registry_name is not None and hasattr(env_cfg.commands.motion, "motion_file"):
        env_cfg.commands.motion.motion_file = registry_name
    
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    # Handle broken symlinks: if any ancestor is a broken symlink, remove it and
    # create a real directory so os.makedirs can succeed.
    _parts = log_dir.split(os.sep)
    for i in range(1, len(_parts) + 1):
        _prefix = os.sep.join(_parts[:i]) or os.sep
        if os.path.islink(_prefix) and not os.path.exists(_prefix):
            os.unlink(_prefix)
            break
    os.makedirs(log_dir, exist_ok=True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    train_cfg = agent_cfg.to_dict()

    # --- Compatibility shim: isaaclab_rl -> rsl_rl 5.x -----------------------
    # isaaclab_rl may serialize actor/critic config under a single "policy" key
    # (older isaaclab_rl) or as separate "actor"/"critic" dicts (newer isaaclab_rl).
    # rsl_rl >= 5.0 expects "actor" and "critic" with "class_name" entries.
    if "policy" in train_cfg and "actor" not in train_cfg:
        # Case 1: old isaaclab_rl -- single "policy" dict with actor_hidden_dims etc.
        pol = train_cfg.pop("policy")
        init_noise_std = pol.get("init_noise_std", 1.0)
        std_type = pol.get("noise_std_type", "scalar")
        train_cfg["actor"] = {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": pol.get("actor_hidden_dims", [256, 256, 256]),
            "activation": pol.get("activation", "elu"),
            "obs_normalization": pol.get("actor_obs_normalization", False),
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.GaussianDistribution",
                "init_std": init_noise_std,
                "std_type": std_type,
            },
        }
        train_cfg["critic"] = {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": pol.get("critic_hidden_dims", [256, 256, 256]),
            "activation": pol.get("activation", "elu"),
            "obs_normalization": pol.get("critic_obs_normalization", False),
        }
        train_cfg.setdefault("obs_groups", {})
    # Case 2: newer isaaclab_rl already provides "actor"/"critic" dicts but may
    # be missing the "class_name" and "distribution_cfg" fields rsl_rl 5.x needs.
    if "actor" in train_cfg and "class_name" not in train_cfg.get("actor", {}):
        actor = train_cfg["actor"]
        actor.setdefault("class_name", "rsl_rl.models.MLPModel")
        # newer isaaclab_rl may use "actor_hidden_dims" inside the actor dict
        if "hidden_dims" not in actor and "actor_hidden_dims" in actor:
            actor["hidden_dims"] = actor.pop("actor_hidden_dims")
        actor.setdefault("hidden_dims", [256, 256, 256])
        actor.setdefault("activation", "elu")
        actor.setdefault("obs_normalization", False)
        if "distribution_cfg" not in actor:
            init_std = actor.pop("init_noise_std", 1.0)
            std_type = actor.pop("noise_std_type", "scalar")
            actor["distribution_cfg"] = {
                "class_name": "rsl_rl.modules.GaussianDistribution",
                "init_std": init_std,
                "std_type": std_type,
            }
    if "critic" in train_cfg and "class_name" not in train_cfg.get("critic", {}):
        critic = train_cfg["critic"]
        critic.setdefault("class_name", "rsl_rl.models.MLPModel")
        if "hidden_dims" not in critic and "critic_hidden_dims" in critic:
            critic["hidden_dims"] = critic.pop("critic_hidden_dims")
        critic.setdefault("hidden_dims", [256, 256, 256])
        critic.setdefault("activation", "elu")
        critic.setdefault("obs_normalization", False)
    train_cfg.setdefault("obs_groups", {})
    # Strip fields that rsl_rl 5.x doesn't accept
    for key in ("init_noise_std", "share_cnn_encoders"):
        train_cfg.get("algorithm", {}).pop(key, None)
    train_cfg.pop("policy", None)  # remove leftover policy key if present
    runner = OnPolicyRunner(
        env, train_cfg, log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # --- Run training ---
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

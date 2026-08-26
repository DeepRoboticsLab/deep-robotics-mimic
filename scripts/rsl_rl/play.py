"""Script to play a checkpoint of an RL agent from RSL-RL.

Example:
    python scripts/rsl_rl/play.py --task=Tracking-Flat-DR02_PRO \
        --motion_file dataset/gmr/jugong.npz \
        --checkpoint_path logs/rsl_rl/DR02_pro_flat/<run>/model_50000.pt \
        --num_envs 2 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint.")
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
import pathlib
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        # --- Load checkpoint from local path ---
        resume_path = args_cli.checkpoint_path
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        motion_path = args_cli.motion_file
        if motion_path is not None and hasattr(env_cfg.commands.motion, "motion_file"):
            env_cfg.commands.motion.motion_file = motion_path

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
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

    # load previously trained model
    runner_cfg = agent_cfg.to_dict()

    # --- Compatibility shim: isaaclab_rl -> rsl_rl 5.x -----------------------
    if "policy" in runner_cfg and "actor" not in runner_cfg:
        pol = runner_cfg.pop("policy")
        init_noise_std = pol.get("init_noise_std", 1.0)
        std_type = pol.get("noise_std_type", "scalar")
        runner_cfg["actor"] = {
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
        runner_cfg["critic"] = {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": pol.get("critic_hidden_dims", [256, 256, 256]),
            "activation": pol.get("activation", "elu"),
            "obs_normalization": pol.get("critic_obs_normalization", False),
        }
        runner_cfg.setdefault("obs_groups", {})
    if "actor" in runner_cfg and "class_name" not in runner_cfg.get("actor", {}):
        actor = runner_cfg["actor"]
        actor.setdefault("class_name", "rsl_rl.models.MLPModel")
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
    if "critic" in runner_cfg and "class_name" not in runner_cfg.get("critic", {}):
        critic = runner_cfg["critic"]
        critic.setdefault("class_name", "rsl_rl.models.MLPModel")
        if "hidden_dims" not in critic and "critic_hidden_dims" in critic:
            critic["hidden_dims"] = critic.pop("critic_hidden_dims")
        critic.setdefault("hidden_dims", [256, 256, 256])
        critic.setdefault("activation", "elu")
        critic.setdefault("obs_normalization", False)
    runner_cfg.setdefault("obs_groups", {})
    for key in ("init_noise_std", "share_cnn_encoders"):
        runner_cfg.get("algorithm", {}).pop(key, None)
    runner_cfg.pop("policy", None)

    ppo_runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)

    # Load checkpoint
    loaded_dict = torch.load(resume_path, weights_only=False, map_location=agent_cfg.device)
    # rsl_rl 5.x uses "actor_state_dict" / "critic_state_dict"; 3.x uses "model_state_dict"
    if "model_state_dict" in loaded_dict:
        ppo_runner.alg.actor.load_state_dict(loaded_dict["model_state_dict"], strict=False)
    else:
        if "actor_state_dict" in loaded_dict:
            ppo_runner.alg.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=False)
        if "critic_state_dict" in loaded_dict:
            ppo_runner.alg.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=False)
    ppo_runner.current_learning_iteration = loaded_dict.get("iter", 0)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    actor_model = getattr(ppo_runner.alg, "policy", getattr(ppo_runner.alg, "actor", None))
    normalizer = getattr(actor_model, "obs_normalizer", getattr(ppo_runner.alg, "obs_normalizer", None))

    # The ONNX exporter expects policy.actor attribute (rsl_rl 3.x convention).
    # In rsl_rl 5.x, the actor IS the model directly. Wrap it so exporter can find .actor.
    class _PolicyWrapper(torch.nn.Module):
        def __init__(self, actor, normalizer):
            super().__init__()
            self.actor = actor.mlp if hasattr(actor, "mlp") else actor
            self.is_recurrent = False
    try:
        wrapped = _PolicyWrapper(actor_model, normalizer)
        export_motion_policy_as_onnx(
            env.unwrapped,
            wrapped,
            normalizer=normalizer,
            path=export_model_dir,
            filename="policy.onnx",
        )
        attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    except Exception as e:
        print(f"[WARN] ONNX export failed (non-fatal): {e}")
    # --- Reset environment and start inference loop ---
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = torch.clamp(policy(obs), min=-100, max=100)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()


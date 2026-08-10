"""Fast ONNX export for DR02_pro policy — no Isaac Sim or motion file required.

Example:
    python scripts/rsl_rl/export_onnx_fast.py \
        --checkpoint_path logs/rsl_rl/DR02_pro_flat/2025-09-03_10-58-16_forward_kick/model_10000.pt \
        --output_name forward_kick.onnx
"""

import argparse
import os

import onnx
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# DR02_pro robot constants (joint order from USD traversal, verified from Isaac Lab)
# ---------------------------------------------------------------------------

DR02_PRO_JOINT_NAMES = [
    "left_hip_y_joint",   "right_hip_y_joint",  "waist_z_joint",
    "left_hip_x_joint",   "right_hip_x_joint",  "waist_x_joint",
    "left_hip_z_joint",   "right_hip_z_joint",  "waist_y_joint",
    "left_knee_joint",    "right_knee_joint",
    "left_shoulder_y_joint",  "right_shoulder_y_joint",
    "left_ankle_y_joint", "right_ankle_y_joint",
    "left_shoulder_x_joint",  "right_shoulder_x_joint",
    "left_ankle_x_joint", "right_ankle_x_joint",
    "left_shoulder_z_joint",  "right_shoulder_z_joint",
    "left_elbow_joint",   "right_elbow_joint",
    "left_wrist_z_joint", "right_wrist_z_joint",
    "left_wrist_y_joint", "right_wrist_y_joint",
    "left_wrist_x_joint", "right_wrist_x_joint",
]

DR02_PRO_BODY_NAMES = [
    "base_link",
    "body",
    "left_shoulder_x_link",  "left_elbow_link",  "left_wrist_x_link",
    "right_shoulder_x_link", "right_elbow_link", "right_wrist_x_link",
    "left_hip_x_link",  "left_knee_link",  "left_ankle_x_link",
    "right_hip_x_link", "right_knee_link", "right_ankle_x_link",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_actor(state_dict: dict) -> nn.Sequential:
    """Reconstruct the actor MLP from checkpoint weights.

    RSL-RL stores actor layers at even indices (0, 2, 4, ...) with ELU
    activations at odd indices.  We detect the depth from the state dict.
    """
    layers: list[nn.Module] = []
    i = 0
    while f"actor.{i}.weight" in state_dict:
        w = state_dict[f"actor.{i}.weight"]
        b = state_dict[f"actor.{i}.bias"]
        linear = nn.Linear(w.shape[1], w.shape[0])
        linear.weight.data.copy_(w)
        linear.bias.data.copy_(b)
        layers.append(linear)
        if f"actor.{i + 2}.weight" in state_dict:  # not the last layer
            layers.append(nn.ELU())
        i += 2
    return nn.Sequential(*layers)


class _NormalizedActor(nn.Module):
    """Actor MLP with empirical normalization baked in."""

    def __init__(self, actor: nn.Sequential, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-2):
        super().__init__()
        self.actor = actor
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._mean) / (self._std + self.eps)
        return self.actor(x)


def _csv(values, decimals: int = 3) -> str:
    fmt = f"{{:.{decimals}f}}"
    return ",".join(fmt.format(v) if isinstance(v, (int, float)) else str(v) for v in values)


def _attach_metadata(onnx_path: str, run_path: str) -> None:
    # Nominal PD gains per joint (from DR02_pro.py, before domain randomization)
    _PD_GAINS = {
        # stiffness, damping
        "*_hip_y_joint":       (300.0, 10.0),
        "*_hip_x_joint":       (300.0, 10.0),
        "*_hip_z_joint":       (300.0, 10.0),
        "*_knee_joint":         (300.0, 10.0),
        "*_ankle_y_joint":      (80.0,  3.0),
        "*_ankle_x_joint":      (30.0,  1.0),
        "waist_z_joint":       (200.0, 10.0),
        "waist_x_joint":       (200.0,  3.0),
        "waist_y_joint":       (200.0,  6.0),
        "*_shoulder_y_joint":  (100.0,  5.0),
        "*_shoulder_x_joint":  (100.0,  5.0),
        "*_shoulder_z_joint":  (100.0,  5.0),
        "*_elbow_joint":       (100.0,  5.0),
        "*_wrist_z_joint":      (80.0,  3.0),
        "*_wrist_y_joint":      (80.0,  3.0),
        "*_wrist_x_joint":      (80.0,  3.0),
    }

    import fnmatch

    def _match_gain(jname: str) -> tuple[float, float]:
        for pattern, gains in _PD_GAINS.items():
            if fnmatch.fnmatch(jname, pattern):
                return gains
        raise ValueError(f"No PD gain entry for joint '{jname}'")

    stiffness = [_match_gain(j)[0] for j in DR02_PRO_JOINT_NAMES]
    damping   = [_match_gain(j)[1] for j in DR02_PRO_JOINT_NAMES]

    n = len(DR02_PRO_JOINT_NAMES)
    metadata = {
        "run_path":           run_path,
        "joint_names":        _csv(DR02_PRO_JOINT_NAMES),
        "joint_stiffness":    _csv([0.0] * n),
        "joint_damping":      _csv([0.0] * n),
        "default_joint_pos":  _csv([0.0] * n),
        "command_names":      "motion",
        "observation_names":  "command,base_link_gravity,base_ang_vel,joint_pos,joint_vel,actions",
        "action_scale":       _csv([0.5] * n),
        "anchor_body_name":   "body",
        "body_names":         _csv(DR02_PRO_BODY_NAMES),
    }

    model = onnx.load(onnx_path)
    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = v
        model.metadata_props.append(entry)
    onnx.save(model, onnx_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export DR02_pro policy to ONNX without Isaac Sim.")
    parser.add_argument("--checkpoint_path", required=True, help="Path to .pt checkpoint file.")
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "onnx_policy"),
        help="Directory to write the .onnx file.",
    )
    parser.add_argument("--output_name", required=True, help="Output filename, e.g. my_policy.onnx")
    parser.add_argument("--run_path", default="none", help="WandB run path to embed as metadata.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- load checkpoint ---
    print(f"[INFO] Loading: {args.checkpoint_path}")
    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)

    # Support both rsl_rl 3.x ("model_state_dict") and 5.x ("actor_state_dict") formats
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "actor_state_dict" in ckpt:
        # Remap rsl_rl 5.x keys to 3.x convention for _build_actor()
        raw_sd = ckpt["actor_state_dict"]
        sd = {}
        for k, v in raw_sd.items():
            new_key = k
            # mlp.0.weight -> actor.0.weight
            if k.startswith("mlp."):
                new_key = "actor." + k[4:]
            # distribution.std_param -> std
            elif k == "distribution.std_param":
                new_key = "std"
            elif k == "distribution.log_std_param":
                new_key = "std"
                v = torch.exp(v)  # convert log_std to std
            # obs_normalizer._mean / _std -> actor_obs_normalizer._mean / _std
            elif k.startswith("obs_normalizer."):
                new_key = "actor_obs_normalizer." + k[len("obs_normalizer."):]
            sd[new_key] = v
        print(f"[INFO] Converted rsl_rl 5.x checkpoint keys to export format")
    else:
        raise KeyError(f"Checkpoint has unknown format. Keys: {list(ckpt.keys())}")

    obs_dim    = sd["actor.0.weight"].shape[1]
    action_dim = sd["std"].shape[0]
    print(f"[INFO] obs_dim={obs_dim}  action_dim={action_dim}")

    # --- build & load actor (with normalizer if present) ---
    actor = _build_actor(sd)
    if "actor_obs_normalizer._mean" in sd:
        mean = sd["actor_obs_normalizer._mean"]
        std = sd["actor_obs_normalizer._std"]
        model = _NormalizedActor(actor, mean, std)
        print("[INFO] Empirical normalizer found and baked into export.")
    else:
        model = actor
        print("[INFO] No normalizer found, exporting raw actor.")
    model.eval()

    # --- export to ONNX ---
    onnx_path = os.path.join(args.output_dir, args.output_name)
    dummy_obs = torch.zeros(1, obs_dim)
    torch.onnx.export(
        model,
        dummy_obs,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )
    print(f"[INFO] ONNX saved: {onnx_path}")

    # --- attach metadata ---
    _attach_metadata(onnx_path, args.run_path)
    print("[INFO] Metadata attached. Done.")


if __name__ == "__main__":
    main()

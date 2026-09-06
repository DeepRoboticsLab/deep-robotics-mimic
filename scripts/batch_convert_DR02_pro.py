"""Batch-convert motion PKL/NPZ files to training-ready FK format.

Input files are picked up by extension:
  - .pkl (gmr retargeting output, only with --retarget_format gmr):
      fps, root_pos (N,3), root_rot (N,4) xyzw, dof_pos (N,29)
  - .npz, interpreted according to --retarget_format:
      deep_retarget: root_trans_offset, root_rot (xyzw), dof (from motion_capture_data_process)
      omniretarget:  qpos = [root_pos(3), root_quat_wxyz(4), joint_dof(29)]
      gmr:           root_pos, root_rot (wxyz), dof_pos

For each source file in the input directory:
  1. Read generalized coordinates at source fps (typically 30).
  2. Interpolate to 50 fps using lerp (positions/dofs) and slerp (quaternions).
  3. Compute velocities at the new dt.
  4. Run batched FK through Isaac Sim with --num_envs parallel robots.
  5. Write training-ready NPZ: fps, joint_pos, joint_vel, body_pos_w, body_quat_w,
     body_lin_vel_w, body_ang_vel_w.

Usage:
    python scripts/batch_convert_DR02_pro.py \
        --input_dir /path/to/source_folder \
        --output_dir dataset/ \
        --num_envs 10000 --output_fps 50 \
        --retarget_format gmr --headless
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Batch FK precompute for DR02_PRO motions.")
parser.add_argument("--input_dir", type=str, required=True, help="Directory containing source PKL/NPZ files.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory for training-ready NPZ output.")
parser.add_argument("--num_envs", type=int, default=10000, help="Parallel environments for batched FK.")
parser.add_argument("--output_fps", type=int, default=50, help="Target FPS for output motions.")
parser.add_argument(
    "--retarget_format", type=str, default="gmr",
    choices=["deep_retarget", "omniretarget", "gmr"],
    help="Source NPZ format: 'deep_retarget' (root_trans_offset/root_rot/dof), "
    "'omniretarget' (qpos), or 'gmr' (root_pos/root_rot/dof_pos; also accepts .pkl inputs)."
)
parser.add_argument("--override", action="store_true", default=False, help="Overwrite existing output files without asking.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Fail fast: .pkl inputs are only supported with the gmr retarget format (before Isaac Sim launch)
if args_cli.retarget_format != "gmr" and glob.glob(os.path.join(args_cli.input_dir, "*.pkl")):
    parser.error(
        f"Found .pkl files in --input_dir but --retarget_format is '{args_cli.retarget_format}'. "
        "PKL inputs are only supported with --retarget_format gmr."
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)

from whole_body_tracking.robots.DR02_pro import DR02_PRO_CYLINDER_CFG

# Robot model joint order (matches MuJoCo XML & URDF traversal, same as the source data)
DR02_PRO_JOINT_NAMES = [
    # waist
    "waist_z_joint", "waist_x_joint", "waist_y_joint",
    # arms
    "left_shoulder_y_joint", "left_shoulder_x_joint", "left_shoulder_z_joint",
    "left_elbow_joint",
    "left_wrist_z_joint", "left_wrist_y_joint", "left_wrist_x_joint",
    "right_shoulder_y_joint", "right_shoulder_x_joint", "right_shoulder_z_joint",
    "right_elbow_joint",
    "right_wrist_z_joint", "right_wrist_y_joint", "right_wrist_x_joint",
    # legs
    "left_hip_y_joint", "left_hip_x_joint", "left_hip_z_joint",
    "left_knee_joint", "left_ankle_y_joint", "left_ankle_x_joint",
    "right_hip_y_joint", "right_hip_x_joint", "right_hip_z_joint",
    "right_knee_joint", "right_ankle_y_joint", "right_ankle_x_joint",
]


@configclass
class FKSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = DR02_PRO_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ---------------------------------------------------------------------------
# Pickle loading (numpy 2.x -> 1.x compat)
# ---------------------------------------------------------------------------

def load_pkl_compat(path: str):
    """Load a pickle that may have been written with numpy>=2.0.

    NumPy 2 renamed ``numpy.core`` to ``numpy._core``, so pickles written
    under numpy 2 fail with ``ModuleNotFoundError: No module named
    'numpy._core'`` on numpy 1.x.  Alias the module paths before
    unpickling; the array pickle payload itself is format-compatible.
    """
    import numpy.core

    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = numpy.core
        sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
        sys.modules["numpy._core.umath"] = numpy.core.umath

    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def interpolate_motion(
    root_trans: torch.Tensor,
    root_rot_wxyz: torch.Tensor,
    dof_positions: torch.Tensor,
    input_fps: int,
    output_fps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Interpolate motion from input_fps to output_fps.

    Returns (root_trans, root_rot_wxyz, dof_positions, num_output_frames).
    """
    if input_fps == output_fps:
        return root_trans, root_rot_wxyz, dof_positions, root_trans.shape[0]

    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    input_frames = root_trans.shape[0]
    duration = (input_frames - 1) * input_dt

    times = torch.arange(0, duration, output_dt, device=device, dtype=torch.float32)
    output_frames = times.shape[0]

    # Compute blend factors
    phase = times / duration
    idx0 = (phase * (input_frames - 1)).floor().long()
    idx1 = torch.minimum(idx0 + 1, torch.tensor(input_frames - 1, device=device))
    blend = phase * (input_frames - 1) - idx0.float()

    # Lerp positions and dofs
    b1 = blend.unsqueeze(1)
    new_trans = root_trans[idx0] * (1 - b1) + root_trans[idx1] * b1
    new_dof = dof_positions[idx0] * (1 - b1) + dof_positions[idx1] * b1

    # Slerp quaternions
    new_rot = torch.zeros(output_frames, 4, device=device, dtype=torch.float32)
    for i in range(output_frames):
        new_rot[i] = quat_slerp(root_rot_wxyz[idx0[i]], root_rot_wxyz[idx1[i]], blend[i])

    return new_trans, new_rot, new_dof, output_frames


def so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """Angular velocities from wxyz quaternion sequence."""
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
    return omega


# ---------------------------------------------------------------------------
# Batched FK
# ---------------------------------------------------------------------------

def run_fk_for_file(
    sim: SimulationContext,
    scene: InteractiveScene,
    input_path: str,
    output_path: str,
    output_fps: int,
):
    """Process one source PKL/NPZ file -> training-ready NPZ via batched FK."""
    robot = scene["robot"]
    device = sim.device
    num_envs = scene.cfg.num_envs

    t_start = time.time()

    # Load source generalized coordinates (PKL is gmr retargeting output)
    if input_path.endswith(".pkl"):
        # PKL: fps, root_pos (N,3), root_rot (N,4) xyzw, dof_pos (N,29)
        src = load_pkl_compat(input_path)
        input_fps = int(src["fps"])
        root_trans = torch.tensor(np.asarray(src["root_pos"], dtype=np.float64).astype(np.float32), device=device)
        root_rot_xyzw = torch.tensor(np.asarray(src["root_rot"], dtype=np.float64).astype(np.float32), device=device)
        dof_positions = torch.tensor(np.asarray(src["dof_pos"], dtype=np.float64).astype(np.float32), device=device)
        # xyzw -> wxyz
        base_rotations = root_rot_xyzw[:, [3, 0, 1, 2]]
    else:
        src = np.load(input_path, allow_pickle=True)
        input_fps = int(float(src["fps"].flat[0]))
        retarget_format = args_cli.retarget_format

        if retarget_format == "omniretarget":
            # OmniRetarget: qpos = [root_pos(3), root_quat_wxyz(4), joint_dof(29)]
            qpos = torch.tensor(np.array(src["qpos"], dtype=np.float64).astype(np.float32), device=device)
            root_trans = qpos[:, :3]
            base_rotations = qpos[:, 3:7]          # already wxyz (MuJoCo convention)
            dof_positions = qpos[:, 7:]            # already in joint order, no remap needed
        elif retarget_format == "deep_retarget":
            # Deep Retarget: separate keys, root_rot in xyzw
            root_trans = torch.tensor(src["root_trans_offset"], dtype=torch.float32, device=device)
            root_rot_xyzw = torch.tensor(
                np.array(src["root_rot"], dtype=np.float64).astype(np.float32), device=device
            )
            dof_positions = torch.tensor(src["dof"], dtype=torch.float32, device=device)
            # xyzw -> wxyz
            base_rotations = root_rot_xyzw[:, [3, 0, 1, 2]]
        elif retarget_format == "gmr":
            # GMR format: root_pos(3), root_rot(4, wxyz), dof_pos(29)
            root_trans = torch.tensor(np.array(src["root_pos"], dtype=np.float64).astype(np.float32), device=device)
            base_rotations = torch.tensor(np.array(src["root_rot"], dtype=np.float64).astype(np.float32), device=device)  # wxyz
            dof_positions = torch.tensor(np.array(src["dof_pos"], dtype=np.float64).astype(np.float32), device=device)
    
    # Interpolate to output fps
    root_trans, base_rotations, dof_positions, num_frames = interpolate_motion(
        root_trans, base_rotations, dof_positions, input_fps, output_fps, device,
    )
    dt = 1.0 / output_fps

    # Compute velocities after interpolation
    base_lin_vels = torch.gradient(root_trans, spacing=dt, dim=0)[0]
    dof_vels = torch.gradient(dof_positions, spacing=dt, dim=0)[0]
    base_ang_vels = so3_derivative(base_rotations, dt)

    num_bodies = robot.num_bodies
    num_joints = robot.num_joints
    robot_joint_indexes = robot.find_joints(DR02_PRO_JOINT_NAMES, preserve_order=True)[0]
    env_origins = scene.env_origins

    # Pre-allocate output buffers on CPU
    all_joint_pos = torch.zeros(num_frames, num_joints, dtype=torch.float32)
    all_joint_vel = torch.zeros(num_frames, num_joints, dtype=torch.float32)
    all_body_pos_w = torch.zeros(num_frames, num_bodies, 3, dtype=torch.float32)
    all_body_quat_w = torch.zeros(num_frames, num_bodies, 4, dtype=torch.float32)
    all_body_lin_vel_w = torch.zeros(num_frames, num_bodies, 3, dtype=torch.float32)
    all_body_ang_vel_w = torch.zeros(num_frames, num_bodies, 3, dtype=torch.float32)

    num_batches = (num_frames + num_envs - 1) // num_envs
    all_env_ids = torch.arange(num_envs, device=device, dtype=torch.long)

    for batch_idx in range(num_batches):
        start = batch_idx * num_envs
        end = min(start + num_envs, num_frames)
        batch_size = end - start
        env_ids = all_env_ids[:batch_size]

        root_states = robot.data.default_root_state[:batch_size].clone()
        root_states[:, :3] = root_trans[start:end]
        root_states[:, :2] += env_origins[:batch_size, :2]
        root_states[:, 3:7] = base_rotations[start:end]
        root_states[:, 7:10] = base_lin_vels[start:end]
        root_states[:, 10:] = base_ang_vels[start:end]
        robot.write_root_state_to_sim(root_states, env_ids=env_ids)

        jpos = robot.data.default_joint_pos[:batch_size].clone()
        jvel = robot.data.default_joint_vel[:batch_size].clone()
        jpos[:, robot_joint_indexes] = dof_positions[start:end]
        jvel[:, robot_joint_indexes] = dof_vels[start:end]
        robot.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids)

        sim.render()
        scene.update(sim.get_physics_dt())

        bp = robot.data.body_pos_w[:batch_size].clone()
        bp -= env_origins[:batch_size, None, :]

        all_joint_pos[start:end] = robot.data.joint_pos[:batch_size].cpu()
        all_joint_vel[start:end] = robot.data.joint_vel[:batch_size].cpu()
        all_body_pos_w[start:end] = bp.cpu()
        all_body_quat_w[start:end] = robot.data.body_quat_w[:batch_size].cpu()
        all_body_lin_vel_w[start:end] = robot.data.body_lin_vel_w[:batch_size].cpu()
        all_body_ang_vel_w[start:end] = robot.data.body_ang_vel_w[:batch_size].cpu()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        fps=np.array([output_fps]),
        joint_pos=all_joint_pos.numpy(),
        joint_vel=all_joint_vel.numpy(),
        body_pos_w=all_body_pos_w.numpy(),
        body_quat_w=all_body_quat_w.numpy(),
        body_lin_vel_w=all_body_lin_vel_w.numpy(),
        body_ang_vel_w=all_body_ang_vel_w.numpy(),
    )

    elapsed = time.time() - t_start
    file_mb = os.path.getsize(output_path) / 1e6
    print(f"    -> {num_frames} frames ({num_batches} batches), {elapsed:.1f}s, {file_mb:.1f} MB")


def main():
    input_dir = os.path.abspath(args_cli.input_dir)
    output_dir = os.path.join(os.path.abspath(args_cli.output_dir), args_cli.retarget_format)
    output_fps = args_cli.output_fps

    input_files = sorted(
        glob.glob(os.path.join(input_dir, "*.npz")) + glob.glob(os.path.join(input_dir, "*.pkl"))
    )
    num_pkl = sum(1 for f in input_files if f.endswith(".pkl"))

    if num_pkl and num_pkl < len(input_files):
        print("[WARN] Mixed .pkl and .npz inputs detected; each file is loaded by its extension.")

    print(f"[INFO] Found {len(input_files)} files ({num_pkl} pkl, {len(input_files) - num_pkl} npz) in {input_dir}")

    # Determine max frames across all files (for scene sizing)
    max_frames = 0
    for f in input_files:
        if f.endswith(".pkl"):
            src = load_pkl_compat(f)
            input_fps_val = int(src["fps"])
            n = int(np.asarray(src["root_pos"]).shape[0])
        else:
            src = np.load(f, allow_pickle=True)
            input_fps_val = int(float(src["fps"].flat[0]))
            if "num_frames" in src:
                n = int(src["num_frames"].flat[0])
            elif "qpos" in src:
                n = src["qpos"].shape[0]
            else:
                n = src[list(src.keys())[0]].shape[0]
        if input_fps_val != output_fps:
            duration = (n - 1) / input_fps_val
            n = int(duration * output_fps)
        max_frames = max(max_frames, n)

    num_envs = min(args_cli.num_envs, max_frames)
    print(f"[INFO] Creating scene with {num_envs} envs (max frames across files: {max_frames})")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / output_fps
    sim = SimulationContext(sim_cfg)

    scene_cfg = FKSceneCfg(num_envs=num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    total_start = time.time()
    skipped = 0
    for i, input_file in enumerate(input_files):
        basename = os.path.basename(input_file)
        output_name = os.path.splitext(basename)[0] + ".npz"
        output_path = os.path.join(output_dir, output_name)

        # If output file already exists, ask whether to overwrite
        if os.path.exists(output_path):
            try:
                with np.load(output_path) as d:
                    if "joint_pos" in d.files:
                        if not args_cli.override:
                            response = input(f"  [{i+1}/{len(input_files)}] File exists: {output_name}\n  Overwrite? [y/N]: ").strip().lower()
                            if response not in ("y", "yes"):
                                print(f"  [{i+1}/{len(input_files)}] SKIP: {basename}")
                                skipped += 1
                                continue
            except Exception:
                pass

        print(f"  [{i+1}/{len(input_files)}] Processing: {basename}")
        run_fk_for_file(sim, scene, input_file, output_path, output_fps)

    total_elapsed = time.time() - total_start
    print(f"\n[DONE] Processed {len(input_files) - skipped}/{len(input_files)} files in {total_elapsed:.1f}s")
    sys.exit(0)


if __name__ == "__main__":
    main()
    simulation_app.close()

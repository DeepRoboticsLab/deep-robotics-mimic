#!/usr/bin/env python3
"""Replay NPZ motion data in MuJoCo viewer.

Reads FK-format NPZ files (from convert_DR02_pro.py) and maps joint
data to MuJoCo qpos for interactive playback via mujoco.viewer.

NPZ data format (keys):
    - fps:            Frame rate (int)
    - joint_pos:      (N, 29) joint positions in IsaacLab BFS order
    - joint_vel:      (N, 29) joint velocities
    - body_pos_w:     (N, 40, 3) body world positions (IsaacLab BFS)
    - body_quat_w:    (N, 40, 4) body world quaternions (wxyz)
    - body_lin_vel_w: (N, 40, 3) linear velocities
    - body_ang_vel_w: (N, 40, 3) angular velocities

Usage:
    python scripts/replay_npz_mujoco.py [npz_file_path]

    Without arguments, enters interactive file selection (with Tab completion).
"""

from __future__ import annotations

import argparse
import glob
import os
import readline
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# Constants
# ============================================================

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MuJoCo model XML path
MODEL_XML_PATH = os.path.join(
    PROJECT_ROOT,
    "source",
    "whole_body_tracking",
    "whole_body_tracking",
    "assets",
    "DR02",
    "DR02_pro.xml",
)

# 29 joint names in NPZ (IsaacLab BFS order, no neck_z/neck_y).
# IsaacLab traverses URDF joints in alphabetical BFS order;
# MuJoCo XML uses DFS order: waist(3) + left_arm(7) + right_arm(7) + neck(2) + left_leg(6) + right_leg(6)
NPZ_JOINT_NAMES = [
    # BFS level 1: hip_y + waist_z
    "left_hip_y_joint", "right_hip_y_joint", "waist_z_joint",
    # BFS level 2: hip_x + waist_x
    "left_hip_x_joint", "right_hip_x_joint", "waist_x_joint",
    # BFS level 3: hip_z + waist_y
    "left_hip_z_joint", "right_hip_z_joint", "waist_y_joint",
    # BFS level 4: knee + shoulder_y
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_y_joint", "right_shoulder_y_joint",
    # BFS level 5: ankle_y + shoulder_x
    "left_ankle_y_joint", "right_ankle_y_joint",
    "left_shoulder_x_joint", "right_shoulder_x_joint",
    # BFS level 6: ankle_x + shoulder_z
    "left_ankle_x_joint", "right_ankle_x_joint",
    "left_shoulder_z_joint", "right_shoulder_z_joint",
    # BFS level 7+: elbow + wrist
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_z_joint", "right_wrist_z_joint",
    "left_wrist_y_joint", "right_wrist_y_joint",
    "left_wrist_x_joint", "right_wrist_x_joint",
]

# 40 body names in NPZ (IsaacLab BFS order).
# MuJoCo XML body order is DFS (waist->arms->neck->legs), different from NPZ.
# Playback sets qpos via joint mapping; body order difference does not affect playback.
# For direct body_pos_w comparison, use this list to build index mapping.
NPZ_BODY_NAMES = [
    "base_link", "left_hip_y_link", "right_hip_y_link", "waist_z_link",
    "left_hip_x_link", "right_hip_x_link", "waist_x_link",
    "left_hip_z_link", "right_hip_z_link", "body",
    "left_knee_link", "right_knee_link",
    "d435_front_link", "d435_head_link", "d435_rear_link",
    "imu_base_link", "imu_head_link",
    "left_shoulder_y_link", "lidar_link", "neck_link",
    "right_shoulder_y_link",
    "left_ankle_y_link", "right_ankle_y_link",
    "left_shoulder_x_link", "head_link", "right_shoulder_x_link",
    "left_ankle_x_link", "right_ankle_x_link",
    "left_shoulder_z_link", "right_shoulder_z_link",
    "left_elbow_link", "right_elbow_link",
    "left_wrist_z_link", "right_wrist_z_link",
    "left_wrist_y_link", "right_wrist_y_link",
    "left_wrist_x_link", "right_wrist_x_link",
    "left_hand_link", "right_hand_link",
]


# ============================================================
# Interactive file selection
# ============================================================

def _complete_path(text: str, state: int) -> str | None:
    """Readline completer: provides Tab path completion for interactive input."""
    matches = glob.glob(text + "*")
    matches = [m + "/" if os.path.isdir(m) else m for m in matches]
    if state < len(matches):
        return matches[state]
    return None


def select_npz_file(default_dir: str = "dataset") -> str:
    """Interactively select an NPZ file with Tab path completion.

    Lists all NPZ files under the default directory and lets the user
    choose by number or enter a custom path.

    Args:
        default_dir: Default search directory (relative to project root).

    Returns:
        Absolute path to the selected NPZ file.
    """
    readline.set_completer(_complete_path)
    readline.parse_and_bind("tab: complete")

    # --- Search for NPZ files ---
    search_dir = os.path.join(PROJECT_ROOT, default_dir)
    npz_files = sorted(glob.glob(os.path.join(search_dir, "**", "*.npz"), recursive=True))

    if npz_files:
        print("\nAvailable NPZ files:")
        for i, f in enumerate(npz_files):
            rel_path = os.path.relpath(f, PROJECT_ROOT)
            print(f"  [{i}] {rel_path}")
        print()

    default_path = os.path.relpath(npz_files[0], PROJECT_ROOT) if npz_files else default_dir
    user_input = input(f"Enter NPZ file path (Enter=default '{default_path}'): ").strip()

    if not user_input:
        user_input = default_path

    # --- Resolve to absolute path ---
    if not os.path.isabs(user_input):
        user_input = os.path.join(PROJECT_ROOT, user_input)

    if not os.path.isfile(user_input):
        print(f"[ERROR] File not found: {user_input}")
        sys.exit(1)

    return user_input


# ============================================================
# Data loading and joint mapping
# ============================================================

class NpzMotionData:
    """Load and manage NPZ motion data with joint mapping.

    Reads joint positions and root body pose from NPZ, and builds
    a mapping from NPZ BFS joint order to MuJoCo DFS qpos addresses.

    NPZ joint order is IsaacLab BFS (alphabetical level-order traversal);
    MuJoCo qpos order is XML DFS (waist->arms->neck->legs).
    The two orders are completely different; remapping via joint name lookup is required.

    Attributes:
        fps: Frame rate.
        num_frames: Total frame count.
        joint_pos: (N, 29) joint positions (BFS order).
        root_pos: (N, 3) root body world positions.
        root_quat_wxyz: (N, 4) root body world quaternions (wxyz).
        qpos_addrs: (29,) mapping from NPZ joint index to MuJoCo qpos address.
    """

    def __init__(self, npz_path: str, model: mujoco.MjModel):
        """Load NPZ data and build joint/body mappings.

        Args:
            npz_path: Path to the NPZ file.
            model: Loaded MuJoCo model for joint mapping.
        """
        data = np.load(npz_path, allow_pickle=True)

        # --- Read basic data ---
        self.fps = float(data["fps"].flat[0])
        self.num_frames = data["joint_pos"].shape[0]
        self.joint_pos = data["joint_pos"].astype(np.float64)

        # --- Root body pose: body_pos_w[:, 0] is base_link (root) world pose ---
        self.root_pos = data["body_pos_w"][:, 0, :].astype(np.float64)
        self.root_quat_wxyz = data["body_quat_w"][:, 0, :].astype(np.float64)

        # --- Build joint mapping: NPZ BFS index -> MuJoCo DFS qpos address ---
        self.qpos_addrs = self._build_joint_mapping(model)

        # --- Build body mapping: NPZ BFS index -> MuJoCo body index ---
        self.body_map = self._build_body_mapping(model)

        print(f"[INFO] Loaded: {os.path.basename(npz_path)}")
        print(f"  Frames: {self.num_frames}, FPS: {self.fps:.1f}")
        print(f"  Duration: {self.num_frames / self.fps:.1f}s")
        print(f"  Joint mapping: {len(NPZ_JOINT_NAMES)} NPZ -> {model.nq - 7} MuJoCo (incl. neck)")
        print(f"  Body mapping: {len(NPZ_BODY_NAMES)} NPZ -> {model.nbody - 1} MuJoCo")

    def _build_joint_mapping(self, model: mujoco.MjModel) -> np.ndarray:
        """Build mapping from NPZ joint index to MuJoCo qpos address.

        NPZ has 29 joints (no neck_z/neck_y); MuJoCo model has 31 hinge
        joints (incl. neck_z/neck_y). Uses joint name lookup to find qpos addresses.

        Returns:
            (29,) array where qpos_addrs[i] is the qpos address for NPZ joint i.
        """
        qpos_addrs = []
        for joint_name in NPZ_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                print(f"[WARNING] Joint '{joint_name}' not found in MuJoCo model, skipping")
                qpos_addrs.append(-1)
            else:
                qpos_addrs.append(model.jnt_qposadr[joint_id])
        return np.array(qpos_addrs, dtype=np.int32)

    def _build_body_mapping(self, model: mujoco.MjModel) -> np.ndarray:
        """Build mapping from NPZ body index to MuJoCo body index.

        NPZ body order is IsaacLab BFS; MuJoCo XML uses DFS.
        Uses body name lookup. MuJoCo body 0 is world; actual bodies start at index 1.

        Returns:
            (40,) array where body_map[i] is the MuJoCo body index for NPZ body i.
        """
        body_map = []
        for body_name in NPZ_BODY_NAMES:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                print(f"[WARNING] Body '{body_name}' not found in MuJoCo model")
            body_map.append(body_id)
        return np.array(body_map, dtype=np.int32)


# ============================================================
# MuJoCo playback
# ============================================================

def replay_motion(npz_path: str, speed: float = 1.0, loop: bool = True) -> None:
    """Replay NPZ motion data in MuJoCo viewer.

    Uses mujoco.viewer.launch_passive for interactive playback.
    Sets qpos frame by frame. Supports play/pause, speed control, and looping.

    Args:
        npz_path: Path to the NPZ file.
        speed: Playback speed multiplier (1.0 = original).
        loop: Whether to loop playback.
    """
    # --- Load model ---
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mujoco.MjData(model)

    # --- Load motion data ---
    motion = NpzMotionData(npz_path, model)

    # --- Disable gravity (pure kinematic playback, no physics simulation) ---
    model.opt.gravity[:] = 0.0

    # --- Initialize qpos to first frame ---
    data.qpos[:3] = motion.root_pos[0]
    data.qpos[3:7] = motion.root_quat_wxyz[0]
    valid_mask = motion.qpos_addrs >= 0
    data.qpos[motion.qpos_addrs[valid_mask]] = motion.joint_pos[0][valid_mask]
    mujoco.mj_forward(model, data)

    # --- Playback parameters ---
    frame_dt = 1.0 / motion.fps
    frame = 0
    playing = True

    print(f"\n[INFO] Starting playback (speed={speed}x, loop={loop})")
    print("  Mouse drag: rotate/pan camera")
    print("  Press Ctrl+C to exit\n")

    # --- Launch viewer and enter playback loop ---
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if playing:
                # Set root body pose for current frame
                f = min(frame, motion.num_frames - 1)
                data.qpos[:3] = motion.root_pos[f]
                data.qpos[3:7] = motion.root_quat_wxyz[f]

                # Set joint angles via qpos address mapping
                npz_joints = motion.joint_pos[f]
                valid_addrs = motion.qpos_addrs[valid_mask]
                data.qpos[valid_addrs] = npz_joints[valid_mask]

                # Forward kinematics (no dynamics simulation)
                mujoco.mj_forward(model, data)

                # Sync to viewer
                viewer.sync()

                # Advance frame
                frame += 1
                if frame >= motion.num_frames:
                    if loop:
                        frame = 0
                    else:
                        playing = False
                        print("[INFO] Playback finished")

            # Wait for next frame
            time.sleep(frame_dt / speed)


# ============================================================
# FK accuracy verification
# ============================================================

def verify_fk(npz_path: str) -> None:
    """Verify FK accuracy by comparing MuJoCo FK output with stored NPZ body poses.

    For each frame: set qpos from NPZ joint_pos + root pose, run mj_forward,
    then compare FK body positions/orientations with stored body_pos_w/body_quat_w.

    Position error: ||fk_pos - npz_pos|| (meters)
    Orientation error: 2 * arccos(|q1 . q2|) (degrees, handles double-cover)

    Args:
        npz_path: Path to the NPZ file.
    """
    # --- Load model and data ---
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    mdata = mujoco.MjData(model)
    model.opt.gravity[:] = 0.0

    motion = NpzMotionData(npz_path, model)
    npz_raw = np.load(npz_path, allow_pickle=True)
    npz_body_pos = npz_raw["body_pos_w"].astype(np.float64)    # (N, 40, 3)
    npz_body_quat = npz_raw["body_quat_w"].astype(np.float64)  # (N, 40, 4) wxyz

    num_frames = motion.num_frames
    num_bodies = len(NPZ_BODY_NAMES)

    # --- Accumulate per-frame errors ---
    pos_errors = np.zeros((num_frames, num_bodies))  # position error per frame per body
    ori_errors = np.zeros((num_frames, num_bodies))  # orientation error per frame per body (degrees)

    for frame in range(num_frames):
        # Set qpos: root pose + joint angles
        mdata.qpos[:3] = motion.root_pos[frame]
        mdata.qpos[3:7] = motion.root_quat_wxyz[frame]
        mdata.qpos[motion.qpos_addrs] = motion.joint_pos[frame]
        mujoco.mj_forward(model, mdata)

        # Compare each body
        for i in range(num_bodies):
            mj_id = motion.body_map[i]
            if mj_id < 0:
                continue

            # Position error
            fk_pos = mdata.xpos[mj_id]
            npz_pos = npz_body_pos[frame, i]
            pos_errors[frame, i] = np.linalg.norm(fk_pos - npz_pos)

            # Orientation error: 2*arccos(|q1.q2|), handles double-cover
            fk_quat = mdata.xquat[mj_id]       # wxyz
            npz_quat = npz_body_quat[frame, i]  # wxyz
            dot = abs(np.dot(fk_quat, npz_quat))
            dot = min(dot, 1.0)  # clamp to prevent floating point overflow
            ori_errors[frame, i] = np.degrees(2.0 * np.arccos(dot))

    # --- Output per-body average errors ---
    avg_pos = pos_errors.mean(axis=0)    # (num_bodies,)
    avg_ori = ori_errors.mean(axis=0)    # (num_bodies,)
    max_pos = pos_errors.max(axis=0)
    max_ori = ori_errors.max(axis=0)

    print(f"\n{'='*80}")
    print(f"FK Accuracy Verification: {os.path.basename(npz_path)}")
    print(f"Frames: {num_frames}, Bodies: {num_bodies}")
    print(f"{'='*80}")
    print(f"{'idx':>4s}  {'body_name':30s}  {'avg_pos(m)':>10s}  {'max_pos(m)':>10s}  "
          f"{'avg_ori(deg)':>12s}  {'max_ori(deg)':>12s}")
    print(f"{'-'*4}  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*12}")

    for i in range(num_bodies):
        name = NPZ_BODY_NAMES[i]
        print(f"{i:4d}  {name:30s}  {avg_pos[i]:10.6f}  {max_pos[i]:10.6f}  "
              f"{avg_ori[i]:12.6f}  {max_ori[i]:12.6f}")

    # --- Summary statistics ---
    print(f"{'-'*80}")
    print(f"{'ALL':>4s}  {'(overall)':30s}  {avg_pos.mean():10.6f}  {max_pos.max():10.6f}  "
          f"{avg_ori.mean():12.6f}  {max_ori.max():12.6f}")
    print()


# ============================================================
# Main entry point
# ============================================================

def main() -> None:
    """Parse arguments and launch playback or verification."""
    parser = argparse.ArgumentParser(
        description="Replay NPZ motion data in MuJoCo",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "npz_file",
        nargs="?",
        default=None,
        help="Path to NPZ file (interactive selection if omitted)",
    )
    parser.add_argument(
        "--speed", "-s",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Disable loop playback",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verification mode: compute FK vs NPZ errors per frame (no viewer)",
    )
    args = parser.parse_args()

    # --- Determine NPZ file path ---
    if args.npz_file:
        npz_path = os.path.abspath(args.npz_file)
        if not os.path.isfile(npz_path):
            print(f"[ERROR] File not found: {npz_path}")
            sys.exit(1)
    else:
        npz_path = select_npz_file()

    # --- Execute based on mode ---
    if args.verify:
        verify_fk(npz_path)
    else:
        try:
            replay_motion(npz_path, speed=args.speed, loop=not args.no_loop)
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user, exiting")


if __name__ == "__main__":
    main()

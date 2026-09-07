#!/usr/bin/env python3
"""Replay NPZ/PKL motion data in MuJoCo viewer.

Reads FK-format NPZ files (from convert_DR02_pro.py) or GMR retargeted PKL
files (from general_motion_retargeting), and maps joint data to MuJoCo qpos
for interactive playback via mujoco.viewer.

Visualization is synced with scripts/vis_robot_motion.py (RobotMotionViewer):
    - Checkered ground plane and a shadow-casting world light are injected
      into the scene (the bare DR02_pro.xml has neither).
    - World lights keep a fixed offset from the moving robot root, so the
      robot stays lit during walking motions.
    - A playback progress bar is drawn in the viewer (track + fill + time
      label), following the robot.
    - Space key toggles pause/resume.

NPZ data format (keys):
    - fps:            Frame rate (int)
    - joint_pos:      (N, 29) joint positions in IsaacLab BFS order
    - joint_vel:      (N, 29) joint velocities
    - body_pos_w:     (N, 40, 3) body world positions (IsaacLab BFS)
    - body_quat_w:    (N, 40, 4) body world quaternions (wxyz)
    - body_lin_vel_w: (N, 40, 3) linear velocities
    - body_ang_vel_w: (N, 40, 3) angular velocities

PKL data format (keys):
    - fps:            Frame rate
    - root_pos:       (N, 3) root body world positions
    - root_rot:       (N, 4) root body world quaternions (xyzw)
    - dof_pos:        (N, 29) joint positions (MuJoCo DFS order, no neck)
    - joint_names:    (29,) joint names (DFS order)

Usage:
    python scripts/replay_npz_mujoco.py dataset/gmr/jugong.npz
    python scripts/replay_npz_mujoco.py dataset/raw/pkl/bow.pkl
    python scripts/replay_npz_mujoco.py dataset/gmr/jugong.npz --speed 0.5 --no-loop
    python scripts/replay_npz_mujoco.py dataset/gmr/jugong.npz --verify   # FK accuracy verification (NPZ only)

    Without arguments, enters interactive file selection (with Tab completion).
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import readline
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

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

# Scene elements injected into the bare robot XML, matching the
# general_motion_retargeting DR02_pro.xml scene (floor + world light).
SCENE_ASSET_XML = (
    '<texture type="2d" name="groundplane" builtin="checker" mark="edge" '
    'rgb1="1 1 1" rgb2="1 1 1" markrgb="0 0 0" width="300" height="300"/>'
    '<material name="groundplane" texture="groundplane" texuniform="true" '
    'texrepeat="5 5" reflectance="0"/>'
)
SCENE_WORLDBODY_XML = (
    '<geom name="floor" size="0 0 0.01" type="plane" material="groundplane" '
    'contype="1" conaffinity="0" priority="1" friction="0.6" condim="3"/>'
    '<light diffuse=".5 .5 .5" pos="-3 -3 5" dir="3 3 -5" castshadow="true"/>'
)

# Viewer camera settings synced with RobotMotionViewer (DR02_pro).
ROBOT_BASE_BODY = "base_link"
VIEWER_CAM_DISTANCE = 2.5
VIEWER_CAM_ELEVATION = -10
VIEWER_CAM_AZIMUTH = 135

# In-viewer progress bar layout (drawn with user scene geoms).
BAR_HALF_LENGTH = 0.4   # half length of the track (m), bar lies along +x
BAR_THICKNESS = 0.015   # half height / half depth of the bar (m)
BAR_HEIGHT = 2.05       # bar height above ground, follows the robot xy
BAR_TRACK_RGBA = [0.25, 0.27, 0.31, 1.0]
BAR_FILL_RGBA = [0.20, 0.80, 0.40, 1.0]

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


def select_motion_file(default_dir: str = "dataset") -> str:
    """Interactively select an NPZ/PKL motion file with Tab path completion.

    Lists all NPZ and PKL files under the default directory and lets the user
    choose by number or enter a custom path.

    Args:
        default_dir: Default search directory (relative to project root).

    Returns:
        Absolute path to the selected motion file.
    """
    readline.set_completer(_complete_path)
    readline.parse_and_bind("tab: complete")

    # --- Search for motion files ---
    search_dir = os.path.join(PROJECT_ROOT, default_dir)
    motion_files = sorted(
        glob.glob(os.path.join(search_dir, "**", "*.npz"), recursive=True)
        + glob.glob(os.path.join(search_dir, "**", "*.pkl"), recursive=True)
    )

    if motion_files:
        print("\nAvailable motion files:")
        for i, f in enumerate(motion_files):
            rel_path = os.path.relpath(f, PROJECT_ROOT)
            print(f"  [{i}] {rel_path}")
        print()

    default_path = os.path.relpath(motion_files[0], PROJECT_ROOT) if motion_files else default_dir
    user_input = input(f"Enter motion file path (Enter=default '{default_path}'): ").strip()

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
# Scene model loading (floor + light injection)
# ============================================================

def load_scene_model() -> mujoco.MjModel:
    """Load the DR02_pro model with ground plane and world light injected.

    The mimic repo's DR02_pro.xml contains only the robot (no floor, no
    light). This injects the same scene elements used by
    general_motion_retargeting's DR02_pro.xml: a checkered ground plane and
    a shadow-casting world light.

    The patched XML is written to a temporary file next to the original so
    that the relative meshdir="meshes/" still resolves.

    Returns:
        The compiled MuJoCo model with floor and light.
    """
    tree = ET.parse(MODEL_XML_PATH)
    root = tree.getroot()

    asset = root.find("asset")
    for elem in ET.fromstring(f"<asset>{SCENE_ASSET_XML}</asset>"):
        asset.append(elem)

    worldbody = root.find("worldbody")
    for i, elem in enumerate(ET.fromstring(f"<worldbody>{SCENE_WORLDBODY_XML}</worldbody>")):
        worldbody.insert(i, elem)

    # --- Write patched XML next to the original (meshdir is relative) ---
    xml_dir = os.path.dirname(MODEL_XML_PATH)
    fd, tmp_path = tempfile.mkstemp(suffix=".xml", dir=xml_dir, prefix="DR02_pro_scene_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(ET.tostring(root, xml_declaration=True, encoding="utf-8"))
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.remove(tmp_path)

    return model


# ============================================================
# Data loading and joint mapping
# ============================================================

class MotionData:
    """Load and manage NPZ/PKL motion data with joint mapping.

    Reads joint positions and root body pose from the motion file, and builds
    a mapping from the file's joint order to MuJoCo DFS qpos addresses.

    NPZ joint order is IsaacLab BFS (alphabetical level-order traversal);
    PKL joint order is MuJoCo DFS (joint_names stored in the file).
    MuJoCo qpos order is XML DFS (waist->arms->neck->legs).
    Remapping via joint name lookup is required for both formats.

    Attributes:
        is_pkl: Whether the source file is a PKL (GMR retarget output).
        fps: Frame rate.
        num_frames: Total frame count.
        joint_names: Joint names in file order.
        joint_pos: (N, 29) joint positions (file order).
        root_pos: (N, 3) root body world positions.
        root_quat_wxyz: (N, 4) root body world quaternions (wxyz).
        qpos_addrs: (29,) mapping from file joint index to MuJoCo qpos address.
        body_map: (40,) NPZ body index -> MuJoCo body index (NPZ only).
    """

    def __init__(self, motion_path: str, model: mujoco.MjModel):
        """Load motion data and build joint/body mappings.

        Args:
            motion_path: Path to the NPZ or PKL motion file.
            model: Loaded MuJoCo model for joint mapping.
        """
        self.is_pkl = motion_path.lower().endswith(".pkl")

        if self.is_pkl:
            self._load_pkl(motion_path)
        else:
            self._load_npz(motion_path)

        # --- Build joint mapping: file index -> MuJoCo DFS qpos address ---
        self.qpos_addrs = self._build_joint_mapping(model)

        # --- Build body mapping: NPZ BFS index -> MuJoCo body index ---
        self.body_map = self._build_body_mapping(model) if not self.is_pkl else None

        fmt = "PKL (GMR retarget)" if self.is_pkl else "NPZ (FK format)"
        print(f"[INFO] Loaded {fmt}: {os.path.basename(motion_path)}")
        print(f"  Frames: {self.num_frames}, FPS: {self.fps:.1f}")
        print(f"  Duration: {self.num_frames / self.fps:.1f}s")
        print(f"  Joint mapping: {len(self.joint_names)} motion -> {model.nq - 7} MuJoCo (incl. neck)")
        if self.body_map is not None:
            print(f"  Body mapping: {len(NPZ_BODY_NAMES)} NPZ -> {model.nbody - 1} MuJoCo")

    def _load_pkl(self, pkl_path: str) -> None:
        """Load GMR retargeted PKL data (same format as load_robot_motion)."""
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)

        if "joint_names" not in raw:
            raise KeyError(
                "PKL file has no 'joint_names' key; cannot build joint mapping. "
                "Expected GMR retarget output (fps/root_pos/root_rot/dof_pos)."
            )

        self.fps = float(raw["fps"])
        self.joint_names = list(raw["joint_names"])
        self.joint_pos = np.asarray(raw["dof_pos"], dtype=np.float64)
        self.root_pos = np.asarray(raw["root_pos"], dtype=np.float64)
        # PKL stores xyzw quaternions; MuJoCo qpos expects wxyz.
        root_rot_xyzw = np.asarray(raw["root_rot"], dtype=np.float64)
        self.root_quat_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
        self.num_frames = self.joint_pos.shape[0]

    def _load_npz(self, npz_path: str) -> None:
        """Load FK-format NPZ data (from convert_DR02_pro.py)."""
        data = np.load(npz_path, allow_pickle=True)

        self.fps = float(data["fps"].flat[0])
        self.joint_names = list(NPZ_JOINT_NAMES)
        self.num_frames = data["joint_pos"].shape[0]
        self.joint_pos = data["joint_pos"].astype(np.float64)

        # --- Root body pose: body_pos_w[:, 0] is base_link (root) world pose ---
        self.root_pos = data["body_pos_w"][:, 0, :].astype(np.float64)
        self.root_quat_wxyz = data["body_quat_w"][:, 0, :].astype(np.float64)

    def _build_joint_mapping(self, model: mujoco.MjModel) -> np.ndarray:
        """Build mapping from file joint index to MuJoCo qpos address.

        Motion files have 29 joints (no neck_z/neck_y); MuJoCo model has 31
        hinge joints (incl. neck_z/neck_y). Uses joint name lookup to find
        qpos addresses.

        Returns:
            (29,) array where qpos_addrs[i] is the qpos address for joint i.
        """
        qpos_addrs = []
        for joint_name in self.joint_names:
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
# Viewer scene drawing (progress bar)
# ============================================================

def draw_progress_bar(viewer, frame: int, num_frames: int, fps: float,
                      root_pos: np.ndarray) -> None:
    """Draw a playback progress bar into the viewer user scene.

    The bar floats above the robot (following its xy position) and consists
    of a gray track plus a green fill that grows with playback progress.
    The fill geom carries a label with frame counter and elapsed time.

    Args:
        viewer: Active mujoco.viewer handle.
        frame: Current frame index (0-based).
        num_frames: Total frame count.
        fps: Frame rate.
        root_pos: (3,) current root position, the bar follows its xy.
    """
    viewer.user_scn.ngeom = 0

    progress = frame / max(num_frames - 1, 1)
    if progress < 0.0:
        progress = 0.0
    elif progress > 1.0:
        progress = 1.0

    track_center = np.array([root_pos[0], root_pos[1], BAR_HEIGHT])
    fill_half = max(progress * BAR_HALF_LENGTH, 1e-6)
    fill_center = track_center + np.array([fill_half - BAR_HALF_LENGTH, 0.0, 0.0])

    # --- Track (gray background bar) ---
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[BAR_HALF_LENGTH, BAR_THICKNESS, BAR_THICKNESS],
        pos=track_center,
        mat=np.eye(3).ravel(),
        rgba=BAR_TRACK_RGBA,
    )
    viewer.user_scn.ngeom += 1

    # --- Fill (green progress portion, left-aligned) ---
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    cur_t = frame / fps
    total_t = num_frames / fps
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[fill_half, BAR_THICKNESS, BAR_THICKNESS],
        pos=fill_center,
        mat=np.eye(3).ravel(),
        rgba=BAR_FILL_RGBA,
    )
    geom.label = f"{frame + 1}/{num_frames}  {cur_t:.1f}s/{total_t:.1f}s"
    viewer.user_scn.ngeom += 1


# ============================================================
# MuJoCo playback
# ============================================================

def replay_motion(motion_path: str, speed: float = 1.0, loop: bool = True,
                  camera_follow: bool = False) -> None:
    """Replay NPZ/PKL motion data in MuJoCo viewer.

    Uses mujoco.viewer.launch_passive for interactive playback.
    Sets qpos frame by frame. Supports pause/resume (Space), speed control,
    looping, ground/light scene, light following, and a progress bar.

    Args:
        motion_path: Path to the NPZ or PKL motion file.
        speed: Playback speed multiplier (1.0 = original).
        loop: Whether to loop playback.
        camera_follow: Whether the camera tracks the robot base (like
            RobotMotionViewer's follow_camera).
    """
    # --- Load model with floor + light scene ---
    model = load_scene_model()
    data = mujoco.MjData(model)

    # --- Load motion data ---
    motion = MotionData(motion_path, model)

    # --- Disable gravity (pure kinematic playback, no physics simulation) ---
    model.opt.gravity[:] = 0.0

    # --- World light following (synced with RobotMotionViewer):
    # keep world lights at their original offset from the moving robot root.
    world_light_ids = np.flatnonzero(model.light_bodyid == 0)
    world_light_offsets = (
        model.light_pos[world_light_ids].copy() - model.qpos0[:3]
    )

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROBOT_BASE_BODY)

    # --- Initialize qpos to first frame ---
    data.qpos[:3] = motion.root_pos[0]
    data.qpos[3:7] = motion.root_quat_wxyz[0]
    valid_mask = motion.qpos_addrs >= 0
    data.qpos[motion.qpos_addrs[valid_mask]] = motion.joint_pos[0][valid_mask]
    mujoco.mj_forward(model, data)

    # --- Playback state (mutated by the keyboard callback) ---
    state = {"paused": False}

    def key_callback(keycode):
        if keycode == ord(" "):  # Space: pause/resume (same as vis_robot_motion.py)
            state["paused"] = not state["paused"]

    # --- Playback parameters ---
    frame_dt = 1.0 / motion.fps
    frame = 0
    playing = True

    print(f"\n[INFO] Starting playback (speed={speed}x, loop={loop})")
    print("  Mouse drag: rotate/pan camera")
    print("  Space: pause/resume")
    print("  Press Ctrl+C to exit\n")

    # --- Launch viewer and enter playback loop ---
    with mujoco.viewer.launch_passive(
        model, data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        # Frame the actual first pose, which may be far from the model origin.
        # Exclude the floor and include each robot geom's bounding radius so
        # that the feet, head, and outstretched hands all fit with some margin.
        robot_geoms = model.geom_bodyid != 0
        centers = data.geom_xpos[robot_geoms]
        radii = model.geom_rbound[robot_geoms]
        lower = np.min(centers - radii[:, None], axis=0)
        upper = np.max(centers + radii[:, None], axis=0)
        lookat = (lower + upper) / 2
        radius = np.max(np.linalg.norm(centers - lookat, axis=1) + radii)
        distance = max(
            VIEWER_CAM_DISTANCE,
            1.15 * radius / np.sin(np.deg2rad(model.vis.global_.fovy / 2)),
        )
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = lookat
            viewer.cam.distance = distance
            viewer.cam.azimuth = VIEWER_CAM_AZIMUTH
            viewer.cam.elevation = VIEWER_CAM_ELEVATION
        viewer.sync()

        while viewer.is_running():
            if playing and not state["paused"]:
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

                # Advance frame
                frame += 1
                if frame >= motion.num_frames:
                    if loop:
                        frame = 0
                    else:
                        playing = False
                        frame = motion.num_frames - 1
                        print("[INFO] Playback finished")

            # --- Keep world lights at a fixed offset from the robot root ---
            model.light_pos[world_light_ids] = (
                world_light_offsets + data.qpos[:3]
            )

            # --- Camera following (synced with RobotMotionViewer) ---
            if camera_follow and base_body_id >= 0:
                viewer.cam.lookat = data.xpos[base_body_id]
                viewer.cam.distance = VIEWER_CAM_DISTANCE
                viewer.cam.elevation = VIEWER_CAM_ELEVATION

            # --- Draw progress bar for the frame currently on screen ---
            draw_progress_bar(
                viewer, min(frame, motion.num_frames - 1),
                motion.num_frames, motion.fps, data.qpos[:3],
            )

            # Sync to viewer
            viewer.sync()

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
    model = load_scene_model()
    mdata = mujoco.MjData(model)
    model.opt.gravity[:] = 0.0

    motion = MotionData(npz_path, model)
    npz_raw = np.load(npz_path, allow_pickle=True)
    npz_body_pos = npz_raw["body_pos_w"].astype(np.float64)    # (N, 40, 3)
    npz_body_quat = npz_raw["body_quat_w"].astype(np.float64)  # (N, 40, 4) wxyz

    num_frames = motion.num_frames
    num_bodies = len(NPZ_BODY_NAMES)
    assert motion.body_map is not None  # NPZ input always builds the body mapping

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
        description="Replay NPZ/PKL motion data in MuJoCo",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "motion_file",
        nargs="?",
        default=None,
        help="Path to NPZ or PKL motion file (interactive selection if omitted)",
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
        "--camera-follow",
        action="store_true",
        help="Camera tracks the robot base (like vis_robot_motion.py follow mode)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verification mode: compute FK vs NPZ errors per frame (no viewer, NPZ only)",
    )
    args = parser.parse_args()

    # --- Determine motion file path ---
    if args.motion_file:
        motion_path = os.path.abspath(args.motion_file)
        if not os.path.isfile(motion_path):
            print(f"[ERROR] File not found: {motion_path}")
            sys.exit(1)
    else:
        motion_path = select_motion_file()

    # --- Verification mode requires an NPZ (PKL has no FK body poses) ---
    if args.verify:
        if motion_path.lower().endswith(".pkl"):
            print("[ERROR] --verify is only supported for NPZ files "
                  "(PKL has no stored FK body poses)")
            sys.exit(1)
        verify_fk(motion_path)
        return

    # --- Playback mode ---
    try:
        replay_motion(motion_path, speed=args.speed, loop=not args.no_loop,
                      camera_follow=args.camera_follow)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user, exiting")


if __name__ == "__main__":
    main()

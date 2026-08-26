"""Replay a folder of precomputed FK .npz files with dual visualization.

Left side:  DR02_pro robot driven by joint angles — Isaac Sim computes FK on the fly.
Right side: Colored sphere markers at precomputed FK body positions (no robot mesh).

Usage (folder of motions):
    python scripts/replay_merged.py --folder dataset/gmr/ --headless

Legacy usage (single file, old format):
    python scripts/replay_merged.py \
        --file <file>.npz --fk_file <fk_file>.npz --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import glob
import os
import time

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay npz motions (dual view).")
parser.add_argument("--folder", type=str, default=None, help="Path to folder of precomputed FK .npz files.")
parser.add_argument("--file", type=str, default=None, help="(Legacy) Path to merged .npz (joint angles).")
parser.add_argument("--fk_file", type=str, default=None, help="(Legacy) Path to merged_fk.npz (precomputed FK).")
parser.add_argument(
    "--retarget_format", type=str, default="deep_retarget",
    choices=["deep_retarget", "omniretarget"],
    help="Source NPZ format for --file mode: 'deep_retarget' or 'omniretarget'."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.folder is None and args_cli.file is None:
    parser.error("Provide either --folder or --file.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.ui as ui

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from whole_body_tracking.robots.DR02_pro import DR02_PRO_CYLINDER_CFG


def hide_default_panels():
    for name in [
        "Content", "Content Browser", "Console", "Console Window",
        "Property", "Property Window", "Stage", "Stage Window",
        "Layer", "Layer Window", "Render Settings",
        "Flow", "Simulation Settings", "Semantics Schema Editor",
    ]:
        win = ui.Workspace.get_window(name)
        if win is not None:
            win.visible = False


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

CAMERA_PRESETS = {
    "3/4 View":  ([5.0, 5.0, 2.5], [0.0, 0.0, 0.8]),
    "Front":     ([6.0, 0.0, 1.5], [0.0, 0.0, 0.8]),
    "Right":     ([0.0, 6.0, 1.5], [0.0, 0.0, 0.8]),
    "Back":      ([-6.0, 0.0, 1.5], [0.0, 0.0, 0.8]),
    "Left":      ([0.0, -6.0, 1.5], [0.0, 0.0, 0.8]),
    "Top":       ([0.0, 0.1, 7.0], [0.0, 0.0, 0.0]),
}


class CameraController:
    def __init__(self, sim: SimulationContext):
        self._sim = sim
        self.tracking = True
        self._eye_offset = np.array(CAMERA_PRESETS["3/4 View"][0])
        self._lookat_offset = np.array(CAMERA_PRESETS["3/4 View"][1])

    def set_preset(self, name: str):
        self._eye_offset = np.array(CAMERA_PRESETS[name][0])
        self._lookat_offset = np.array(CAMERA_PRESETS[name][1])
        self.tracking = True

    def update(self, target: np.ndarray):
        if not self.tracking:
            return
        self._sim.set_camera_view(eye=target + self._eye_offset, target=target + self._lookat_offset)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

DR02_PRO_JOINT_NAMES = [
    "left_hip_y_joint", "left_hip_x_joint", "left_hip_z_joint",
    "left_knee_joint", "left_ankle_y_joint", "left_ankle_x_joint",
    "right_hip_y_joint", "right_hip_x_joint", "right_hip_z_joint",
    "right_knee_joint", "right_ankle_y_joint", "right_ankle_x_joint",
    "waist_z_joint", "waist_x_joint", "waist_y_joint",
    "left_shoulder_y_joint", "left_shoulder_x_joint", "left_shoulder_z_joint",
    "left_elbow_joint", "left_wrist_z_joint", "left_wrist_y_joint", "left_wrist_x_joint",
    "right_shoulder_y_joint", "right_shoulder_x_joint", "right_shoulder_z_joint",
    "right_elbow_joint", "right_wrist_z_joint", "right_wrist_y_joint", "right_wrist_x_joint",
]

# OmniRetarget DR02_pro.xml uses waist-arms-legs order; DR02_PRO_JOINT_NAMES uses legs-waist-arms.
# Maps omni qpos[7+i] -> DR02_PRO_JOINT_NAMES[OMNI_TO_DEEP[i]].
OMNI_TO_DEEP = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Root body index in the body arrays
ROOT_BODY_IDX = 0

# Lateral offset to place FK keypoints to the right of the robot
FK_X_OFFSET = 3.0


class FkNpzMotion:
    """Loads a single precomputed FK .npz file (new format from batch_convert.py).

    Keys: fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w
    """

    def __init__(self, file_path: str, device: str = "cpu"):
        data = np.load(file_path, allow_pickle=True)
        self.name = os.path.splitext(os.path.basename(file_path))[0]
        self.fps = float(data["fps"][0])
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self.body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self.body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self.num_frames = self.joint_pos.shape[0]
        self.num_bodies = self.body_pos_w.shape[1]

        # Root state extracted from body arrays (body 0)
        # body_quat_w is already wxyz
        self.root_pos = self.body_pos_w[:, ROOT_BODY_IDX, :]          # [F, 3]
        self.root_quat_wxyz = self.body_quat_w[:, ROOT_BODY_IDX, :]   # [F, 4]

    def __repr__(self):
        return f"FkNpzMotion({self.name}: {self.num_frames} frames @ {self.fps} fps, {self.num_bodies} bodies)"


class LegacyMergedNpzMotion:
    """Loads old-format merged .npz (dof, root_trans_offset, root_rot)."""

    def __init__(self, file_path: str, device: str = "cpu"):
        data = np.load(file_path, allow_pickle=True)
        self.name = os.path.splitext(os.path.basename(file_path))[0]
        self.fps = float(data["fps"][0])
        self.num_frames = int(data["num_frames"][0])
        self.joint_pos = torch.tensor(data["dof"], dtype=torch.float32, device=device)
        self.root_pos = torch.tensor(data["root_trans_offset"], dtype=torch.float32, device=device)
        root_rot_xyzw = torch.tensor(data["root_rot"], dtype=torch.float32, device=device)
        self.root_quat_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
        self.num_bodies = 0
        self.body_pos_w = None

        dt = 1.0 / self.fps
        self.joint_vel = torch.zeros_like(self.joint_pos)
        self.joint_vel[1:-1] = (self.joint_pos[2:] - self.joint_pos[:-2]) / (2.0 * dt)
        self.joint_vel[0] = (self.joint_pos[1] - self.joint_pos[0]) / dt
        self.joint_vel[-1] = (self.joint_pos[-1] - self.joint_pos[-2]) / dt


class OmniRetargetNpzMotion:
    """Loads an OmniRetarget .npz (qpos = [root_pos(3), root_quat_wxyz(4), dof(23)])."""

    def __init__(self, file_path: str, device: str = "cpu"):
        data = np.load(file_path, allow_pickle=True)
        self.name = os.path.splitext(os.path.basename(file_path))[0]
        self.fps = float(data["fps"])
        qpos = torch.tensor(np.array(data["qpos"], dtype=np.float64).astype(np.float32), device=device)
        self.num_frames = qpos.shape[0]

        self.root_pos = qpos[:, :3]
        self.root_quat_wxyz = qpos[:, 3:7]             # already wxyz (MuJoCo convention)
        omni_dof = qpos[:, 7:]                          # 23 joints in omniretarget order
        # Remap from omniretarget order (waist-arms-legs) to DR02_PRO_JOINT_NAMES (legs-waist-arms)
        self.joint_pos = omni_dof[:, OMNI_TO_DEEP]

        self.num_bodies = 0
        self.body_pos_w = None

        dt = 1.0 / self.fps
        self.joint_vel = torch.zeros_like(self.joint_pos)
        self.joint_vel[1:-1] = (self.joint_pos[2:] - self.joint_pos[:-2]) / (2.0 * dt)
        self.joint_vel[0] = (self.joint_pos[1] - self.joint_pos[0]) / dt
        self.joint_vel[-1] = (self.joint_pos[-1] - self.joint_pos[-2]) / dt


class LegacyFkNpzMotion:
    """Loads old-format FK .npz (body_pos_w only)."""

    def __init__(self, file_path: str, device: str = "cpu"):
        data = np.load(file_path, allow_pickle=True)
        self.body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self.num_frames = self.body_pos_w.shape[0]
        self.num_bodies = self.body_pos_w.shape[1]


# ---------------------------------------------------------------------------
# Playback UI
# ---------------------------------------------------------------------------

class ReplayUI:
    def __init__(self, motion, cam_ctrl: CameraController, motion_names=None):
        self._cam = cam_ctrl
        self._motion_names = motion_names or []
        self._current_motion_idx = 0

        self.playing = True
        self.current_frame = 0
        self.playback_speed = 1.0
        self._wall_time = time.perf_counter()
        self._playback_time = 0.0
        self._updating_slider = False
        self._motion_switch_requested = None

        self._set_motion(motion)
        self._build_ui()
        self._dock_bottom()

    def _set_motion(self, motion):
        self._fps = motion.fps
        self._total_frames = motion.num_frames
        self._duration = motion.num_frames / motion.fps
        self.current_frame = 0
        self._playback_time = 0.0

    def update_motion(self, motion, idx):
        self._current_motion_idx = idx
        self._set_motion(motion)
        if hasattr(self, "_slider"):
            self._slider.max = self._total_frames - 1
            self._updating_slider = True
            self._slider.model.set_value(0)
            self._updating_slider = False
        if hasattr(self, "_motion_label"):
            name = self._motion_names[idx] if idx < len(self._motion_names) else "?"
            self._motion_label.text = f"[{idx + 1}/{len(self._motion_names)}] {name}"

    @property
    def motion_switch_requested(self):
        val = self._motion_switch_requested
        self._motion_switch_requested = None
        return val

    def _dock_bottom(self):
        viewport = ui.Workspace.get_window("Viewport")
        if viewport is not None:
            self._window.dock_in(viewport, ui.DockPosition.BOTTOM, 0.18)

    def _build_ui(self):
        self._window = ui.Window("Playback", width=700, height=140, flags=ui.WINDOW_FLAGS_NO_COLLAPSE)
        with self._window.frame:
            with ui.VStack(spacing=4):
                # Row 1: playback controls
                with ui.HStack(height=24, spacing=4):
                    self._play_btn = ui.Button("Pause", width=50, clicked_fn=self._toggle_play)
                    self._slider = ui.IntSlider(min=0, max=self._total_frames - 1)
                    self._slider.model.add_value_changed_fn(self._on_slider_changed)
                    self._time_label = ui.Label(
                        f"0.00 / {self._duration:.2f}s", width=130, alignment=ui.Alignment.RIGHT,
                    )
                # Row 2: speed + frame
                with ui.HStack(height=24, spacing=4):
                    ui.Label("Speed:", width=45)
                    for spd in (0.25, 0.5, 1.0, 2.0, 4.0):
                        ui.Button(f"{spd:g}x", width=40, clicked_fn=lambda s=spd: self._set_speed(s))
                    self._speed_label = ui.Label("1x", width=40)
                    ui.Spacer()
                    self._frame_label = ui.Label(
                        f"Frame 0 / {self._total_frames - 1}", width=160, alignment=ui.Alignment.RIGHT,
                    )
                # Row 3: camera
                with ui.HStack(height=24, spacing=4):
                    ui.Label("Camera:", width=55)
                    for name in CAMERA_PRESETS:
                        ui.Button(name, width=55, clicked_fn=lambda n=name: self._select_preset(n))
                    ui.Spacer(width=8)
                    self._free_btn = ui.Button("Free Camera", width=90, clicked_fn=self._toggle_free)
                # Row 4: motion selector (only if multiple motions)
                if len(self._motion_names) > 1:
                    with ui.HStack(height=24, spacing=4):
                        ui.Button("< Prev", width=55, clicked_fn=self._prev_motion)
                        ui.Button("Next >", width=55, clicked_fn=self._next_motion)
                        name = self._motion_names[0] if self._motion_names else "?"
                        self._motion_label = ui.Label(
                            f"[1/{len(self._motion_names)}] {name}", width=500,
                        )

    def _toggle_play(self):
        self.playing = not self.playing
        self._play_btn.text = "Pause" if self.playing else "Play"
        if self.playing:
            self._wall_time = time.perf_counter()

    def _set_speed(self, speed: float):
        self.playback_speed = speed
        self._speed_label.text = f"{speed:g}x"

    def _on_slider_changed(self, model):
        if self._updating_slider:
            return
        self.current_frame = int(model.as_int)
        self._playback_time = self.current_frame / self._fps
        if self.playing:
            self.playing = False
            self._play_btn.text = "Play"

    def _select_preset(self, name: str):
        self._cam.set_preset(name)
        self._free_btn.text = "Free Camera"

    def _toggle_free(self):
        self._cam.tracking = not self._cam.tracking
        self._free_btn.text = "Track Camera" if not self._cam.tracking else "Free Camera"

    def _prev_motion(self):
        idx = (self._current_motion_idx - 1) % len(self._motion_names)
        self._motion_switch_requested = idx

    def _next_motion(self):
        idx = (self._current_motion_idx + 1) % len(self._motion_names)
        self._motion_switch_requested = idx

    def tick(self) -> int:
        if self.playing:
            now = time.perf_counter()
            dt = now - self._wall_time
            self._wall_time = now
            self._playback_time += dt * self.playback_speed
            if self._playback_time >= self._duration:
                self._playback_time %= self._duration
            self.current_frame = min(int(self._playback_time * self._fps), self._total_frames - 1)
            self._updating_slider = True
            self._slider.model.set_value(self.current_frame)
            self._updating_slider = False

        t = self.current_frame / self._fps
        self._time_label.text = f"{t:.2f} / {self._duration:.2f}s"
        self._frame_label.text = f"Frame {self.current_frame} / {self._total_frames - 1}"
        return self.current_frame


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
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
# Main
# ---------------------------------------------------------------------------

def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()
    device = sim.device

    robot_joint_indices = robot.find_joints(DR02_PRO_JOINT_NAMES, preserve_order=True)[0]

    # ---- Load motions ----
    motions = []
    motion_names = []

    if args_cli.folder is not None:
        npz_files = sorted(glob.glob(os.path.join(args_cli.folder, "*.npz")))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {args_cli.folder}")
        for f in npz_files:
            motions.append(FkNpzMotion(f, device=device))
            motion_names.append(motions[-1].name)
        print(f"[INFO] Loaded {len(motions)} motions from {args_cli.folder}")
    else:
        # Legacy single-file mode
        if args_cli.retarget_format == "omniretarget":
            motions.append(OmniRetargetNpzMotion(args_cli.file, device=device))
        else:
            motions.append(LegacyMergedNpzMotion(args_cli.file, device=device))
        motion_names.append(motions[0].name)

    # Legacy FK file for sphere visualization with old format
    legacy_fk = None
    if args_cli.fk_file is not None:
        legacy_fk = LegacyFkNpzMotion(args_cli.fk_file, device=device)

    current_idx = 0
    motion = motions[current_idx]

    # ---- FK sphere markers ----
    fk_offset = torch.tensor([FK_X_OFFSET, 0.0, 0.0], device=device)
    has_fk = motion.body_pos_w is not None or legacy_fk is not None
    fk_markers = None
    if has_fk:
        num_bodies = motion.num_bodies if motion.body_pos_w is not None else legacy_fk.num_bodies
        fk_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/FkKeypoints",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.05,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.2, 0.2),
                            emissive_color=(0.6, 0.05, 0.05),
                        ),
                    ),
                },
            )
        )

    cam_ctrl = CameraController(sim)
    replay_ui = ReplayUI(motion, cam_ctrl, motion_names=motion_names)

    env_origin = scene.env_origins[0]
    env_origin_np = env_origin.cpu().numpy()
    env_ids = torch.tensor([0], device=device, dtype=torch.long)

    while simulation_app.is_running():
        # ---- Check for motion switch ----
        switch_idx = replay_ui.motion_switch_requested
        if switch_idx is not None:
            current_idx = switch_idx
            motion = motions[current_idx]
            replay_ui.update_motion(motion, current_idx)
            replay_ui.playing = True
            replay_ui._play_btn.text = "Pause"
            replay_ui._wall_time = time.perf_counter()

        frame = replay_ui.tick()
        frame = min(frame, motion.num_frames - 1)

        # ---- Robot (joint angles, sim FK) ----
        root_state = robot.data.default_root_state[0:1].clone()
        root_state[0, :3] = motion.root_pos[frame]
        root_state[0, :2] += env_origin[:2]
        root_state[0, 3:7] = motion.root_quat_wxyz[frame]
        robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        jpos = robot.data.default_joint_pos[0:1].clone()
        jvel = robot.data.default_joint_vel[0:1].clone()
        if isinstance(motion, FkNpzMotion):
            # New format: joint_pos is already in articulation order
            jpos[0, :] = motion.joint_pos[frame]
            jvel[0, :] = motion.joint_vel[frame]
        else:
            # Legacy / OmniRetarget: dof is in DR02_PRO_JOINT_NAMES order, needs remapping
            jpos[0, robot_joint_indices] = motion.joint_pos[frame]
            jvel[0, robot_joint_indices] = motion.joint_vel[frame]
        robot.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids)
        scene.write_data_to_sim()

        # ---- FK keypoints (spheres only, no robot) ----
        fk_body_pos = None
        fk_num_bodies = 0
        if motion.body_pos_w is not None:
            fk_body_pos = motion.body_pos_w[frame].clone()
            fk_num_bodies = motion.num_bodies
        elif legacy_fk is not None:
            fk_frame = min(frame, legacy_fk.num_frames - 1)
            fk_body_pos = legacy_fk.body_pos_w[fk_frame].clone()
            fk_num_bodies = legacy_fk.num_bodies

        if fk_body_pos is not None and fk_markers is not None:
            fk_body_pos += env_origin + fk_offset
            fk_markers.visualize(
                translations=fk_body_pos,
                marker_indices=torch.zeros(fk_num_bodies, dtype=torch.int32, device=device),
            )

        # Camera tracks midpoint
        robot_root_np = motion.root_pos[frame].cpu().numpy() + env_origin_np
        if fk_body_pos is not None:
            midpoint = robot_root_np + np.array([FK_X_OFFSET / 2.0, 0.0, 0.0])
        else:
            midpoint = robot_root_np
        cam_ctrl.update(midpoint)

        sim.render()
        scene.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    hide_default_panels()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()

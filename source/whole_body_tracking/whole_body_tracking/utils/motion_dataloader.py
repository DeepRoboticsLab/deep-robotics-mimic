"""Motion dataloader with concatenated GPU buffer and offset-based indexing.

All motion sequences are concatenated into single contiguous tensors on GPU.
Per-motion access uses offset + local_time_step for efficient vectorized gather.
"""

from collections.abc import Sequence

import torch

from whole_body_tracking.utils.motion_dataset import Motion_Dataset


class Motion_Dataloader:
    """Dataloader that pre-loads all motions into a single concatenated GPU buffer.

    Usage:
        >>> dataset = Motion_Dataset(...)
        >>> loader = Motion_Dataloader(dataset, body_indexes=[...], device="cuda")
        >>> # Access frame t of motion m:
        >>> global_idx = loader.motion_offsets[m] + t
        >>> joint_pos = loader.motion_buffer.joint_pos[global_idx]
    """

    class MotionBuffer:
        """Stores concatenated motion tensors on GPU."""

        def __init__(self, body_indexes: Sequence[int]):
            self.joint_pos: torch.Tensor | None = None
            self.joint_vel: torch.Tensor | None = None
            self._body_pos_w: torch.Tensor | None = None
            self._body_quat_w: torch.Tensor | None = None
            self._body_lin_vel_w: torch.Tensor | None = None
            self._body_ang_vel_w: torch.Tensor | None = None
            self.body_indexes = body_indexes

        @property
        def body_pos_w(self) -> torch.Tensor:
            return self._body_pos_w[:, self.body_indexes]

        @property
        def body_quat_w(self) -> torch.Tensor:
            return self._body_quat_w[:, self.body_indexes]

        @property
        def body_lin_vel_w(self) -> torch.Tensor:
            return self._body_lin_vel_w[:, self.body_indexes]

        @property
        def body_ang_vel_w(self) -> torch.Tensor:
            return self._body_ang_vel_w[:, self.body_indexes]

    def __init__(
        self,
        dataset: Motion_Dataset,
        body_indexes: Sequence[int],
        device: str = "cuda",
    ):
        self.dataset = dataset
        self.device = device
        self._body_indexes = body_indexes

        self.motion_buffer = self.MotionBuffer(self._body_indexes)

        self.motion_lengths: torch.Tensor   # [num_motions]
        self.motion_offsets: torch.Tensor   # [num_motions]
        self.motion_fps: torch.Tensor       # [num_motions]
        self.time_step_total: int
        self.num_motions: int

        print(f"[Motion_Dataloader] Loading and concatenating {len(dataset)} motions...")
        self._preload_and_concatenate()
        print(f"[Motion_Dataloader] Done. Total frames: {self.time_step_total}")

    def _preload_and_concatenate(self):
        """Load all motions and concatenate into single GPU tensors."""
        data_lists = {
            "joint_pos": [],
            "joint_vel": [],
            "body_pos_w": [],
            "body_quat_w": [],
            "body_lin_vel_w": [],
            "body_ang_vel_w": [],
        }
        lengths = []
        fps_list = []

        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            motion = sample["motion"]

            for key in data_lists:
                data_lists[key].append(
                    torch.tensor(motion[key], dtype=torch.float32, device=self.device)
                )
            lengths.append(sample["length"])
            fps_list.append(sample["fps"])

        # Concatenate along time dimension
        self.motion_buffer.joint_pos = torch.cat(data_lists["joint_pos"], dim=0)
        self.motion_buffer.joint_vel = torch.cat(data_lists["joint_vel"], dim=0)
        self.motion_buffer._body_pos_w = torch.cat(data_lists["body_pos_w"], dim=0)
        self.motion_buffer._body_quat_w = torch.cat(data_lists["body_quat_w"], dim=0)
        self.motion_buffer._body_lin_vel_w = torch.cat(data_lists["body_lin_vel_w"], dim=0)
        self.motion_buffer._body_ang_vel_w = torch.cat(data_lists["body_ang_vel_w"], dim=0)

        self.num_motions = len(lengths)
        self.motion_lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.motion_offsets = torch.cat([
            torch.tensor([0], device=self.device),
            torch.cumsum(self.motion_lengths, dim=0)[:-1],
        ], dim=0)
        self.motion_fps = torch.tensor(fps_list, dtype=torch.float32, device=self.device)
        self.time_step_total = self.motion_buffer.joint_pos.shape[0]

        print(f"  - Motions: {self.num_motions}")
        print(f"  - joint_pos: {self.motion_buffer.joint_pos.shape}")
        print(f"  - body_pos_w: {self.motion_buffer.body_pos_w.shape}")
        print(f"  - Total frames: {self.time_step_total}")
        print(f"  - Length range: [{self.motion_lengths.min()}, {self.motion_lengths.max()}]")

    def sample(self, n: int) -> torch.Tensor:
        """Uniformly sample n motion indices.

        Returns:
            motion_indices: Tensor[n] of sampled motion IDs.
        """
        return torch.randint(0, self.num_motions, (n,), device=self.device)

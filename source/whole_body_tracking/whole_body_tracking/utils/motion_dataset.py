"""Motion dataset for loading motion data from NPZ files with YAML-based configuration.

Supports multiple dataset directories and train/val splits.
"""

import os
from pathlib import Path
from typing import Any, Union

import numpy as np
import yaml
from torch.utils.data import Dataset


class Motion_Dataset(Dataset):
    """PyTorch Dataset for loading motion data from NPZ files.

    Each dataset directory must contain an info.yaml with structure:
        dataset: "dataset_name"
        train:
          train/motion_name1: 1
          train/motion_name2: 1
          ...

    NPZ files are located at: <dataset_dir>/<robot_name>/<motion_name>.npz

    Args:
        dataset_dirs: List of dataset directory paths.
        robot_name: Robot subdirectory name (e.g. "cr1").
        splits: List of splits, one per dataset_dir. Each can be a string
                or list of strings to combine multiple splits.
    """

    def __init__(
        self,
        dataset_dirs: list[str],
        robot_name: str,
        splits: list[Union[str, list[str]]],
    ):
        super().__init__()

        if len(splits) != len(dataset_dirs):
            raise ValueError(
                f"Length of splits ({len(splits)}) must match dataset_dirs ({len(dataset_dirs)})"
            )

        self.dataset_dirs = [Path(d).expanduser().resolve() for d in dataset_dirs]
        self.robot_name = robot_name
        self.splits = splits

        self.npz_paths: list[Path] = []
        self.quantities: list[int] = []
        self.motion_names: list[str] = []

        self._load_dataset_info()

        print(f"[Motion_Dataset] Loaded {len(self.npz_paths)} motion clips")

    def _load_dataset_info(self):
        """Parse info.yaml files and collect NPZ paths."""
        for dataset_idx, dataset_dir in enumerate(self.dataset_dirs):
            split_config = self.splits[dataset_idx]
            split_names = [split_config] if isinstance(split_config, str) else split_config

            # Find info.yaml
            info_path = None
            for ext in [".yaml", ".yml"]:
                candidate = dataset_dir / f"info{ext}"
                if candidate.exists():
                    info_path = candidate
                    break
            if info_path is None:
                raise FileNotFoundError(f"No info.yaml found in {dataset_dir}")

            with open(info_path, "r") as f:
                info = yaml.safe_load(f)

            robot_dir = dataset_dir / self.robot_name
            if not robot_dir.exists():
                raise FileNotFoundError(f"Robot directory not found: {robot_dir}")

            for split in split_names:
                split_info = info.get(split, {})
                if not split_info:
                    raise ValueError(f"No '{split}' split in {info_path}")

                for motion_name, quantity in split_info.items():
                    npz_path = robot_dir / f"{motion_name}.npz"
                    if npz_path.exists():
                        self.npz_paths.append(npz_path)
                        self.quantities.append(quantity)
                        self.motion_names.append(motion_name)
                    else:
                        print(f"[Motion_Dataset] Warning: NPZ not found: {npz_path}")

    def __len__(self) -> int:
        return len(self.npz_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load and return a single motion clip (lazy loading).

        Returns dict with keys: motion, fps, length, motion_name.
        """
        npz_path = self.npz_paths[idx]
        data = np.load(npz_path)

        motion = {
            "joint_pos": data["joint_pos"],
            "joint_vel": data["joint_vel"],
            "body_pos_w": data["body_pos_w"],
            "body_quat_w": data["body_quat_w"],
            "body_lin_vel_w": data["body_lin_vel_w"],
            "body_ang_vel_w": data["body_ang_vel_w"],
        }

        fps = int(data["fps"][0])
        length = motion["joint_pos"].shape[0]

        return {
            "motion": motion,
            "fps": fps,
            "length": length,
            "motion_name": self.motion_names[idx],
        }

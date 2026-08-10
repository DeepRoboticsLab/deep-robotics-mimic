"""Convert robot motion NPZ files to JSON format for visualization.

Usage:
    python scripts/npz_to_json.py --input motion.npz --output motion.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DR02_pro joint name order (matches Isaac Lab USD traversal order)
JOINT_NAMES = [
    "left_hip_y_joint", "right_hip_y_joint", "waist_z_joint",
    "left_hip_x_joint", "right_hip_x_joint", "waist_x_joint",
    "left_hip_z_joint", "right_hip_z_joint", "waist_y_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_y_joint", "right_shoulder_y_joint",
    "left_ankle_y_joint", "right_ankle_y_joint",
    "left_shoulder_x_joint", "right_shoulder_x_joint",
    "left_ankle_x_joint", "right_ankle_x_joint",
    "left_shoulder_z_joint", "right_shoulder_z_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_z_joint", "right_wrist_z_joint",
    "left_wrist_y_joint", "right_wrist_y_joint",
    "left_wrist_x_joint", "right_wrist_x_joint",
]


def numpy_to_python(obj):
    """Recursively convert numpy data types to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def convert_npz_to_json(npz_path, json_path):
    """Convert an NPZ motion file to a JSON file with ordered keys.

    The output JSON contains: fps, total_frames, joint_names, joint_pos,
    joint_vel, and any additional arrays from the NPZ.
    """
    try:
        # --- Load NPZ and convert all arrays to native Python ---
        data = np.load(npz_path)
        json_data = {}
        for key in data.files:
            json_data[key] = numpy_to_python(data[key])

        # --- Determine total frame count from the first matching key ---
        n = None
        for key in ['joint_pos', 'joint_vel', 'body_pos_w',
                     'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w']:
            if key in json_data:
                n = len(json_data[key])
                break
        if n is None:
            n = 0

        # --- Build ordered output: fps → total_frames → joint_names → data ---
        ordered_data = {}
        if 'fps' in json_data:
            ordered_data['fps'] = json_data['fps']
        ordered_data['total_frames'] = n
        ordered_data['joint_names'] = JOINT_NAMES
        for key in json_data:
            if key not in ('fps',) and key in ('joint_pos', 'joint_vel'):
                ordered_data[key] = json_data[key]

        # --- Write JSON ---
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(ordered_data, f, indent=2, ensure_ascii=False)

        print(f"Converted: {npz_path} -> {json_path}")
        print(f"Keys: {list(ordered_data.keys())}")

    except Exception as e:
        print(f"Error during conversion: {e}")
        raise


def main():
    """Parse CLI arguments and run the conversion."""
    parser = argparse.ArgumentParser(
        description='Convert robot motion NPZ file to JSON format.')
    parser.add_argument('--input', type=str, help='Input NPZ file path')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    args = parser.parse_args()

    # --- Validate input ---
    if not Path(args.input).exists():
        print(f"Error: input file {args.input} does not exist")
        return

    # --- Ensure output directory exists ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Run conversion ---
    convert_npz_to_json(args.input, args.output)


if __name__ == "__main__":
    main()

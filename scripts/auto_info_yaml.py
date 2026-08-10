"""Auto-generate info.yaml for a dataset directory by scanning NPZ files.

Usage:
    python scripts/auto_info_yaml.py \
        --npz_dir dataset/DR02_pro/multi_motion_npz \
        --dataset_name DR02_pro_multi_motion \
        --robot_name DR02_pro \
        --output_dir dataset/DR02_pro_multi_motion

This creates:
    datasets/DR02_pro_multi_motion/info.yaml
    datasets/DR02_pro_multi_motion/DR02_pro/train/  (symlinks to NPZ files)
"""

import argparse
import glob
import os
import yaml


def main():
    """Generate info.yaml and symlink structure for a motion dataset."""
    parser = argparse.ArgumentParser(description="Auto-generate info.yaml for motion dataset.")
    parser.add_argument("--npz_dir", type=str, required=True, help="Directory with training-ready NPZ files.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name for info.yaml.")
    parser.add_argument("--robot_name", type=str, default="DR02_pro", help="Robot name (subdirectory).")
    parser.add_argument("--output_dir", type=str, required=True, help="Output dataset directory.")
    parser.add_argument("--split", type=str, default="train", help="Split name.")
    parser.add_argument("--symlink", action="store_true", default=True, help="Create symlinks to NPZ files.")
    args = parser.parse_args()

    # --- Resolve paths ---
    npz_dir = os.path.abspath(args.npz_dir)
    output_dir = os.path.abspath(args.output_dir)
    robot_dir = os.path.join(output_dir, args.robot_name)
    split_dir = os.path.join(robot_dir, args.split)

    # --- Scan NPZ files ---
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    print(f"Found {len(npz_files)} NPZ files in {npz_dir}")

    # --- Build info.yaml content ---
    info = {"dataset": args.dataset_name}
    split_entries = {}
    for npz_file in npz_files:
        stem = os.path.splitext(os.path.basename(npz_file))[0]
        motion_key = f"{args.split}/{stem}"
        split_entries[motion_key] = 1  # quantity = 1 (good quality)
    info[args.split] = split_entries

    # --- Create output directories ---
    os.makedirs(split_dir, exist_ok=True)

    # --- Write info.yaml ---
    info_path = os.path.join(output_dir, "info.yaml")
    with open(info_path, "w") as f:
        yaml.dump(info, f, default_flow_style=False, sort_keys=False)
    print(f"Written: {info_path}")

    # --- Create symlinks to NPZ files ---
    if args.symlink:
        for npz_file in npz_files:
            basename = os.path.basename(npz_file)
            link_path = os.path.join(split_dir, basename)
            if not os.path.exists(link_path):
                os.symlink(os.path.abspath(npz_file), link_path)
        print(f"Created {len(npz_files)} symlinks in {split_dir}")

    print(f"[DONE] Dataset directory ready at {output_dir}")


if __name__ == "__main__":
    main()

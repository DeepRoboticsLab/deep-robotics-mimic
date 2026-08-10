# deep-robotics-mimic

Humanoid motion tracking training on Isaac Lab. Trains PPO policies that track reference motions (DeepMimic-style reward). Primary robot: **DR02_pro**.

## Installation

1. Install [Isaac Lab v2.3.2](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) (conda recommended)

2. **PyTorch 2.7.0, CUDA 12.8** — other versions will degrade simulation speed

3. **rsl-rl-lib 5.0.1** — `pip install rsl-rl-lib==5.0.1`

```bash
conda create -n mimic python=3.11
conda activate mimic

pip install --upgrade pip

# Install base tools
pip install setuptools==80.9.0 wheel packaging

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

pip install flatdict==4.0.1 --no-build-isolation

pip install "isaaclab[isaacsim,all]==2.3.2" --extra-index-url https://pypi.nvidia.com --no-build-isolation

pip install rsl-rl-lib==5.0.1

# Check versions
pip list | grep -E "torch|isaac|rsl|stable"
```

4. Clone and install:

```bash
git clone https://github.com/DeepRoboticsLab/deep-robotics-mimic.git
cd deep-robotics-mimic
python -m pip install -e source/whole_body_tracking
```

## Data Pipeline

Raw pipeline: BVH/SMPLX → `.pkl` (retargeting) → `.npz` → FK `.npz` → Training → `.json` (deployment)

### BVH/SMPLX → pkl -> npz(retargeting)

Retargeting is done by the companion project `deep-robotics-retarget`. Its outputs are robot joint `.pkl` files.


### npz → FK npz (forward kinematics, requires Isaac Sim)

**Single file:**
```bash
python scripts/convert_DR02_pro.py \
  --input <motion>.npz \
  --output dataset/gmr/<motion>.npz \
  --num_envs 10000 \
  --output_fps 50 \
  --retarget_format gmr \
  --headless
```

**Batch folder:**
```bash
python scripts/batch_convert_DR02_pro.py \
  --input_dir <source_npz_folder>/ \
  --output_dir dataset/ \
  --num_envs 10000 \
  --output_fps 50 \
  --retarget_format gmr \
  --headless
```

Supported `--retarget_format`: `deep_retarget`, `omniretarget`, `gmr`

### npz → json (for deployment controller)

```bash
python scripts/npz_to_json.py --input <file>.npz --output <file>.json
```

## Visualization (requires Isaac Sim)

```bash
python scripts/replay_merged.py --folder dataset/gmr/   # folder of FK npz files
python scripts/replay_merged.py --file <file>.npz --fk_file <fk_file>.npz  # legacy single-file mode
```

## Visualization (MuJoCo, no Isaac Sim required)

```bash
pip install mujoco

python scripts/replay_npz_mujoco.py                       # interactive file selection
python scripts/replay_npz_mujoco.py dataset/gmr/<motion>.npz
python scripts/replay_npz_mujoco.py dataset/gmr/<motion>.npz --verify   # FK accuracy verification
```

## Training (requires Isaac Sim)

### Single GPU

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-DR02_PRO \
  --registry_name dataset/gmr/motion.npz \
  --logger tensorboard \
  --log_project_name logs/ \
  --run_name <run_name> \
  --headless \
  --device cuda:0 \
  --max_iterations 100000
```

### Multi-GPU

```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
  scripts/rsl_rl/train.py \
  --task=Tracking-Flat-DR02_PRO \
  --registry_name dataset/gmr/motion.npz \
  --logger tensorboard \
  --log_project_name logs/ \
  --run_name <run_name> \
  --headless \
  --distributed \
  --num_envs 4096 \
  --max_iterations 200000
```

### Resume from checkpoint

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-DR02_PRO \
  --registry_name dataset/gmr/motion.npz \
  --logger tensorboard \
  --log_project_name logs/ \
  --run_name <run_name> \
  --headless \
  --device cuda:0 \
  --max_iterations 200000 \
  --resume True \
  --checkpoint <model>.pt \
  --load_run <YYYY-MM-DD_HH-MM-SS_run_name>
```

`--checkpoint` is just the filename. `--load_run` is the folder name under `logs/rsl_rl/{experiment_name}/`.

### Auto-restart on NaN (recommended for long runs)

Wraps `train.py` in a subprocess. On NaN loss, automatically restarts from a checkpoint ~1000 iterations earlier. Up to 10 retries.

```bash
# Single GPU
python scripts/rsl_rl/train_auto_restart.py \
  --task=Tracking-Flat-DR02_PRO \
  --registry_name dataset/gmr/motion.npz \
  --logger tensorboard \
  --log_project_name logs/ \
  --run_name <run_name> \
  --headless \
  --max_iterations 100000

# Multi-GPU — pass --nproc_per_node directly, do NOT use torch.distributed.run
python scripts/rsl_rl/train_auto_restart.py \
  --nproc_per_node 2 \
  --task=Tracking-Flat-DR02_PRO \
  --registry_name dataset/gmr/motion.npz \
  --logger tensorboard \
  --log_project_name logs/ \
  --run_name <run_name> \
  --headless \
  --distributed \
  --num_envs 4096 \
  --max_iterations 200000
```

### Available tasks

`Tracking-Flat-DR02_PRO`

## Evaluation (requires Isaac Sim)

```bash
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-DR02_PRO \
  --motion_file dataset/gmr/motion.npz \
  --checkpoint_path logs/rsl_rl/DR02_pro_flat/<run>/model_10000.pt \
  --num_envs 2
```

Also auto-exports ONNX to an `exported/` subdirectory next to the checkpoint.

## ONNX Export

### Fast export (no Isaac Sim)

```bash
python scripts/rsl_rl/export_onnx_fast.py \
  --checkpoint_path logs/rsl_rl/DR02_pro_flat/<run>/model_10000.pt \
  --output_name <motion>.onnx
```

Infers network shape from checkpoint. Embeds hardcoded DR02_pro metadata (joint names, stiffness/damping, action scale).

### Export motion json + policy onnx (interactive)

```bash
python scripts/export_motion_and_policy.py
```

Scans training runs under `logs/rsl_rl/`, converts the motion file from `params/env.yaml` to JSON via `npz_to_json.py`, and exports the selected `model_*.pt` checkpoint to ONNX via `export_onnx_fast.py`.

## Utilities

```bash
# Compare configs between two training runs
python scripts/compare_runs.py logs/rsl_rl/DR02_pro_flat/<run1> logs/rsl_rl/DR02_pro_flat/<run2>

# Generate dataset info.yaml from a folder of NPZ files
python scripts/auto_info_yaml.py \
  --npz_dir dataset/gmr/ \
  --dataset_name DR02_pro_multi_motion \
  --robot_name DR02_pro \
  --output_dir dataset/DR02_pro_multi_motion
```

## Quick Reference

| Task | Script |
|---|---|
| npz → FK npz (single) | `scripts/convert_DR02_pro.py` |
| npz → FK npz (batch) | `scripts/batch_convert_DR02_pro.py` |
| npz → json | `scripts/npz_to_json.py` |
| Export motion + policy ONNX | `scripts/export_motion_and_policy.py` |
| Generate dataset info.yaml | `scripts/auto_info_yaml.py` |
| Compare run configs | `scripts/compare_runs.py` |
| Replay motion (Isaac Sim) | `scripts/replay_merged.py` |
| Replay motion (MuJoCo) | `scripts/replay_npz_mujoco.py` |
| Train policy | `scripts/rsl_rl/train.py` |
| Train + auto NaN restart | `scripts/rsl_rl/train_auto_restart.py` |
| Evaluate policy | `scripts/rsl_rl/play.py` |
| Export ONNX (fast) | `scripts/rsl_rl/export_onnx_fast.py` |

## Motion Data

Training-ready FK `.npz` files are under `dataset/gmr/` (e.g. `jugong.npz`, `huishou.npz`, `daquan.npz`).

## License

BSD 3-Clause — See [LICENSE](LICENSE) for details.

# deep-robotics-mimic

Humanoid motion tracking training on Isaac Lab. Trains PPO policies that track reference motions (DeepMimic-style reward). Primary robot: **DR02_pro**.

## 1. Installation

> [!NOTE]
> **This project has been verified on Ubuntu 24.04 + NVIDIA driver 580.173.02 + CUDA 13.0**

- Install dependencies (Isaac Lab v2.3.2 + PyTorch 2.7.0 + rsl-rl-lib 5.0.1)

```bash
conda create -n deep-robotics-humanoid python=3.11 # same environment as the deep-robotics-retarget project
conda activate deep-robotics-humanoid

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

- Clone and install:

```bash
git clone https://github.com/DeepRoboticsLab/deep-robotics-mimic.git
cd deep-robotics-mimic
python -m pip install -e source/whole_body_tracking
```

## 2. Data Pipeline

Full pipeline: BVH/SMPLX → `.pkl` (retargeting) → FK `.npz` → Training

### 2.1. BVH/SMPLX → pkl (retargeting)

Retargeting is done by the companion project `deep-robotics-retarget`. Its outputs are robot joint `.pkl` files, which can be converted directly to FK `.npz` in one step (see 2.2).

### 2.2. pkl/npz → FK npz (forward kinematics, requires Isaac Sim)

**Single file (PKL input, one step to FK npz):**
```bash
python scripts/convert_DR02_pro.py \
  --input <motion>.pkl \
  --output dataset/gmr/<motion>.npz \
  --num_envs 10000 \
  --output_fps 50 \
  --retarget_format gmr \
  --headless
```

**Single file (NPZ input):**
```bash
python scripts/convert_DR02_pro.py \
  --input <motion>.npz \
  --output dataset/gmr/<motion>_fk.npz \
  --num_envs 10000 \
  --output_fps 50 \
  --retarget_format gmr \
  --headless
```

**Batch folder:**
```bash
python scripts/batch_convert_DR02_pro.py \
  --input_dir <source_folder>/ \
  --output_dir dataset/ \
  --num_envs 10000 \
  --output_fps 50 \
  --retarget_format gmr \
  --headless
```

Supported `--retarget_format`: `deep_retarget`, `omniretarget`, `gmr`. `.pkl` inputs (gmr retargeting output) are only supported with `gmr`; a batch folder may contain mixed `.pkl` and `.npz` files.

## 3. Visualization (Isaac Sim, requires Isaac Sim)

```bash
python scripts/replay_merged.py --folder dataset/gmr/   # folder of FK npz files
python scripts/replay_merged.py --file dataset/gmr/boxing.npz   # single-file mode
```

## 4. Visualization (MuJoCo, no Isaac Sim required)

```bash
pip install mujoco

python scripts/replay_npz_mujoco.py                       # interactive file selection
python scripts/replay_npz_mujoco.py dataset/gmr/<motion>.npz
python scripts/replay_npz_mujoco.py dataset/raw/pkl/<motion>.pkl   # also supports gmr retargeted PKL
python scripts/replay_npz_mujoco.py dataset/gmr/<motion>.npz --verify   # FK accuracy verification (NPZ only)
```

Playback includes a ground plane, robot-following lighting, an on-screen progress bar, and Space-key pause/resume.

## 5. Training (requires Isaac Sim)

### 5.1. Single GPU

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

**Parameter reference:**

| Flag | Description |
|---|---|
| `--task` | Task name; loads the environment and default PPO config from the Isaac Lab registry |
| `--registry_name` | Path to the reference motion file (FK npz); only needed for single-motion training — multi-motion tasks use `info.yaml` dataset indexing |
| `--logger` | Logging backend: `tensorboard` / `wandb` / `neptune` |
| `--log_project_name` | Log output directory |
| `--run_name` | Run name suffix used to tell experiments apart under the log directory |
| `--headless` | No GUI mode; disables the Isaac Sim render window (recommended for large-scale training) |
| `--num_envs` | Total number of parallel environments (split across GPUs) |
| `--max_iterations` | Maximum PPO training iterations |
| `--device` | Compute device (e.g. `cuda:0`); used for single-GPU training |

### 5.2. Multi-GPU

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

**Parameter reference:**

| Flag | Description |
|---|---|
| `--nnodes=1 --nproc_per_node=2` | PyTorch distributed launcher options: 1 node, 2 processes per node (i.e. data-parallel training on 2 GPUs) |
| `--task` | Task name; loads the environment and default PPO config from the Isaac Lab registry |
| `--registry_name` | Path to the reference motion file (FK npz); only needed for single-motion training — multi-motion tasks use `info.yaml` dataset indexing |
| `--logger` | Logging backend: `tensorboard` / `wandb` / `neptune` |
| `--log_project_name` | Log output directory |
| `--run_name` | Run name suffix used to tell experiments apart under the log directory |
| `--headless` | No GUI mode; disables the Isaac Sim render window (recommended for large-scale training) |
| `--distributed` | Marks a distributed run; only use together with `torch.distributed.run` |
| `--num_envs` | Total number of parallel environments (split across GPUs) |
| `--max_iterations` | Maximum PPO training iterations |

### 5.3. Resume from checkpoint

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

### 5.4. Auto-restart on NaN (recommended for long runs)

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

### 5.5. Available tasks

`Tracking-Flat-DR02_PRO`

## 6. Evaluation (requires Isaac Sim)

```bash
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-DR02_PRO \
  --motion_file dataset/gmr/motion.npz \
  --checkpoint_path logs/rsl_rl/DR02_pro_flat/<run>/model_10000.pt \
  --num_envs 2
```

Also auto-exports ONNX to an `exported/` subdirectory next to the checkpoint.

## 7. ONNX Export

### 7.1. Fast export (no Isaac Sim)

```bash
python scripts/rsl_rl/export_onnx_fast.py \
  --checkpoint_path logs/rsl_rl/DR02_pro_flat/<run>/model_10000.pt \
  --output_name <motion>.onnx
```

Infers network shape from checkpoint. Embeds hardcoded DR02_pro metadata (joint names, stiffness/damping, action scale).

### 7.2. Export motion json + policy onnx (interactive)

```bash
python scripts/export_motion_and_policy.py
```

Scans training runs under `logs/rsl_rl/`, converts the motion file from `params/env.yaml` to JSON via `npz_to_json.py`, and exports the selected `model_*.pt` checkpoint to ONNX via `export_onnx_fast.py`.

## 8. Utilities

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

## 9. Quick Reference

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

## 10. Motion Data

Training-ready FK `.npz` files are under `dataset/gmr/` (e.g. `bow.npz`, `boxing.npz`, `wave_hand.npz`).

## 11. License

BSD 3-Clause — See [LICENSE](LICENSE) for details.

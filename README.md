# deep-robotics-mimic

Humanoid motion tracking training on Isaac Lab. Trains PPO policies that track reference motions (DeepMimic-style reward). Primary robot: **DR02_pro**.

## 1. Installation

> [!NOTE]
> **This project has been verified on Ubuntu 22.04/24.04 + NVIDIA driver 580.173.02 + CUDA 13.0**

- Install dependencies (Isaac Lab v2.3.2 + PyTorch 2.7.0 + rsl-rl-lib 5.0.1)

```bash
conda create -n deep-robotics-humanoid python=3.11 -y # same environment as the deep-robotics-retarget project
conda activate deep-robotics-humanoid

pip install --upgrade pip

# Install base tools
pip install setuptools==80.9.0 wheel packaging

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0

pip install flatdict==4.0.1 --no-build-isolation

pip install "isaaclab[isaacsim,all]==2.3.2" --extra-index-url https://pypi.nvidia.com --no-build-isolation

pip install rsl-rl-lib==5.0.1

# Check versions
pip list | grep -E "torch|isaac|rsl|stable"
```

Expected versions:

```
isaaclab                             2.3.2
isaacsim                             5.1.0.0
isaacsim-app                         5.1.0.0
isaacsim-asset                       5.1.0.0
isaacsim-benchmark                   5.1.0.0
isaacsim-code-editor                 5.1.0.0
isaacsim-core                        5.1.0.0
isaacsim-cortex                      5.1.0.0
isaacsim-example                     5.1.0.0
isaacsim-extscache-kit               5.1.0.0
isaacsim-extscache-kit-sdk           5.1.0.0
isaacsim-extscache-physics           5.1.0.0
isaacsim-gui                         5.1.0.0
isaacsim-kernel                      5.1.0.0
isaacsim-replicator                  5.1.0.0
isaacsim-rl                          5.1.0.0
isaacsim-robot                       5.1.0.0
isaacsim-robot-motion                5.1.0.0
isaacsim-robot-setup                 5.1.0.0
isaacsim-ros1                        5.1.0.0
isaacsim-ros2                        5.1.0.0
isaacsim-sensor                      5.1.0.0
isaacsim-storage                     5.1.0.0
isaacsim-template                    5.1.0.0
isaacsim-test                        5.1.0.0
isaacsim-utils                       5.1.0.0
rsl-rl-lib                           5.0.1
stable_baselines3                    2.8.0
torch                                2.7.0
torchaudio                           2.7.0
torchvision                          0.22.0
```
- Clone and install:

```bash
git clone https://github.com/DeepRoboticsLab/deep-robotics-mimic.git
cd deep-robotics-mimic
python -m pip install -e source/whole_body_tracking
```

### Known issue: Isaac Sim 5.1.0 startup crash with NVIDIA driver 595.x

A startup segmentation fault in `librtx.scenedb.plugin.so` was reported on Ubuntu 24.04.4 with an RTX 4090 and driver 595.71.05, even after disabling IOMMU. NVIDIA support attributes the crash to driver incompatibility and recommends switching to the validated Linux driver 580.65.06. This project was tested with driver 580.173.02, as noted above. See the [NVIDIA forum discussion](https://forums.developer.nvidia.com/t/isaac-sim-5-1-0-crashes-on-startup-with-rtx-4090-on-ubuntu-24-04-4-segfaulting-in-librtx-scenedb-plugin-so-after-iommu-was-disabled/371957) for details.

## 2. Data Pipeline

Full pipeline: BVH/SMPLX → `.pkl` (retargeting) → FK `.npz` → Training/Deployment

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
python scripts/replay_merged.py --file dataset/gmr/<motion.npz>   # single-file mode
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

We have uploaded the training logs of the boxing motion to [Google Drive](https://drive.google.com/file/d/1WZSotMt6sdUiRtC0KPP94JEuQiuvMezd/view?usp=sharing). They contain the training environment and agent configuration files as well as the policy checkpoints. The policies have been deployment-tested and can be used as a reference for comparison.

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

```bash
python scripts/rsl_rl/export_onnx_fast.py \
  --checkpoint_path logs/rsl_rl/DR02_pro_flat/<run>/model_10000.pt \
  --output_name <motion>.onnx
```

Infers network shape from checkpoint. Embeds hardcoded DR02_pro metadata (joint names, stiffness/damping, action scale).

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

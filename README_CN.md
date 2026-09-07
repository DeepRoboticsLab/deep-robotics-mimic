# deep-robotics-mimic

基于 Isaac Lab 的人形机器人运动跟踪训练。训练 PPO 策略跟踪参考动作（DeepMimic 风格奖励）。主要机器人：**DR02_pro**。

## 1. 安装

> [!NOTE]
> **该项目已在 ubuntu 24.04 + NVIDIA 驱动 580.173.02 + CUDA 13.0 环境下验证通过**

- 安装依赖环境（Isaac Lab v2.3.2 + PyTorch 2.7.0 + rsl-rl-lib 5.0.1）

```bash
conda create -n deep-robotics-humanoid python=3.11 # 该环境与deep-robotics-retarget项目的环境一致
conda activate deep-robotics-humanoid

pip install --upgrade pip

#  安装基础工具
pip install setuptools==80.9.0 wheel packaging

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

pip install flatdict==4.0.1 --no-build-isolation

pip install "isaaclab[isaacsim,all]==2.3.2" --extra-index-url https://pypi.nvidia.com --no-build-isolation

pip install rsl-rl-lib==5.0.1

#  检查版本
pip list | grep -E "torch|isaac|rsl|stable"
```

- 克隆并安装：

```bash
git clone https://github.com/DeepRoboticsLab/deep-robotics-mimic.git
cd deep-robotics-mimic
python -m pip install -e source/whole_body_tracking
```

## 2. 数据格式转换

完整转换流程：BVH/SMPLX → `.pkl`（重定向）→ FK `.npz` → 训练

### 2.1. BVH/SMPLX → pkl（重定向）

重定向由配套项目 `deep-robotics-retarget` 完成，其输出为机器人关节 `.pkl` 文件，可直接一步转换为 FK `.npz`（见 2.2）。

### 2.2. pkl/npz → FK npz（正运动学，需要 Isaac Sim）

**单文件（pkl 输入，一步到 FK npz）：**
```bash
python scripts/convert_DR02_pro.py \
  --input <motion>.pkl \
  --output dataset/gmr/<motion>.npz \
  --num_envs 10000 \
  --output_fps 50 \
  --retarget_format gmr \
  --headless
```

**单文件（npz 输入）：**
```bash
python scripts/convert_DR02_pro.py \
  --input <motion>.npz \
  --output dataset/gmr/<motion>_fk.npz \
  --num_envs 10000 \
  --output_fps 50 \
  --retarget_format gmr \
  --headless
```

**批量文件夹：**
```bash
python scripts/batch_convert_DR02_pro.py \
  --input_dir <source_folder>/ \
  --output_dir dataset/ \
  --num_envs 10000 \
  --output_fps 50 \
  --retarget_format gmr \
  --headless
```

支持的 `--retarget_format`：`deep_retarget`、`omniretarget`、`gmr`。`.pkl` 输入（gmr 重定向输出）仅支持 `gmr`；批量文件夹可混合包含 `.pkl` 和 `.npz` 文件。

## 3. 可视化（Isaac Sim，需要 Isaac Sim）

```bash
python scripts/replay_merged.py --folder dataset/gmr/   # FK npz 文件文件夹
python scripts/replay_merged.py --file dataset/gmr/boxing.npz   # 单文件模式
```

## 4. 可视化（MuJoCo，无需 Isaac Sim）

```bash
pip install mujoco

python scripts/replay_npz_mujoco.py                       # 交互式文件选择
python scripts/replay_npz_mujoco.py dataset/gmr/<motion>.npz
python scripts/replay_npz_mujoco.py dataset/raw/pkl/<motion>.pkl   # 同时支持 gmr 重定向的 PKL
python scripts/replay_npz_mujoco.py dataset/gmr/<motion>.npz --verify   # FK 精度验证（仅限 NPZ）
```

回放包含地面、跟随机器人的灯光、进度条，支持空格键暂停/恢复。

## 5. 训练（需要 Isaac Sim）

### 5.1. 单 GPU

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

**参数说明：**

| 参数 | 说明 |
|---|---|
| `--task` | 任务名，从 Isaac Lab 注册表加载环境与默认 PPO 配置 |
| `--registry_name` | 参考动作文件路径（FK npz）；仅单动作训练需要，多动作任务走 `info.yaml` 数据集索引 |
| `--logger` | 日志后端：`tensorboard` / `wandb` / `neptune` |
| `--log_project_name` | 日志输出目录 |
| `--run_name` | 本次运行名称后缀，用于区分日志目录下不同的实验 |
| `--headless` | 无界面模式，不启动渲染窗口（大规模训练必开） |
| `--num_envs` | 并行仿真环境总数（会分摊到各 GPU 上） |
| `--max_iterations` | PPO 最大训练迭代次数 |
| `--device` | 指定计算设备（如 `cuda:0`），单 GPU 时使用 |

### 5.2. 多 GPU

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

**参数说明：**

| 参数 | 说明 |
|---|---|
| `--nnodes=1 --nproc_per_node=2` | PyTorch 分布式启动器参数：1 个节点、每节点 2 个进程（即 2 张 GPU 数据并行） |
| `--task` | 任务名，从 Isaac Lab 注册表加载环境与默认 PPO 配置 |
| `--registry_name` | 参考动作文件路径（FK npz）；仅单动作训练需要，多动作任务走 `info.yaml` 数据集索引 |
| `--logger` | 日志后端：`tensorboard` / `wandb` / `neptune` |
| `--log_project_name` | 日志输出目录 |
| `--run_name` | 本次运行名称后缀，用于区分日志目录下不同的实验 |
| `--headless` | 无界面模式，不启动渲染窗口（大规模训练必开） |
| `--distributed` | 声明分布式运行，仅在配合 `torch.distributed.run` 启动时使用 |
| `--num_envs` | 并行仿真环境总数（会分摊到各 GPU 上） |
| `--max_iterations` | PPO 最大训练迭代次数 |

### 5.3. 从检查点续训

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

`--checkpoint` 只需文件名。`--load_run` 是 `logs/rsl_rl/{experiment_name}/` 下的文件夹名。

### 5.4. NaN 自动重启（长时间训练推荐）

将 `train.py` 包装在子进程中运行。检测到 NaN 损失时，自动从约 1000 迭代前的检查点重启。最多重试 10 次。

```bash
# 单 GPU
python scripts/rsl_rl/train_auto_restart.py \
  --task=Tracking-Flat-DR02_PRO \
  --registry_name dataset/gmr/motion.npz \
  --logger tensorboard \
  --log_project_name logs/ \
  --run_name <run_name> \
  --headless \
  --max_iterations 100000

# 多 GPU — 直接传 --nproc_per_node，不要使用 torch.distributed.run
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

### 5.5. 可用任务

`Tracking-Flat-DR02_PRO`

## 6. 评估（需要 Isaac Sim）

```bash
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-DR02_PRO \
  --motion_file dataset/gmr/motion.npz \
  --checkpoint_path logs/rsl_rl/DR02_pro_flat/<run>/model_10000.pt \
  --num_envs 2
```

同时会自动将 ONNX 导出到检查点旁边的 `exported/` 子目录。

## 7. ONNX 导出


```bash
python scripts/rsl_rl/export_onnx_fast.py \
  --checkpoint_path logs/rsl_rl/DR02_pro_flat/<run>/model_10000.pt \
  --output_name <motion>.onnx
```

从检查点推断网络结构。嵌入硬编码的 DR02_pro 元数据（关节名称、刚度/阻尼、动作缩放）。

## 8. 工具

```bash
# 对比两次训练运行的配置
python scripts/compare_runs.py logs/rsl_rl/DR02_pro_flat/<run1> logs/rsl_rl/DR02_pro_flat/<run2>

# 从 NPZ 文件文件夹生成数据集 info.yaml
python scripts/auto_info_yaml.py \
  --npz_dir dataset/gmr/ \
  --dataset_name DR02_pro_multi_motion \
  --robot_name DR02_pro \
  --output_dir dataset/DR02_pro_multi_motion
```

## 9. 快速参考

| 任务 | 脚本 |
|---|---|
| npz → FK npz（单文件） | `scripts/convert_DR02_pro.py` |
| npz → FK npz（批量） | `scripts/batch_convert_DR02_pro.py` |
| npz → json | `scripts/npz_to_json.py` |
| 导出动作 + 策略 ONNX | `scripts/export_motion_and_policy.py` |
| 生成数据集 info.yaml | `scripts/auto_info_yaml.py` |
| 对比运行配置 | `scripts/compare_runs.py` |
| 回放动作（Isaac Sim） | `scripts/replay_merged.py` |
| 回放动作（MuJoCo） | `scripts/replay_npz_mujoco.py` |
| 训练策略 | `scripts/rsl_rl/train.py` |
| 训练 + NaN 自动重启 | `scripts/rsl_rl/train_auto_restart.py` |
| 评估策略 | `scripts/rsl_rl/play.py` |
| 快速导出 ONNX | `scripts/rsl_rl/export_onnx_fast.py` |

## 10. 动作数据

训练就绪的 FK `.npz` 文件位于 `dataset/gmr/`（例如 `bow.npz`、`boxing.npz`、`wave_hand.npz`）。

## 11. 许可证

BSD 3-Clause — 详见 [LICENSE](LICENSE)。

# deep-robotics-mimic

基于 Isaac Lab 的人形机器人运动跟踪训练。训练 PPO 策略跟踪参考动作（DeepMimic 风格奖励）。主要机器人：**DR02_pro**。

## 1. 安装

> [!NOTE]
> **该项目已在 Ubuntu 22.04/24.04 + NVIDIA 驱动 580.173.02 + CUDA 13.0 环境下验证通过**

- 安装依赖环境（Isaac Lab v2.3.2 + PyTorch 2.7.0 + rsl-rl-lib 5.0.1）

```bash
conda create -n deep-robotics-humanoid python=3.11 -y # 该环境与deep-robotics-retarget项目的环境一致
conda activate deep-robotics-humanoid

pip install --upgrade pip

#  安装基础工具
pip install setuptools==80.9.0 wheel packaging

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 

pip install flatdict==4.0.1 --no-build-isolation

pip install "isaaclab[isaacsim,all]==2.3.2" --extra-index-url https://pypi.nvidia.com --no-build-isolation

pip install rsl-rl-lib==5.0.1

#  检查版本
pip list | grep -E "torch|isaac|rsl|stable"
```

预期版本：

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
torch                                2.7.0+cu128
torchaudio                           2.7.0+cu128
torchvision                          0.22.0+cu128
```

- 克隆并安装：

```bash
git clone https://github.com/DeepRoboticsLab/deep-robotics-mimic.git
cd deep-robotics-mimic
python -m pip install -e source/whole_body_tracking
```


### Conda C++ 运行库配置（Linux/Bash）

启动 Isaac Sim 前，在共享环境中，从本仓库根目录执行一次：

```bash
python scripts/setup_conda_runtime.py
conda deactivate
conda activate deep-robotics-humanoid
```

如果脚本提示运行库缺失或版本过低，请执行 `conda install -c conda-forge "libstdcxx-ng>=15"` 后重试。脚本安装环境激活/退出钩子，让 Isaac Sim 优先加载 Conda 的 `libstdc++.so.6`；退出环境时恢复原有 `LD_PRELOAD`，不修改系统库。此钩子与 `rl_training` 共用，可重复执行；共享同一环境的仓库只需配置一次。

此配置解决 `CXXABI_1.3.15 not found` 及其引发的 `omni.kit.test` / `omni.graph.core.tests` 导入错误。训练、播放和转换脚本统一依赖此环境钩子，在 Python 启动前选择运行库。重新激活后需重启已有 Python 进程。无窗口模式、URDF 惯量/关节轴和 GPU 性能警告属于其他问题，不会由此配置消除。

### 已知问题：NVIDIA 595.x 驱动下 Isaac Sim 5.1.0 启动崩溃

有用户报告，在 Ubuntu 24.04.4、RTX 4090 和 595.71.05 驱动环境下，启动时出现 `librtx.scenedb.plugin.so` 段错误，禁用 IOMMU 后仍然崩溃。NVIDIA 支持人员将其归因于驱动兼容性问题，建议切换到经过验证的 Linux 驱动 580.65.06。本项目使用的测试驱动为上文所列的 580.173.02。详情参见 [NVIDIA 论坛讨论](https://forums.developer.nvidia.com/t/isaac-sim-5-1-0-crashes-on-startup-with-rtx-4090-on-ubuntu-24-04-4-segfaulting-in-librtx-scenedb-plugin-so-after-iommu-was-disabled/371957)。

## 2. 数据格式转换

完整转换流程：BVH/SMPLX → `.pkl`（重定向）→ FK `.npz` → 训练/部署

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
python scripts/replay_merged.py --file dataset/gmr/<motion.npz>   # 单文件模式
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

我们在[Google Drive](https://drive.google.com/file/d/1WZSotMt6sdUiRtC0KPP94JEuQiuvMezd/view?usp=sharing)里上传了拳击动作的训练日志，其中包含训练环境和智能体的配置信息和策略文件，策略文件已经经过部署测试，可供对比参考使用。

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
| `--headless` | 无界面模式，不启动渲染窗口（大规模训练推荐） |
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
| `--headless` | 无界面模式，不启动渲染窗口（大规模训练推荐） |
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

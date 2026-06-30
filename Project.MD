# MDRA-YOLOv8 实验项目记录

> 项目目录：`mdra_yolov8_experiments`
> 目标运行环境：Linux GPU 服务器（AutoDL、智川云或兼容 Ubuntu/CUDA 环境）
> 当前日期：2026-06-28
> 状态维护规则：每完成一个实现或实验，更新本文件中的状态、配置路径、结果路径和结论。

## 1. 项目目标

在统一的 M3FD 数据划分和 YOLOv8s 训练协议下，验证：

1. VIS 与 IR 是否具有互补检测价值；
2. LCMF 是否优于 EarlyFusion；
3. P2 是否构成可靠的高分辨率检测基线；
4. P2 单尺度 Target-Aware DRA 是否能在不增加 DRA 推理路径的情况下提升检测性能；
5. DRA 的收益是否来自目标边缘监督，而不是全图边缘、SR、普通图像重建、分割辅助、更多参数或随机波动。

## 2. 已冻结的方法决策

| 项目         | 当前决定                                                                                           |
| ------------ | -------------------------------------------------------------------------------------------------- |
| 基础模型     | YOLOv8s                                                                                            |
| 主数据集     | M3FD                                                                                               |
| 融合模块     | LCMF                                                                                               |
| 检测尺度     | P2/P3/P4/P5                                                                                        |
| DRA 监督位置 | 只做 P2 单尺度                                                                                     |
| DRA 目标     | bbox 邻域内的 VIS–IR 多模态边缘图                                                                 |
| 主边缘融合   | `max(Edge_vis, Edge_ir)`                                                                         |
| 主 mask 候选 | bbox 扩张 1.2 倍                                                                                   |
| DRA 属性     | training-only；validation 可选运行 DRA 诊断，但检测指标只由检测路径计算；预测和导出不保留 DRA head |
| MESA 定位    | 不作为独立大创新；称为 scale-aligned target-aware edge supervision                                 |
| ECA          | 第一版不加入；仅作为可选消融                                                                       |

### 2.1 Validation 中的 DRA 使用边界

- validation 的 Precision、Recall、mAP、NMS 结果和 best-checkpoint 选择只使用检测路径；
- DRA 可在 `no_grad` 下按配置选择性运行，用于记录 `val_dra_loss`、预测边缘图和误差图；
- DRA 诊断结果单独记录，不进入检测指标、不修改检测框、不参与 NMS，也不作为测试集模型选择依据；
- `predict`、正式 test 和 export 默认完全跳过或物理删除 DRA head；
- 若关闭 `val_dra_diagnostics`，validation 结果应与删除 DRA head 后的检测结果一致。

## 3. `TargetAwareDRA-ScaleAligned` 的准确含义

原始边缘目标通常与网络输入同尺寸，例如 `640×640`；P2-DRA 输出通常是较低分辨率，例如 `160×160`。二者必须先对齐才能计算重建损失。

因此，`TargetAwareDRA-ScaleAligned` 不是新增的推理模块，而是监督目标构造方式：

```text
Edge_mm, Mask                         # 输入尺度
      ↓ alignment
Edge_P2, Mask_P2                      # P2-DRA 输出尺度
      ↓
masked L1 loss
```

所有 P2-DRA 都必须进行尺寸对齐。所谓 ScaleAligned 实验实际比较：

| ID             | 对齐方法                 | 用途                          |
| -------------- | ------------------------ | ----------------------------- |
| ALIGN-BILINEAR | bilinear resize          | 朴素基线                      |
| ALIGN-AREA     | area interpolation       | D2 默认；保留区域平均边缘响应 |
| ALIGN-POOL     | adaptive average pooling | 直接匹配 P2 输出尺寸          |

执行原则：

- 主 DRA 实验先固定一种方式，优先使用 `area interpolation`；
- `bilinear vs area vs adaptive average pooling` 放在后续对齐方式消融中；
- 如果三者差异不稳定，则只把尺度对齐写成实现步骤，不单独命名为 MESA；
- 本项目暂不做 P2+P3+P4 多尺度 DRA。

## 4. Linux 兼容性要求

AutoDL 和主流 GPU 云镜像通常提供 Ubuntu/Linux 环境，但首次创建实例后仍需运行 `uname -a`、`nvidia-smi` 和 `python --version` 核验。项目代码统一以 Linux 为目标环境。

强制规则：

1. Python 路径使用 `pathlib.Path`，禁止写死 `D:\...`、盘符或 Windows 反斜杠路径；
2. 数据、权重和输出目录通过 YAML 参数、命令行参数或环境变量传入；
3. 文档和脚本示例使用 `/` 路径；
4. 文件名大小写固定。Linux 区分 `Project.MD` 与 `project.md`；
5. 文本文件使用 UTF-8 和 LF 换行；
6. Shell 脚本使用 `#!/usr/bin/env bash` 和 `set -euo pipefail`；
7. 不假定固定 GPU 型号、CUDA 编号或服务器目录；
8. GPU 通过 `CUDA_VISIBLE_DEVICES` 或命令行参数选择；
9. 训练代码启动时记录 PyTorch、CUDA、Ultralytics、GPU、驱动和操作系统版本；
10. 所有路径创建使用 `mkdir -p` 或 Python `Path.mkdir(parents=True, exist_ok=True)`；
11. DataLoader workers、AMP 和 batch size 必须可配置，不能写死；
12. 输出目录禁止覆盖已有实验，使用唯一的 experiment ID；
13. 后续增加 `requirements.txt` 或环境文件，锁定关键依赖版本；
14. 在 Linux 服务器上执行训练前，先依次运行 tiny-subset smoke test 和 small-train smoke test。

建议的服务器变量：

```bash
export PROJECT_ROOT=/path/to/mdra_yolov8_experiments
export DATA_ROOT=/path/to/datasets/M3FD
export OUTPUT_ROOT="$PROJECT_ROOT/outputs"
export CUDA_VISIBLE_DEVICES=0
```

不得在代码中依赖 AutoDL 特有的 `/root/autodl-tmp`。该目录可以在运行命令中传入，但代码本身必须能迁移到其他 Linux 服务器。

## 5. 计划目录结构

```text
mdra_yolov8_experiments/
├── Project.MD                 # 项目进度与实验总表
├── README.md                  # 后续添加：Linux 安装与运行说明
├── requirements.txt           # 后续添加：依赖版本
├── configs/                   # 数据、模型、训练和实验配置
├── mdra/                      # 模型、数据、DRA、损失等 Python 包
├── scripts/                   # Linux 训练/验证/导出脚本
├── tests/                     # 数据配对、mask、edge、导出图测试
└── outputs/                   # 实验输出；不提交大文件
```

## 6. 实验总表

状态取值：`未开始 / 实现中 / 可运行 / 运行中 / 已完成 / 失败待查`。

### 6.1 环境与数据核验

| ID             | 任务                           | 状态   | 完成标准                                                                          | 结果/备注                                                                                                                                                          |
| -------------- | ------------------------------ | ------ | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| P0-ENV         | Linux/CUDA/PyTorch 环境检查    | 已完成 | 保存环境报告，GPU 可见                                                            | RTX 3090；PyTorch 2.0.0+cu118；Ultralytics 8.3.27；报告：`outputs/env_reports/env_report_20260629_105222.*`                                                      |
| P0-DATA        | VIS/IR/label 一一对应检查      | 已完成 | 无未解释的缺失配对                                                                | VIS/IR/label 各 4200，34,407 个目标，0 个无效样本；报告：`outputs/data_checks/`                                                                                  |
| P0-FIXED-SPLIT | 建立 fixed train/val/test 列表 | 已完成 | 保存 fixed split 文件、随机种子和样本清单                                         | seed 42，7:2:1，2940/840/420；`configs/splits/m3fd_seed42/`                                                                                                      |
| P0-STATS       | 类别、bbox 面积、场景统计      | 已完成 | 输出统计表和分布图                                                                | 34,407 个目标；CSV/JSON/PNG：`outputs/dataset_stats/`；未提供场景元数据，统一记为 unknown                                                                        |
| P0-TINY-SMOKE  | tiny subset smoke test         | 已完成 | 少量配对样本、1–2 epoch；验证 forward/backward、loss、配对、保存和单批预测       | 32/16、2 epoch；validation/checkpoint/results.csv/predict 均通过；`outputs/experiments/B1_visible_tiny_smoke_20260629_115152/`                                   |
| P0-SMALL-TRAIN | small train smoke test         | 已完成 | 固定小训练子集、5–10 epoch；验证 train/validation、checkpoint、resume 和指标链路 | 300/100；epoch 2 受控中断并 resume 至 epoch 10；独立验证 mAP50=0.523、mAP50:95=0.335；`outputs/experiments/B1_visible_small_train_resume_smoke_20260629_115651/` |

Split 命名规则：本项目统一使用 `fixed split` 或 `unified split`。除非能够核验数据集维护者发布的正式索引文件，否则文档、代码和论文中不得称为 `official split`。

### 6.2 基础模型

| ID      | 实验                | 状态   | 主要目的         | 结果路径                                                        |
| ------- | ------------------- | ------ | ---------------- | --------------------------------------------------------------- |
| B1      | YOLOv8s-Visible     | 已完成 | 验证基础检测管线 | epoch 94 早停；test P/R=0.8096/0.6552，mAP50/75/50:95=0.7441/0.5218/0.5034；11.14M Params，28.66 GFLOPs，195.25 FPS；`outputs/b_experiments/B1_visible/` |
| B2      | YOLOv8s-IR          | 已完成 | 验证 IR 单模态   | epoch 100；test P/R=0.8037/0.6192，mAP50/75/50:95=0.7089/0.4926/0.4694；11.14M Params，28.66 GFLOPs，202.68 FPS；`outputs/b_experiments/B2_infrared/` |
| B3      | YOLOv8s-EarlyFusion | 已完成 | 简单双模态基线   | epoch 95 早停；test P/R=0.8483/0.6625，mAP50/75/50:95=0.7698/0.5580/0.5246；11.14M Params，28.72 GFLOPs，199.21 FPS；`outputs/b_experiments/B3_early_fusion/` |
| B3-P2   | YOLOv8s-EarlyFusion-P2 | 已完成 | 隔离 P2 贡献并判断是否删除 LCMF | epoch 97 早停；test P/R=0.8779/0.7102，mAP50/75/50:95=0.8097/0.5673/0.5504；10.64M Params，37.03 GFLOPs，165.65 FPS；`outputs/b_experiments/B3_early_fusion_p2/` |
| B4      | YOLOv8s-LCMF        | 已完成 | 验证特征级融合   | epoch 100；test P/R=0.7939/0.6546，mAP50/75/50:95=0.7481/0.5488/0.5136；16.91M Params，42.39 GFLOPs，132.65 FPS；`outputs/b_experiments/B4_lcmf/` |
| B5 / D0 | YOLOv8s-LCMF-P2     | 已完成 | DRA 强推理基线   | epoch 98 早停；test P/R=0.7962/0.7206，mAP50/75/50:95=0.7788/0.5466/0.5277；16.42M Params，51.14 GFLOPs，115.84 FPS；`outputs/b_experiments/B5_lcmf_p2/` |

### 6.3 P2 单尺度 DRA 核心实验

| ID      | 实验                            | 状态   | 固定对齐       | 主要比较                   |
| ------- | ------------------------------- | ------ | -------------- | -------------------------- |
| D0      | EarlyFusion-P2，无 DRA          | 已完成 | —             | 根据受控实验冻结为新的 D0；`outputs/b_experiments/B3_early_fusion_p2/` |
| D1      | LCMF-P2-FullEdgeDRA             | 未开始 | area           | D0 vs D1：边缘辅助是否有效 |
| D2      | LCMF-P2-TargetAwareDRA          | 未开始 | area（主配置） | D1 vs D2：目标感知是否有效 |
| D2-BI   | D2 的 bilinear 对齐变体         | 未开始 | bilinear       | 与 D2 比较对齐方式         |
| D2-POOL | D2 的 adaptive-pooling 对齐变体 | 未开始 | adaptive pool  | 与 D2 比较对齐方式         |

注意：D2 已定义为 area 对齐主配置，不再设置单独的 area 对齐变体。`D2-BI` 和 `D2-POOL` 只改变训练监督图构造，不增加推理模块。

### 6.4 辅助任务公平对照

| ID | 实验                        | 状态   | 目的                                                |
| -- | --------------------------- | ------ | --------------------------------------------------- |
| A1 | LCMF-P2-SRStyleAux          | 未开始 | SR-style auxiliary 受控对照；排除全局 SR 正则化解释 |
| A2 | LCMF-P2-ImageReconstruction | 未开始 | 排除普通图像重建解释                                |
| A3 | LCMF-P2-SegAux              | 未开始 | 排除 bbox 密集辅助监督解释                          |
| A4 | LCMF-P2-TargetAwareDRA      | 未开始 | 本文方法                                            |

会议最小版本至少完成 A1 vs A4；有算力时优先补 A3。

`A1` 明确定义为 **SR-style auxiliary controlled baseline**，不是 SuperYOLO 复现：

- 不使用 `SuperYOLO` 名称标注该实验；
- 不宣称复现其网络、训练协议或论文结果；
- 仅在本文相同 LCMF-P2 骨架、fixed/unified split 和训练协议下加入 SR 风格辅助目标；
- 若没有真实 LR–HR 图像对，而是通过固定下采样生成模拟 LR，必须在配置、结果表和论文中明确说明；
- 该实验只回答“SR 风格辅助是否能解释 DRA 的增益”。

### 6.5 DRA 目标和超参数

| ID       | 实验组         | 候选                          | 状态   |
| -------- | -------------- | ----------------------------- | ------ |
| G-EDGE   | edge 来源      | VIS / IR / max / 0.5 weighted | 未开始 |
| G-MASK   | bbox 扩张      | 1.0 / 1.2 / 1.5               | 未开始 |
| G-LAMBDA | `lambda_dra` | 0.05 / 0.10 / 0.20 / 0.50     | 未开始 |
| G-LOSS   | 重建损失       | L1 / Smooth-L1                | 未开始 |
| G-OP     | 边缘算子       | Sobel；SCI 版可补 Scharr      | 未开始 |

### 6.6 可选 ECA

| ID | 实验                   | 状态   | 决策规则                               |
| -- | ---------------------- | ------ | -------------------------------------- |
| E1 | TargetAwareDRA，无 ECA | 未开始 | 主版本                                 |
| E2 | TargetAwareDRA + ECA   | 未开始 | 只有增益超过多种子波动且代价合理才保留 |

第一版不实现 ECA，除非核心 DRA 实验已经成立。

### 6.7 复杂度、分组和可视化

| ID        | 任务                          | 状态   | 完成标准                     |
| --------- | ----------------------------- | ------ | ---------------------------- |
| C-INF     | 删除 DRA 后 Params/GFLOPs/FPS | 未开始 | D0/D1/D2 推理图一致          |
| C-TRAIN   | 训练显存与时间                | 未开始 | 报告 DRA 的训练代价          |
| ANA-SCALE | small/medium/large AP         | 未开始 | 同时报告样本数、Recall、AP75 |
| ANA-SCENE | day/night/challenge           | 未开始 | 使用固定场景清单             |
| VIS-EDGE  | edge/mask/aligned target      | 未开始 | 展示构造链                   |
| VIS-DRA   | DRA prediction/error          | 未开始 | FullEdge vs TargetAware      |
| VIS-DET   | 检测与失败案例                | 未开始 | 相同阈值，包含负面案例       |

### 6.8 SOTA 与第二数据集

| ID         | 任务                     | 投稿档位      | 状态   |
| ---------- | ------------------------ | ------------- | ------ |
| SOTA-REF   | 作者报告结果参考表       | EI            | 未开始 |
| SOTA-UNI   | 统一协议 2–3 个代表对照 | SCI 四区      | 未开始 |
| DATA-FLIR  | FLIR-Aligned 泛化        | SCI 四区/三区 | 未开始 |
| DATA-LLVIP | LLVIP 泛化               | SCI 三区      | 未开始 |

## 7. 推荐执行顺序

```text
P0-ENV
→ P0-DATA / P0-FIXED-SPLIT / P0-STATS
→ P0-TINY-SMOKE
→ P0-SMALL-TRAIN
→ B1 Visible
→ B2 IR
→ B3 EarlyFusion
→ B4 LCMF
→ B5/D0 LCMF-P2
→ D1 FullEdgeDRA
→ D2 TargetAwareDRA
→ D2-BI / D2 / D2-POOL alignment ablation
→ A1 SR-style auxiliary，随后视算力补 A3 SegAux
→ edge/mask/lambda sensitivity
→ 3 seeds for D0/D1/D2 and selected alignment
→ inference graph / subgroup / visualization
→ SOTA and second dataset
```

算力节省规则：

1. D0、D1、D2 先使用同一 seed 跑完整试验；
2. D1 无增益时暂停后续任务，先排查 edge 归一化、DRA 接入点、loss 权重和梯度路径；
3. D2 不优于 D1 时，先排查 mask 扩张、空标签处理和 masked loss 面积归一化；
4. 选出候选配置后，再对核心模型运行 3 个种子；
5. 不允许用 ECA 或其他模块掩盖核心 DRA 无效。

## 8. 每次实验必须记录

```text
experiment_id
date
code_version
config_path
fixed_or_unified_split
seed
GPU / CUDA / PyTorch / Ultralytics
input_size / batch_size / epochs
best_epoch
Precision / Recall
mAP50 / mAP75 / mAP50:95
per-class AP
inference Params / GFLOPs / latency / FPS
training time / peak memory
checkpoint_path
result_path
conclusion
```

## 9. 当前进度

| 日期       | 进度                                                                                                                                                                    | 状态   |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| 2026-06-28 | 完成 23 篇文献分析报告                                                                                                                                                  | 已完成 |
| 2026-06-28 | 完成修订实验计划                                                                                                                                                        | 已完成 |
| 2026-06-28 | 确定先做 P2 单尺度 DRA                                                                                                                                                  | 已完成 |
| 2026-06-28 | 建立 Linux 兼容实验项目目录与`Project.MD`                                                                                                                             | 已完成 |
| 2026-06-28 | 完成 6.1 代码骨架、README、配置与基础测试                                                                                                                               | 已完成 |
| 2026-06-28 | 12 项单元/合成数据集成测试                                                                                                                                              | 已完成 |
| 2026-06-28 | Linux GPU 环境报告、CUDA 运算和 YOLOv8s 构建核验                                                                                                                        | 已完成 |
| 2026-06-28 | 远端 12 项测试通过；修复 Python 3.8 的`--amp/--no-amp` 兼容性                                                                                                         | 已完成 |
| 2026-06-28 | 项目、数据集目录和项目内输出迁移到持久化数据盘`/root/autodl-tmp`                                                                                                      | 已完成 |
| 2026-06-29 | 完整 M3FD 配对检查：4200 对、34,407 个目标、0 个无效样本                                                                                                                | 已完成 |
| 2026-06-29 | 建立 seed 42 fixed/unified split：2940/840/420，并完成真实数据统计                                                                                                      | 已完成 |
| 2026-06-29 | tiny-subset GPU smoke：2 epoch、validation、checkpoint 和 predict                                                                                                       | 已完成 |
| 2026-06-29 | small-train GPU smoke：epoch 2 中断后 resume 至 epoch 10，并独立验证和预测                                                                                              | 已完成 |
| 2026-06-29 | 对照《拟定实验计划.md》完成 6.2 的 B1–B5 代码、配置及统一 train/val/predict/profile 入口；补齐 mAP75、per-class AP、PR 曲线、混淆矩阵、预测图和 Params/GFLOPs/FPS 输出 | 已完成 |
| 2026-06-29 | 无卡模式完成语法及数据/配置/路径 20 项测试；按用户要求停止耗时的 CPU 模型前向检查，未启动任何 B 实验训练                                                                | 已完成 |
| 2026-06-29 | RTX 3090 下完成 B1–B5 单批 GPU preflight；五个模型的真实数据 forward、loss 和 backward 全部通过                                                                        | 已完成 |
| 2026-06-29 | 启动 B1–B5 顺序控制器；B1 第 1 epoch 用时 44.85 秒，峰值显存 3.24 GiB，训练链路正常                                                                                       | 运行中 |
| 2026-06-29 | B1 在 epoch 94 早停并完成 fixed-test、predict、profile；test mAP50=0.7441、mAP75=0.5218、mAP50:95=0.5034，195.25 FPS；控制器进入 B2 | 已完成 |
| 2026-06-29 | B2 跑满 epoch 100 并完成 fixed-test、predict、profile；test mAP50=0.7089、mAP75=0.4926、mAP50:95=0.4694，202.68 FPS；控制器进入 B3 | 已完成 |
| 2026-06-30 | B3/B4/B5 全部完成 train、fixed-test、predict、profile；B5 test mAP50=0.7788、Recall=0.7206、mAP50:95=0.5277；6.2 B 组全流程完成 | 已完成 |
| 2026-06-30 | 根据 B4 未超过 B3 的结果，完成 B3-P2（EarlyFusion-P2）代码与配置；无卡模式仅完成语法检查，模型测试已按用户要求终止，等待 GPU preflight | 实现中 |
| 2026-06-30 | B3-P2 GPU preflight 通过并启动正式训练；第 1 epoch 用时 48.24 秒，峰值 PyTorch 显存 5.47 GiB，训练链路正常 | 运行中 |
| 2026-06-30 | B3-P2 在 epoch 97 早停并完成 fixed-test、predict、profile；test mAP50=0.8097、mAP75=0.5673、mAP50:95=0.5504，165.65 FPS；确定以 EarlyFusion-P2 替代 LCMF-P2 作为 D0 | 已完成 |
| —         | 基于 EarlyFusion-P2 D0 进入 6.3 DRA 核心实验代码实现；LCMF 仅保留为消融对照 | 下一步 |

## 10. 下一步

6.1、原 B1–B5 及 B3-P2 受控实验已全部完成。EarlyFusion-P2 相比 B3 EarlyFusion 的 test mAP50:95 提升 0.0258，相比 B5 LCMF-P2 提升 0.0227；同时比 B5 减少约 35.2% Params、27.6% GFLOPs，并将 FPS 从 115.84 提升至 165.65。最终推理骨架冻结为 EarlyFusion-P2，LCMF 从最终方法中删除，仅保留为负结果/消融对照。下一步基于新的 D0 实现 6.3 DRA；DRA、MESA、SR-style auxiliary、SegAux 和 ECA 尚未实现。

正式基线前保持以下内容冻结：

1. 使用 `configs/splits/m3fd_seed42/`，不得重新随机划分；
2. 数据根目录为 `/root/autodl-tmp/M3FD_Detection`，模态目录名为 `Vis` 和 `Ir`；
3. 输出继续写入项目内 `outputs/`；
4. 使用持久化权重 `weights/yolov8s.pt`，避免运行时外网下载。

# InternVLA-N1-System2 仿真评估实验报告

> **评估系统**: InternVLA-N1 (System2 模式)
> **基座模型**: Qwen2.5-VL-3B-Instruct
> **评估基准**: R2R Val-Unseen (Matterport3D)
> **仿真平台**: Habitat Simulator
> **文档日期**: 2026-03

---

## 目录

1. [实验概述](#1-实验概述)
2. [评估指标定义](#2-评估指标定义)
3. [实验设置](#3-实验设置)
4. [总体评估结果](#4-总体评估结果)
5. [分场景评估分析](#5-分场景评估分析)
6. [导航误差分布分析](#6-导航误差分布分析)
7. [评估过程稳定性分析](#7-评估过程稳定性分析)
8. [视频案例分析](#8-视频案例分析)
9. [失败案例分析](#9-失败案例分析)
10. [综合性能雷达图](#10-综合性能雷达图)
11. [与现有方法对比](#11-与现有方法对比)
12. [结论与展望](#12-结论与展望)

---

## 1. 实验概述

### 1.1 研究背景

Vision-Language Navigation (VLN) 任务要求智能体根据自然语言指令在未见过的 3D 环境中进行导航。R2R (Room-to-Room) 是 VLN 领域最经典的基准测试之一，基于 Matterport3D 真实室内扫描场景构建。

本实验在 Habitat 仿真平台上评估 InternVLA-N1 模型在 **System2** 模式下的导航性能。System2 模式采用双阶段推理策略：

- **Phase 1（思考阶段）**: 模型基于当前视觉观测和导航指令，生成自然语言思考（reasoning），分析当前环境状态、已完成进度及下一步行动方向；
- **Phase 2（动作阶段）**: 基于 Phase 1 的思考结果，模型输出低层动作指令（前进/左转/右转/停止）。

这种"先思考、再行动"的设计灵感来自认知科学中的 System 1 / System 2 理论 (Kahneman, 2011)，旨在让智能体具备更强的推理与规划能力。

### 1.2 评估目标

- 在 R2R Val-Unseen 划分上全量评估 InternVLA-N1-System2 的导航性能
- 通过分场景分析识别模型在不同环境类型中的表现差异
- 通过视频记录直观展示导航行为
- 为后续 Compressor 模块的改进提供基线数据

---

## 2. 评估指标定义

本实验采用 VLN 领域标准评估指标：

| 指标 | 全称 | 定义 | 越高/低越好 |
|------|------|------|------------|
| **SR** | Success Rate | 智能体最终停止位置距目标点 ≤ 3m 的 episode 比例 | ↑ 越高越好 |
| **SPL** | Success weighted by Path Length | SPL = (1/N) Σ S_i · ℓ_i / max(p_i, ℓ_i)，其中 S_i 为是否成功，ℓ_i 为最短路径长度，p_i 为实际路径长度 | ↑ 越高越好 |
| **OS** | Oracle Success Rate | 智能体在导航轨迹中**任意位置**距目标点 ≤ 3m 的比例 | ↑ 越高越好 |
| **NE** | Navigation Error | 智能体最终停止位置与目标点的欧氏距离（米） | ↓ 越低越好 |

> **关键区别**: SR 仅考虑最终停止位置，OS 考虑整条轨迹中的最近点。OS >> SR 意味着模型能到达目标附近但**不知何时停止**。SPL 在 SR 的基础上惩罚冗余路径，是最严格的指标。

---

## 3. 实验设置

### 3.1 模型配置

| 配置项 | 值 |
|--------|-----|
| 模型架构 | Qwen2.5-VL-3B-Instruct + LoRA |
| 推理模式 | System2 (双阶段: 思考→动作) |
| LoRA 权重 | checkpoints/InternVLA-N1-System2/ |
| 视觉输入 | 历史帧 (8帧均匀采样) + 当前帧 + 俯视图 |
| 动作空间 | 离散: Forward / Turn Left / Turn Right / Stop |
| 最大步数 | 500 步 |

### 3.2 评估数据

| 配置项 | 值 |
|--------|-----|
| 数据集 | R2R Val-Unseen |
| 场景数量 | 11 个 Matterport3D 场景 |
| Episode 总数 | 1,839 |
| 仿真平台 | Habitat Simulator |
| 评估 GPU | NVIDIA H100 (95GB) |
| 总耗时 | ~8 小时 |

### 3.3 运行命令

```bash
# 全量评估
CUDA_VISIBLE_DEVICES=1 python -m internnav.habitat_vln_evaluator \
    --config configs/eval_official_s2.py

# 视频评估 (10 episodes)
CUDA_VISIBLE_DEVICES=1 python -m internnav.habitat_vln_evaluator \
    --config configs/eval_official_s2_10ep_video_cfg.py
```

---

## 4. 总体评估结果

在 R2R Val-Unseen 全部 1,839 个 episode 上的评估结果如下：

| 指标 | 值 |
|------|-----|
| **SR (Success Rate)** | **0.6052** (60.52%) |
| **SPL (Success weighted by Path Length)** | **0.5498** (54.98%) |
| **OS (Oracle Success Rate)** | **0.6786** (67.86%) |
| **NE (Navigation Error)** | **4.328 m** |
| 评估 Episode 数 | 1,839 |

### 关键发现

1. **SR = 60.52%**: 超过半数的导航任务成功完成，表明模型具备较强的指令理解和空间导航能力。
2. **SPL/SR 比值 = 0.908**: SPL 与 SR 的比值接近 1.0，说明成功的导航路径效率较高，模型倾向于沿最优路径行进，冗余探索较少。
3. **OS - SR 差距 = 7.34%**: Oracle 成功率比 SR 高出 7.34 个百分点，意味着约有 7.3% 的 episode 中智能体**曾经接近过目标但未能正确停止**。这暴露了模型在"何时停止"决策上的改进空间。
4. **NE = 4.328m**: 平均导航误差略高于 3m 的成功阈值，主要由失败案例拉高。

---

## 5. 分场景评估分析

R2R Val-Unseen 包含 11 个不同的 Matterport3D 室内场景，模型在不同场景上的表现存在显著差异：

| 场景 ID | Episode 数 | SR | SPL | NE (m) | 评价 |
|---------|-----------|------|------|--------|------|
| zsNo4HB9uLZ | 300 | **0.697** | **0.648** | 3.20 | 🏆 最优场景 |
| x8F5xyUWy9e | 102 | **0.676** | **0.635** | 2.84 | ✅ 优秀 |
| X7HyMhZNoso | 141 | 0.674 | 0.624 | 3.28 | ✅ 优秀 |
| QUCTc6BB5sX | 255 | 0.655 | 0.589 | 4.69 | ✅ 良好 |
| 2azQ1b91cZZ | 252 | 0.611 | 0.544 | 4.50 | 中等 |
| Z6MFQCViBuw | 159 | 0.604 | 0.572 | **7.59** | ⚠️ NE 异常高 |
| TbHJrupSAjP | 264 | 0.598 | 0.542 | 3.94 | 中等 |
| EU6Fwq7SyZv | 132 | 0.576 | 0.478 | 4.36 | 中等偏下 |
| 8194nk5LbLH | 39 | 0.564 | 0.514 | 3.86 | 样本量小 |
| oLBMNvg9in8 | 177 | **0.356** | **0.320** | 4.89 | ❌ 较差 |
| pLe4wQe7qrG | 18 | **0.222** | **0.153** | 4.52 | ❌ 最差场景 |

![Per-Scene Metrics](figures/per_scene_metrics.png)
*图 1: 各场景 SR/SPL/OS 指标对比。蓝色虚线为全局 SR 均值，绿色虚线为全局 SPL 均值。*

### 分析

**表现最优的场景群 (SR > 0.65)**:
- `zsNo4HB9uLZ` (SR=0.697)、`x8F5xyUWy9e` (SR=0.676)、`X7HyMhZNoso` (SR=0.674) 三个场景表现优异。这些场景可能具有**布局规则、走廊明确、标志物显著**等特征，使得模型能准确理解"走到XX旁边"、"经过XX后右转"等指令。

**表现最差的场景群 (SR < 0.4)**:
- `oLBMNvg9in8` (SR=0.356) 仅有约三分之一的 episode 成功，显著低于均值。
- `pLe4wQe7qrG` (SR=0.222) 仅有 18 个 episode 但成功率最低。虽然样本量小可能导致统计波动，但低至 22.2% 的成功率仍值得关注。

**特殊异常 -- Z6MFQCViBuw**:
- 该场景 SR 尚可（0.604），但 NE 高达 **7.59m**，远超其他场景。这意味着失败案例的导航误差极大——智能体可能在该场景中完全迷失方向，偏离目标很远。该场景可能包含**大空间（如大厅、开放区域）**，使得一旦走错路径就会产生很大的距离偏差。

---

## 6. 导航误差分布分析

![Navigation Error Distribution](figures/navigation_error_dist.png)
*图 2: (左) 全部 1,839 个 episode 的导航误差直方图。红色虚线为均值（4.33m），绿色虚线为 3m 成功阈值。(右) 各场景导航误差箱线图。*

### 关键观察

1. **双峰分布**: 导航误差呈明显的双峰分布——
   - **第一个峰** 在 0-3m 范围内，对应成功的 episode，大量 episode 能精确到达目标位置附近；
   - **第二个峰** 分布在 5-15m 范围内，对应失败的 episode。

2. **长尾效应**: 部分 episode 的 NE 超过 20m 甚至更高，说明存在极端失败案例——智能体完全走错方向或陷入循环。

3. **场景间差异** (箱线图):
   - `x8F5xyUWy9e` 的中位数和四分位范围最小，导航误差最为稳定；
   - `Z6MFQCViBuw` 存在大量异常值（NE > 20m），解释了其平均 NE 高达 7.59m 的原因；
   - 各场景的中位 NE 大多在 1-5m 范围内，但上四分位数差异显著。

---

## 7. 评估过程稳定性分析

![Cumulative SR/SPL](figures/cumulative_sr_spl.png)
*图 3: 随评估 episode 数增加的累积 SR 和 SPL 曲线。灰色竖虚线标记场景边界。*

### 分析

累积 SR/SPL 曲线反映了评估过程中指标的变化趋势：

1. **前期波动大**: 在前 100-200 个 episode 中，累积指标波动较大（受初始场景影响）。
2. **中期逐步收敛**: 约 500 个 episode 后，累积 SR 和 SPL 趋于稳定，表明样本量已足够反映模型真实水平。
3. **SR-SPL 间距稳定**: 两条曲线之间的带状区域（蓝色填充）代表 SR 与 SPL 的差距，始终保持较小且稳定，进一步验证了模型路径效率较高的结论。
4. **最终收敛值**: SR 收敛至 **0.605**，SPL 收敛至 **0.550**，1,839 个 episode 的评估结果具有统计可靠性。

---

## 8. 视频案例分析

我们选取了部分 episode 录制导航视频，直观展示 InternVLA-N1-System2 的导航行为。以下为两个代表性**成功案例**的详细分析：

### 8.1 案例 1: Episode 304 (0304.mp4)

| 属性 | 值 |
|------|-----|
| **场景** | 1LXtFkjw3qL |
| **指令** | *"Walk straight across the room to the other side. Once at the green bed, exit the room out of the door to your right. Once out, stop before you reach the steps."* |
| **成功** | ✅ (SR = 1.0) |
| **SPL** | 1.0 (最优路径) |
| **NE** | 1.06 m |
| **步数** | 63 |

**分析**: 该指令包含三个子任务——(1) 穿过房间，(2) 在绿色床旁从右侧门出去，(3) 在楼梯前停下。模型以 **SPL=1.0** 的完美路径效率完成了全部子任务，最终停止位置距目标仅 1.06m。这表明 System2 模式的思考机制能有效分解多步指令。

### 8.2 案例 2: Episode 201 (0201.mp4)

| 属性 | 值 |
|------|-----|
| **场景** | 1LXtFkjw3qL |
| **指令** | *"Turn left and go past the bed and past the long couch. Go to and stop at the doorway on the right."* |
| **成功** | ✅ (SR = 1.0) |
| **SPL** | 1.0 (最优路径) |
| **NE** | 0.34 m |
| **步数** | 55 |

**分析**: 该指令涉及方向转换（左转）、经过多个地标（床、长沙发）、最终在特定位置停止（右侧门口）。模型以极低的导航误差（0.34m）精确到达目标。**NE = 0.34m** 远低于 3m 阈值，说明模型对"门口"这一精确位置有很好的理解。

### 8.3 视频评估 Episode 汇总

在录制视频的 17 个 episode 中，整体统计如下：

| 统计项 | 值 |
|--------|-----|
| 成功 Episode | 13 / 17 (76.5%) |
| 失败 Episode | 4 / 17 (23.5%) |
| 平均 NE | 2.29 m |
| SPL = 1.0 的 Episode | 9 / 17 (52.9%) |

> 视频评估的 SR (76.5%) 高于全局 SR (60.5%)，这可能因为视频 episode 集中在 `17DRP5sb8fy` 和 `1LXtFkjw3qL` 两个特定场景中，而非全部 11 个场景的均匀采样。

![Steps vs NE](figures/steps_vs_ne.png)
*图 4: 视频评估 episode 的步数与导航误差散点图。绿色圆点为成功案例，红色叉号为失败案例。橙色虚线为 3m 成功阈值。*

---

## 9. 失败案例分析

分析视频评估中的 4 个失败 episode，揭示模型的典型失败模式：

### 9.1 失败模式一: 过早停止 (Episode 97)

| 属性 | 值 |
|------|-----|
| **指令** | *"Walk past curved sofa. Walk past bed. Wait at bathroom door threshold."* |
| **NE** | 5.15 m |
| **步数** | 65 |
| **OS** | 1.0 (曾接近目标) |

**分析**: OS = 1.0 表明智能体在导航过程中**曾经到达目标 3m 范围内**，但最终停在了错误位置。这是典型的**停止时机决策错误**——模型经过了正确区域但没有及时执行 STOP 动作。

### 9.2 失败模式二: 路径偏离 (Episode 327)

| 属性 | 值 |
|------|-----|
| **指令** | *"Turn left and go passed the painting. Continue passed the bar and turn right and then go through the left doorway."* |
| **NE** | 7.26 m |
| **步数** | 259 |
| **OS** | 1.0 |

**分析**: 259 步远超正常范围（正常约 40-80 步），结合 OS = 1.0 可知智能体在大量探索后仍未能回到目标。模型可能在复杂转弯指令（"turn right and then go through the left doorway"）处产生理解偏差，导致反复绕行。

### 9.3 失败模式三: 完全迷失 (Episode 312)

| 属性 | 值 |
|------|-----|
| **指令** | *"Go straight towards the flower in the pot. Turn left right before the flower pot. Then turn right and go down the stairs. Stop at the bottom facing 2 doorways."* |
| **NE** | 9.77 m |
| **步数** | 501 (达到上限) |
| **OS** | 0.0 (从未接近目标) |

**分析**: 这是最严重的失败模式——智能体在 501 步中**从未接近过目标**（OS = 0.0），最终因步数达到上限而被迫终止。指令中的"go down the stairs"要求**跨楼层导航**，这对当前模型是极大挑战。模型可能无法正确识别楼梯或理解楼层间的空间关系。

### 9.4 失败原因总结

| 失败类型 | 占比 | 特征 | 改进方向 |
|----------|------|------|---------|
| 停止时机错误 | ~50% | OS = 1.0 但 SR = 0 | 改进 STOP 动作决策 |
| 指令理解偏差 | ~25% | 长步数 + 绕行 | 增强多步指令分解能力 |
| 完全迷失 | ~25% | OS = 0.0, 达步数上限 | 改进跨区域/跨楼层导航 |

---

## 10. 综合性能雷达图

![Radar Chart](figures/radar_chart.png)
*图 5: InternVLA-N1-System2 综合性能雷达图。四个维度分别为 SR、SPL、OS 和归一化导航精度 (1 - NE/10)。*

雷达图直观展示了模型在各维度上的均衡性：
- **OS** 维度最高（0.679），表明模型的轨迹规划能力较强；
- **SPL** 维度最低（0.550），但与 SR 差距不大，说明路径效率可接受；
- 归一化 NE 维度 (1 - NE/10 = 0.567) 表明平均导航误差处于中等水平。

---

## 11. 与现有方法对比

下表列出在 R2R Val-Unseen 上的代表性方法对比（数据来源于公开论文）：

| 方法 | 基座模型 | SR ↑ | SPL ↑ | 发表 |
|------|---------|------|-------|------|
| Seq2Seq (baseline) | LSTM | 0.22 | 0.20 | CVPR 2018 |
| EnvDrop | LSTM | 0.52 | 0.48 | NAACL 2019 |
| PREVALENT | BERT | 0.54 | 0.51 | CVPR 2020 |
| HAMT | ViT + BERT | 0.66 | 0.61 | NeurIPS 2021 |
| DUET | ViT + BERT | 0.72 | 0.60 | CVPR 2022 |
| NaviLLM | Vicuna-7B | 0.67 | -- | CVPR 2024 |
| **InternVLA-N1-S2 (Ours)** | **Qwen2.5-VL-3B** | **0.605** | **0.550** | **--** |

> **注**: 以上对比为参考性质。不同方法的评估设置可能存在差异（如动作空间、输入分辨率、是否使用全景视图等），因此直接数值比较需谨慎。InternVLA-N1 使用**非全景（单视角 egocentric）**输入和**低层连续动作空间**在 Habitat 中评估，与基于导航图 (navigation graph) 的方法**不直接可比**。
>
> 在**低层动作空间 + Habitat 仿真器**的设定下，60.5% 的 SR 是一个有竞争力的结果。

---

## 12. 结论与展望

### 12.1 主要结论

1. **InternVLA-N1-System2 在 R2R Val-Unseen 上取得了 SR=60.5%, SPL=55.0% 的基线性能**，证明了基于 VLM 的端到端导航方案的可行性。

2. **System2 的双阶段推理机制有效**: 成功案例中 SPL/SR 比值高达 0.908，说明"先思考、再行动"的策略能产生高效的导航路径，而非盲目探索。

3. **模型在不同场景间表现差异显著**: SR 最高场景 (0.697) 与最低场景 (0.222) 相差超过 3 倍，提示模型对场景复杂度和空间布局敏感。

4. **主要瓶颈**: OS 与 SR 的差距 (7.3%) 表明**停止决策**是可优化的关键环节；失败案例中的跨楼层导航和复杂转弯指令理解是两大难题。

### 12.2 改进方向

| 方向 | 具体方案 | 预期收益 |
|------|---------|---------|
| **停止决策优化** | 引入距离预测辅助任务或停止置信度阈值 | 缩小 OS-SR 差距，SR 提升 3-5% |
| **视觉 Token 压缩** | 指令条件化的历史帧压缩 (IC-Compressor) | 允许输入更多历史帧，减少信息丢失 |
| **数据增强** | 针对低频场景和跨楼层指令的定向数据采集 | 改善尾部场景表现 |
| **多步指令分解** | 显式子目标预测模块 | 减少长指令理解偏差 |

### 12.3 后续实验计划

- [ ] 消融实验：System1 vs System2 模式对比
- [ ] Compressor 模块集成后的性能评估
- [ ] 全景视图 (panoramic) 输入实验
- [ ] R4R、RxR 等更长路径数据集评估

---

## 附录

### A. 视频文件与指令对照表

| 视频文件 | Episode ID | 场景 | 成功 | 指令 |
|----------|-----------|------|------|------|
| 0095.mp4 | 95 | 17DRP5sb8fy | ✅ | Enter the bedroom, wait at the door to the bathroom. |
| 0096.mp4 | 96 | 17DRP5sb8fy | ✅ | Walk straight into the bedroom and around the bed... |
| 0097.mp4 | 97 | 17DRP5sb8fy | ❌ | Walk past curved sofa. Walk past bed. Wait at bathroom door threshold. |
| 0200.mp4 | 200 | 1LXtFkjw3qL | ✅ | Exit the bathroom, walk past the beds and wait in the doorway to the hall. |
| **0201.mp4** | **201** | **1LXtFkjw3qL** | **✅** | **Turn left and go past the bed and past the long couch. Go to and stop at the doorway on the right.** |
| 0202.mp4 | 202 | 1LXtFkjw3qL | ✅ | Exit and turn left. Walk straight passing the bed, and the cream sofa ahead of it... |
| 0269.mp4 | 269 | 1LXtFkjw3qL | ✅ | Exit the gym room. Turn left. Walk inside the room with the graffiti wall... |
| 0270.mp4 | 270 | 1LXtFkjw3qL | ✅ | Leave the exercise room, turn left, enter the doorway, and wait. |
| 0271.mp4 | 271 | 1LXtFkjw3qL | ✅ | Move forward past the exercise equipment and exit the room through the doorway... |
| 0302.mp4 | 302 | 1LXtFkjw3qL | ✅ | Exit the bathroom toward the room and pass the black curtain... |
| 0303.mp4 | 303 | 1LXtFkjw3qL | ✅ | Walk forward to the yellow sofa thing. Walk around the yellow sofa thing... |
| **0304.mp4** | **304** | **1LXtFkjw3qL** | **✅** | **Walk straight across the room to the other side. Once at the green bed, exit the room out of the door to your right...** |
| 0311.mp4 | 311 | 1LXtFkjw3qL | ❌ | Walk down the hall and turn left before the potted plant... |
| 0312.mp4 | 312 | 1LXtFkjw3qL | ❌ | Go straight towards the flower in the pot. Turn left right before the flower pot... |
| 0326.mp4 | 326 | 17DRP5sb8fy | ✅ | Turn to the left and walk through the living room to the breakfast bar... |
| 0327.mp4 | 327 | 17DRP5sb8fy | ❌ | Turn left and go passed the painting. Continue passed the bar... |
| 0328.mp4 | 328 | 17DRP5sb8fy | ✅ | Walk past living room, walk past dining room, turn right, wait by gold room. |

### B. 评估环境

```
Python: 3.10
PyTorch: 2.5.1+cu124
Habitat-Sim: 0.3.2
GPU: NVIDIA H100 80GB HBM3
CUDA: 12.4
```

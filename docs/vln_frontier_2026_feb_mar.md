# 2026年2-3月 VLN领域前沿论文调研

> **调研范围**: 2026年2月-3月 arXiv/顶会 VLN相关工作
> **论文数量**: 28篇
> **文档日期**: 2026年3月

---

## 目录

1. [Token压缩与效率优化](#1-token压缩与效率优化)
2. [VLA模型与端到端导航](#2-vla模型与端到端导航)
3. [RL训练策略](#3-rl训练策略)
4. [记忆与历史管理](#4-记忆与历史管理)
5. [零样本与通用导航](#5-零样本与通用导航)
6. [多智能体与协作导航](#6-多智能体与协作导航)
7. [场景理解与表征](#7-场景理解与表征)

---

## 1. Token压缩与效率优化

### 1.1 History-Conditioned Token Pruning for Efficient VLN
- **arXiv**: 2603.06480
- **问题**: VLN中多帧历史导致视觉token数量爆炸，推理延迟高
- **核心创新**: 提出无训练的时空双粒度token剪枝方法。空间维度：基于LLM attention热图识别低重要性token；时间维度：利用历史帧间的query引导去除冗余帧
- **方案**: (1) Spatial pruning: 计算每个visual token在LLM attention中的累积权重，剪枝低权重token; (2) Temporal pruning: 用当前帧query与历史帧做余弦相似度，去除冗余帧
- **结果**: 1.52x推理加速，性能损失<1%

### 1.2 VLN-Cache: Dynamic Token Caching for Efficient VLN
- **arXiv**: 2603.07080
- **问题**: 连续帧之间存在大量视觉冗余，但简单缓存策略无法处理视角变化
- **核心创新**: 视觉动态感知的token缓存+视图重映射机制。低动态时缓存KV cache，高动态时触发重新编码
- **方案**: (1) 计算帧间光流/特征差异作为"动态得分"; (2) 低动态: 重用上一帧KV cache; (3) 高动态: 重新编码但保留部分共享token的cache
- **结果**: 减少60%+的ViT计算量，SR降低<0.5%

### 1.3 DecoVLN: Decouple Visual Navigation with Language Models (CVPR 2026)
- **arXiv**: 2603.13133
- **问题**: 长距离导航中历史帧线性增长，导致VLM推理效率急剧下降
- **核心创新**: 三维自适应记忆管理——结合语义相关性、视觉多样性和时间衰减三个维度筛选关键帧
- **方案**: 解耦视觉编码与语言决策：(1) 视觉编码器独立处理每帧; (2) 记忆模块用三维评分选择top-K关键帧; (3) 仅将关键帧送入LLM决策
- **结果**: R2R SOTA，显著降低推理时间

### 1.4 SPAN-Nav: Spatial Token for Zero-Shot Navigation
- **arXiv**: 2603.09163
- **问题**: 全景视图产生大量token，但大部分是空间冗余的
- **核心创新**: 用单个spatial token编码全景空间关系，取代多视角拼接
- **方案**: (1) 训练轻量spatial encoder将12视角panorama编码为1个spatial token; (2) 与语言指令拼接后送入LLM; (3) LLM直接输出离散导航动作
- **结果**: Token数量降低90%+，零样本迁移能力强

### 1.5 BudVLN / Nipping the Drift in Budget
- **arXiv**: 2603.xxxx
- **问题**: VLN中token预算有限时，如何最优分配
- **核心创新**: 预算感知的token分配策略，根据导航进度动态调整每帧token预算
- **方案**: (1) 导航初期: 多分配token给指令理解; (2) 导航中期: 均匀分配给场景理解; (3) 接近目标: 多分配给停止决策
- **结果**: 固定预算下性能最优

---

## 2. VLA模型与端到端导航

### 2.1 AgentVLN: Vision-Language-Action Agent for Navigation
- **arXiv**: 2603.xxxxx
- **问题**: 现有VLN方法多为"感知-规划"两阶段，端到端VLA导航研究不足
- **核心创新**: 将VLN建模为VLA任务，直接从视觉观测和语言指令预测低级动作
- **方案**: (1) 基于预训练VLM(InternVL2)构建VLA; (2) 设计action tokenizer将连续动作离散化; (3) 使用导航专用数据进行instruction tuning
- **结果**: R2R上接近SOTA，且可直接部署到真实机器人

### 2.2 P³Nav: Perceive, Plan, Predict Navigation
- **arXiv**: 2603.xxxxx
- **问题**: VLN缺乏长期规划能力，逐步决策容易陷入局部最优
- **核心创新**: 三阶段感知-规划-预测框架，引入子目标预测
- **方案**: (1) Perceive: VLM编码当前观测; (2) Plan: 基于指令拆分生成子目标序列; (3) Predict: 预测到达下一子目标的动作序列
- **结果**: 长距离导航提升显著(+5% SR on R4R)

### 2.3 AlldayWalker / Tucker Adaptation (ICLR 2026)
- **arXiv**: 2603.14276
- **问题**: LoRA的2D矩阵适配无法同时编码多层级导航知识（场景共享/场景特定/环境变化）
- **核心创新**: 提出Tucker Adaptation (TuKA)，使用高阶张量分解来解耦多层级知识
- **方案**: W_adapted = W_base + G ×₁ U₁ ×₂ U₂ ×₃ U₃，其中G是核心tensor，U_i是各维度的适配因子
- **结果**: 在不同场景/光照条件下均优于LoRA，跨环境迁移能力强

### 2.4 ABot-N0: Autonomous Navigation Bot
- **arXiv**: 2603.xxxxx
- **问题**: VLN智能体难以处理真实世界中的未见场景和动态障碍
- **核心创新**: 将VLN与避障、路径规划结合的统一框架
- **方案**: (1) VLM做高层决策(目标导航); (2) 局部规划器做避障; (3) 安全约束层确保物理可行性
- **结果**: 真实世界导航成功率提升显著

---

## 3. RL训练策略

### 3.1 LongNav-R1: Reinforcement Learning for Long-Horizon VLN
- **arXiv**: 2602.12351
- **问题**: SFT训练的VLN智能体在长距离导航中性能急剧下降
- **核心创新**: 多轮对话格式的RL训练 + horizon-adaptive advantage estimation
- **方案**: (1) 将VLN建模为多轮对话(每步一轮); (2) 使用GRPO但加入horizon-adaptive baseline; (3) 仅4000条rollout即有效
- **结果**: Qwen3-VL-2B上SR 64.3%→73.0% (+8.7%)，长距离导航提升最大

### 3.2 NavGRPO: Group Relative Policy Optimization for VLN
- **arXiv**: 2603.15370
- **问题**: 标准PPO/DPO在VLN长序列决策中效率低、方差大
- **核心创新**: 将GRPO应用于VLN，设计导航专用的reward function
- **方案**: (1) 每个指令生成K条轨迹; (2) 组内相对排序; (3) 导航reward = 距离缩减 + 到达奖励 + 效率惩罚; (4) 组间优势估计
- **结果**: 比PPO更稳定，比DPO更高效

### 3.3 SACA: Step-Aware Contrastive Alignment
- **arXiv**: 2603.09740
- **问题**: CE Loss被简单token(前进/停止)主导，困难空间推理token学不好
- **核心创新**: 步级对比对齐——在每个时间步构造正负样本对，提供dense supervision
- **方案**: (1) 正样本: 正确动作执行后的下一状态; (2) 负样本: 错误动作执行后的下一状态; (3) 对比损失鼓励模型区分好/坏决策
- **结果**: 对困难转向决策提升显著(+3% SR on R2R)

---

## 4. 记忆与历史管理

### 4.1 GSMem: Graph-Structured Memory for VLN
- **arXiv**: 2603.xxxxx
- **问题**: 线性历史序列无法有效表达空间拓扑关系
- **核心创新**: 将导航历史组织为图结构记忆，节点=visited viewpoint，边=可达性
- **方案**: (1) 每个viewpoint存储压缩视觉token; (2) 用GNN传播空间关系; (3) 当前决策时attend到图中所有节点
- **结果**: 回溯/探索决策提升显著

### 4.2 HiMemVLN: Hierarchical Memory for VLN
- **arXiv**: 2603.xxxxx
- **问题**: 长导航中记忆容量与计算效率的矛盾
- **核心创新**: 三层记忆架构：工作记忆(当前帧，全量token) + 短期记忆(近期帧，压缩token) + 长期记忆(远期帧，极度压缩)
- **方案**: 随时间推移自动将帧从高层迁移到低层，逐步压缩
- **结果**: 超长导航(100+步)性能稳定

### 4.3 HaltNav: When to Stop Navigating
- **arXiv**: 2603.xxxxx
- **问题**: VLN中停止决策是最大失败来源(~50%错误来自过早/过晚停止)
- **核心创新**: 独立的停止决策模块 + 基于历史的置信度估计
- **方案**: (1) 主导航器生成动作概率; (2) 独立Halt模块综合指令完成度+场景匹配度判断是否停止; (3) 两个模块联合训练但分离推理
- **结果**: 停止决策准确率提升12%，OS-SR gap减少3%

---

## 5. 零样本与通用导航

### 5.1 OmniVLN: Omni-Modal VLN with Dynamic Scene Graphs
- **arXiv**: 2603.17351
- **问题**: 需要在预训练VLM上做大量VLN特定微调
- **核心创新**: 用Dynamic Scene Graph替代原始视觉token，实现61.7% token reduction
- **方案**: (1) 在线构建动态场景图(物体节点+空间边); (2) 将场景图序列化为结构化文本; (3) 与导航指令一起送入LLM
- **结果**: 零样本迁移到新场景效果好，但丢失低级纹理信息

### 5.2 SysNav: System 1 & System 2 Navigation
- **arXiv**: 2603.xxxxx
- **问题**: 简单导航(直行到尽头)与复杂导航(多步空间推理)应使用不同策略
- **核心创新**: 双系统架构——System 1(快速直觉)处理简单决策，System 2(慢速推理)处理复杂决策
- **方案**: (1) 路由器判断当前决策复杂度; (2) System 1: 轻量MLP直接输出动作; (3) System 2: 完整VLM推理链
- **结果**: 推理速度提升3x，性能持平

### 5.3 CMMR-VLN: Cross-Modal Map Reasoning
- **arXiv**: 2603.xxxxx
- **问题**: 文本指令与视觉观测的跨模态对齐不充分
- **核心创新**: 引入语义地图作为跨模态桥梁
- **方案**: (1) 在线构建2D语义地图(BEV); (2) 指令grounding到地图上; (3) 地图引导的注意力机制增强视觉-语言对齐
- **结果**: 跨模态推理能力提升，尤其是空间关系描述

### 5.4 One Agent to Guide Them All
- **arXiv**: 2603.xxxxx
- **问题**: 不同VLN任务(R2R/REVERIE/SOON/ScanQA)需要分别训练不同模型
- **核心创新**: 统一VLN智能体——单个模型处理多种导航任务
- **方案**: (1) 多任务指令模板标准化; (2) 共享backbone + 任务特定head; (3) 混合数据训练策略
- **结果**: 单模型在所有任务上接近各自SOTA

### 5.5 EmergeNav: Emergent Navigation from LLM
- **arXiv**: 2603.xxxxx
- **问题**: LLM是否具有内在的空间导航推理能力
- **核心创新**: 探索纯语言LLM(无视觉)的导航潜力
- **方案**: (1) 将场景描述为详细文本(物体位置/方向/距离); (2) 纯文本推理链; (3) 分析LLM内部的空间表征
- **结果**: 发现LLM确实存在空间推理涌现能力，但依赖高质量场景描述

---

## 6. 多智能体与协作导航

### 6.1 MA-CoNav: Multi-Agent Cooperative Navigation
- **arXiv**: 2603.xxxxx
- **问题**: 大规模环境中单智能体效率低
- **核心创新**: 多智能体协作导航——分工探索+信息共享
- **方案**: (1) 任务分解: 将长指令拆分为子任务; (2) 多智能体并行执行子任务; (3) 通信机制共享已探索信息
- **结果**: 大规模环境导航效率提升2x

### 6.2 AutoFly: Autonomous UAV Navigation
- **arXiv**: 2603.xxxxx
- **问题**: VLN主要针对室内，室外UAV导航研究不足
- **核心创新**: 将VLN范式扩展到3D UAV导航
- **方案**: (1) 3D视觉编码(深度估计+点云); (2) 高度感知的动作空间; (3) 安全约束(禁飞区/障碍物)
- **结果**: 室外UAV导航benchmark新SOTA

### 6.3 pFedNavi: Personalized Federated Navigation
- **arXiv**: 2603.xxxxx
- **问题**: 不同用户的导航偏好不同(速度/安全/路径偏好)
- **核心创新**: 联邦学习框架下的个性化导航
- **方案**: (1) 全局模型共享通用导航能力; (2) 本地模型适配用户偏好; (3) 隐私保护的参数聚合
- **结果**: 个性化满意度提升30%

---

## 7. 场景理解与表征

### 7.1 DACo: Depth-Aware Composition for VLN
- **arXiv**: 2603.xxxxx
- **问题**: 2D视觉特征缺乏3D空间信息
- **核心创新**: 深度感知的特征组合——将深度信息融入视觉token
- **方案**: (1) 单目深度估计; (2) 深度引导的attention权重; (3) 近处物体获得更多attention
- **结果**: 空间关系理解提升，近/远物体区分能力增强

### 7.2 ReasonNavi: Reasoning-Enhanced Navigation
- **arXiv**: 2603.xxxxx
- **问题**: VLN缺乏多步推理能力
- **核心创新**: 在导航决策中引入显式推理链(Chain-of-Thought)
- **方案**: (1) 指令→子目标分解; (2) 每步生成推理文本(当前位置/已完成子目标/下一步计划); (3) 推理文本辅助动作预测
- **结果**: 复杂指令(包含多个子任务)的SR提升6%

### 7.3 MerNav: Metric-Aware Navigation
- **arXiv**: 2603.xxxxx
- **问题**: VLN智能体对距离/角度等度量信息不敏感
- **核心创新**: 度量感知的导航训练——增加距离/角度预测辅助任务
- **方案**: (1) 主任务: 动作预测; (2) 辅助任务1: 到目标距离估计; (3) 辅助任务2: 到下一viewpoint的方向估计
- **结果**: NE降低0.5m，路径效率提升

### 7.4 FloorPlan-VLN: Using Floor Plans for Navigation
- **arXiv**: 2603.xxxxx
- **问题**: 纯视觉导航缺乏全局信息
- **核心创新**: 将楼层平面图作为辅助输入
- **方案**: (1) 楼层图编码为2D特征图; (2) 当前位置在楼层图上的定位; (3) 楼层图+视觉+语言三模态融合
- **结果**: 全局规划能力显著提升

### 7.5 TagaVLM: Tagging-Augmented VLM
- **arXiv**: 2603.xxxxx
- **问题**: VLM在导航中对物体标签/功能理解不够精确
- **核心创新**: 在视觉特征上叠加物体标签token，增强语义理解
- **方案**: (1) 开放集目标检测; (2) 将检测到的物体名称tokenize; (3) 在对应位置嵌入物体语义token
- **结果**: 物体相关指令(go to the red chair)导航准确率提升8%

### 7.6 WalkGPT: Walking with GPT for Outdoor Navigation
- **arXiv**: 2603.xxxxx
- **问题**: 室外步行导航需要理解街景、交通标志等复杂场景
- **核心创新**: 基于GPT-4V的室外步行导航框架
- **方案**: (1) 街景图像理解; (2) 交通规则reasoning; (3) 安全约束的路径规划
- **结果**: 室外导航benchmark新SOTA

### 7.7 BEACON: BEV-Conditioned Navigation
- **arXiv**: 2603.xxxxx
- **问题**: 第一人称视角缺乏全局空间感知
- **核心创新**: BEV(鸟瞰图)条件化的导航决策
- **方案**: (1) 实时构建局部BEV地图; (2) BEV特征作为额外条件输入; (3) BEV+第一人称双视角融合
- **结果**: 空间推理能力提升，回溯决策更准确

---

## 趋势总结

### 1. 六大技术趋势

| 趋势 | 代表工作 | 核心观点 |
|------|---------|---------|
| **Token压缩** | History-Cond. Pruning, VLN-Cache, DecoVLN, SPAN-Nav | 从无训练剪枝→可训练压缩→结构化替代 |
| **RL后训练** | LongNav-R1, NavGRPO, SACA | SFT上界有限，RL是突破口，GRPO优于PPO |
| **记忆管理** | GSMem, HiMemVLN, HaltNav | 从线性历史→图结构→层级记忆 |
| **多任务统一** | One Agent, AlldayWalker | 单模型处理多种VLN任务/场景 |
| **空间推理增强** | DACo, MerNav, BEACON | 深度/BEV/度量信息是关键补充 |
| **停止决策优化** | HaltNav, SACA | 停止决策是VLN最大瓶颈之一 |

### 2. 与IC-Compressor项目的关系

| 相关度 | 论文 | 可借鉴点 |
|--------|------|---------|
| ⭐⭐⭐ 直接相关 | History-Cond. Pruning, VLN-Cache, DecoVLN | token压缩策略对比基线 |
| ⭐⭐⭐ 直接相关 | SPAN-Nav, BudVLN | 极端压缩率的可行性验证 |
| ⭐⭐ 密切相关 | LongNav-R1, NavGRPO | RL后训练可作为Stage 4 |
| ⭐⭐ 密切相关 | SACA | Loss重加权解决过早停止 |
| ⭐⭐ 密切相关 | HiMemVLN | 多粒度压缩策略参考 |
| ⭐⭐ 密切相关 | AlldayWalker/TuKA | LoRA替代方案 |
| ⭐ 有参考价值 | HaltNav | 停止决策独立模块 |
| ⭐ 有参考价值 | OmniVLN, BEACON | 替代性场景表征 |

### 3. 对后续实验的建议

1. **优先验证**: Compressor在R2R上的压缩有效性(vs History-Cond. Pruning无训练基线)
2. **Loss策略**: 参考SACA加入动作token加权或对比损失
3. **RL后训练**: 参考LongNav-R1在Stage 3b后加入GRPO
4. **多粒度**: 参考HiMemVLN实现近期多token/远期少token的动态策略
5. **停止优化**: 参考HaltNav加入独立停止决策头

---

> **本文档为IC-Compressor项目提供2026年2-3月VLN前沿技术参考。共收录28篇论文，覆盖7个技术方向。**

# LoRA微调 + FiLM历史Token压缩：实施方案与训练策略

> **项目**: InternVLA-N1 IC-Compressor
> **基线系统**: DualVLN (Qwen2.5-VL-3B / Qwen3-VL-2B)
> **当前最佳结果**: R2R Val-Unseen SR=60.52%, SPL=54.98%
> **文档日期**: 2026年3月

---

## 目录

1. [方法概述与现有系统回顾](#1-方法概述与现有系统回顾)
2. [相关工作深度分析](#2-相关工作深度分析)
3. [当前实现的代码剖析](#3-当前实现的代码剖析)
4. [已有实验结果分析与问题诊断](#4-已有实验结果分析与问题诊断)
5. [后续改进方案](#5-后续改进方案)
6. [具体训练策略与超参数](#6-具体训练策略与超参数)
7. [实验计划与消融设计](#7-实验计划与消融设计)
8. [风险评估与缓解策略](#8-风险评估与缓解策略)

---

## 1. 方法概述与现有系统回顾

### 1.1 核心思路

我们的方法由两个核心组件构成：

1. **FiLM-Conditioned BottleneckCompressor**: 一个轻量级可训练模块，利用导航指令的语义信息(通过FiLM调制)来选择性压缩历史帧的视觉token（每帧 144/196 tokens → 16 tokens）
2. **LoRA微调LLM**: 通过Low-Rank Adaptation让LLM学会利用压缩后的16-token历史表征

### 1.2 现有实现架构

```
Navigation Instruction Text
        │
        ▼
  embed_tokens → mean_pool → instr_emb [d_model=2048]
        │
        ▼
  ┌─────────────────────────────────────────┐
  │    FiLM Conditioning Layer              │
  │    γ = Linear(instr_emb)   [d_bn=512]   │
  │    β = Linear(instr_emb)   [d_bn=512]   │
  └────────────────┬────────────────────────┘
                   │
    Learnable Queries [n_q=16, d_bn=512]
    Q_cond = γ ⊙ Q + β
                   │
  ┌────────────────┼────────────────────────┐
  │                │                        │
  History Frame 1  History Frame 2  ...  Frame N
  [144, 2048]      [144, 2048]          [144, 2048]
  → proj_in →      → proj_in →          → proj_in →
  [144, 512]       [144, 512]           [144, 512]
  │                │                     │
  CrossAttn×2      CrossAttn×2           CrossAttn×2
  (Q=Q_cond,KV)    (Q=Q_cond,KV)         (Q=Q_cond,KV)
  │                │                     │
  [16, 512]        [16, 512]             [16, 512]
  → proj_out →     → proj_out →         → proj_out →
  [16, 2048]       [16, 2048]            [16, 2048]
  └────────────────┴────────────────────────┘
                   │
                   ▼
  Compressed History: [N×16, 2048]
        +
  Current Frame [144, 2048] (unchanged)
  Birdseye Frame [144, 2048] (unchanged)
        │
        ▼
  ┌─────────────────────────────┐
  │  Qwen3-VL LLM Decoder      │
  │  (with LoRA r=32, α=64)    │
  │  36 layers, GQA 16Q/2KV    │
  │  exclude_modules=visual.*  │
  └─────────────────────────────┘
        │
        ▼
  Action Prediction (←, →, ↑, ↓, STOP, etc.)
```

### 1.3 分阶段训练策略（已实现）

| 阶段 | 可训练参数 | 冻结参数 | 训练目标 |
|------|-----------|---------|---------|
| **Stage 1a** | Compressor (~10.5M) | ViT + Merger + LLM + lm_head 全部 | 让Compressor学会有效压缩 |
| **Stage 2** | Compressor + Merger末层(linear_fc2+norm) | ViT + LLM主体 | 微调merger适配压缩表征 |
| **Stage 3b** | Compressor (~10.5M) + LLM LoRA (~14M) | ViT + Merger + embed_tokens + lm_head | 让LLM学会利用压缩token |

### 1.4 Token数量对比

| 方案 | 历史帧数 | 每帧tokens | 历史总tokens | 当前帧+俯视 | 总视觉tokens |
|------|---------|-----------|-------------|------------|-------------|
| **基线** (均匀采样) | 8 | 196 | 1,568 | 392 | **1,960** |
| **Stage 1a** (压缩) | 48 | 16 | 768 | 288 | **1,056** |
| **优势** | 6× more frames | 12× compression | **51% reduction** | — | **46% reduction** |

---

## 2. 相关工作深度分析

### 2.1 视觉Token压缩方法分类

根据最新文献（2025-2026），视觉token压缩方法主要分为三类：

#### A. Training-Free Pruning（无训练剪枝）

| 工作 | 方法 | 优势 | 劣势 |
|------|------|------|------|
| **History-Cond. Pruning** (2603.06480) | 基于attention热图+query引导的时空双粒度剪枝 | 即插即用，1.52x加速 | 信息丢失不可控，无法学习最优剪枝策略 |
| **VLN-Cache** (2603.07080) | 基于视觉动态感知的token缓存+视图重映射 | 低动态场景效果好 | 高动态转场时退化为全量计算 |
| **FastV** (ECCV 2024) | 基于attention score剪枝低重要性visual token | 简单高效 | 文本无关，VLN中不利 |
| **SparseVLM** (ICML 2025) | Text-informed visual token sparsification | 考虑文本引导 | 仅做推理加速，无训练优化 |
| **DUET-VLM** (CVPR 2026) | 训练+推理双阶段联合优化 | 通用性强 | 不针对VLN长序列场景 |

**关键洞察**：无训练方法的上界有限。当压缩率高时（如12:1），基于attention的剪枝会丢失大量信息。**可训练压缩器（如我们的方案）在高压缩率下有理论优势**。

#### B. Trainable Compression（可训练压缩）

| 工作 | 方法 | 与本项目关系 |
|------|------|------------|
| **HICom** (视频理解) | FiLM+CrossAttn learnable query | ⭐ **直接参考源**——我们的架构直接借鉴此设计 |
| **Compressor-VLA** (机器人VLA) | 双路交叉注意力(lang→vis + vis→lang) | 验证了指令条件化压缩在VLA中的有效性 |
| **CogVLA** (NeurIPS 2025) | AdaLN条件化+指令驱动路由+稀疏化 | 3阶段渐进架构(Encoder→Assembler→Decoder) |
| **VLA-Pruner** (2025) | 时间感知双级视觉token剪枝 | 帧级+token级双层剪枝 |
| **BFA++** (2602.20566) | 层级最佳特征感知token剪枝 | 多视图VLA场景 |

**关键洞察**：HICom和Compressor-VLA证明了**FiLM conditioning + learnable query + cross-attention**是视觉token压缩最有效的范式。我们的BottleneckCompressor正是遵循这个设计。

#### C. Structured Representation（结构化表征替代）

| 工作 | 方法 | 优劣 |
|------|------|------|
| **OmniVLN** (2603.17351) | Dynamic Scene Graph替代原始visual token | 61.7% token reduction，但丢失低级视觉信息 |
| **DecoVLN** (CVPR 2026) | 三维自适应记忆筛选(语义+视觉+时间) | 帧级筛选，粒度粗 |

### 2.2 LoRA在VLN/VLA中的应用

#### A. 标准LoRA微调

| 工作 | LoRA配置 | 效果 |
|------|---------|------|
| **InternVLA-N1 (基线)** | r=128, α=256, ViT+LLM全部LoRA | SR=60.52% |
| **LongNav-R1** (2602.12351) | Qwen3-VL-2B + multi-turn RL, 可能用LoRA | SR 64.3%→73.0% (+8.7%) |
| **AlldayWalker** (ICLR 2026) | 指出LoRA的2D矩阵形式无法捕获多层级导航知识 | 提出Tucker Adaptation (TuKA) |

#### B. LoRA的局限性分析

**AlldayWalker (ICLR 2026, 2603.14276)** 的关键发现：

> LoRA的二维矩阵形式只能在单一维度上适配，无法同时编码：
> 1. 跨场景的共享知识
> 2. 场景特定的专家知识
> 3. 环境变化(白天/夜晚)的层级知识

他们提出的**Tucker Adaptation (TuKA)**使用高阶张量分解来解耦多层级知识：
```
W_adapted = W_base + G ×₁ U₁ ×₂ U₂ ×₃ U₃
```
其中 G是核心tensor，U₁/U₂/U₃是不同维度的适配因子。

**对我们的启示**：
- 如果Stage 3b的LoRA效果不理想，可以考虑TuKA作为替代
- 但对于单场景(R2R)微调，标准LoRA应该足够

#### C. LoRA与Compressor的协同

**CogVLA (NeurIPS 2025)** 的3阶段训练策略与我们高度相似：

```
CogVLA Stage 1: Encoder alignment (冻结LLM，训练视觉encoder adapter)
CogVLA Stage 2: Assembler training (训练指令驱动的token路由器)
CogVLA Stage 3: Full LoRA fine-tuning (LoRA微调LLM + 解冻上面的模块)
```

**映射到我们的系统**：
```
我们 Stage 1a ≈ CogVLA Stage 2 (训练Compressor/Assembler，冻结LLM)
我们 Stage 3b ≈ CogVLA Stage 3 (LoRA微调LLM + Compressor继续训练)
```

### 2.3 FiLM Conditioning的理论基础

**Feature-wise Linear Modulation (FiLM)** 最初来自视觉推理任务(Perez et al., 2018)。在我们的场景中：

```python
# FiLM的数学形式
γ = W_γ · instr_emb + b_γ   # Scale factor
β = W_β · instr_emb + b_β   # Shift factor
Q_conditioned = γ ⊙ Q + β    # Modulated query
```

**为什么FiLM适合VLN token压缩**：

1. **指令引导的注意力偏置**: γ调制query的"关注强度"，使某些query维度更敏感于与指令相关的视觉特征
2. **任务自适应**: 不同指令产生不同的γ/β，使同一组query tokens在不同任务中提取不同信息
3. **计算轻量**: 仅需2个Linear层(d_model → d_bottleneck)，参数量约2×2048×512 ≈ 2M
4. **Identity初始化**: γ→1, β→0时FiLM退化为恒等映射，训练起步稳定

**与AdaLN(CogVLA)的对比**：
- AdaLN: `h = scale * LN(x) + shift` — 调制的是整个表征
- FiLM on queries: `Q_cond = γ⊙Q + β` — 仅调制查询信号
- 我们的选择: FiLM on queries更灵活，因为query决定了"看什么"，而不是"如何归一化"

### 2.4 其他相关训练策略参考

#### A. Dense Supervision vs Sparse Reward

**SACA (2603.09740)** 的核心发现与我们相关：
- 标准CE Loss受简单token(↓/STOP)主导
- 困难空间推理token(←←/→→)是少数
- 需要更dense的supervision

**解决方案选项**：
1. **Weighted CE Loss**: 对动作token加权，降低高频token权重
2. **步级对比学习**: 正确步+1/错误步-1的对比信号
3. **RL后训练**: SFT后接GRPO/DPO

#### B. 多轮RL后训练

**LongNav-R1 (2602.12351)** 证明在Qwen3-VL-2B上：
- 仅4000条rollout → SR +8.7%
- 关键：多轮对话格式 + horizon-adaptive advantage estimation

---

## 3. 当前实现的代码剖析

### 3.1 BottleneckCompressor 架构详解

```python
# 文件: internnav/model/compressor.py
class BottleneckCompressor(nn.Module):
    # 关键超参数
    d_model = 2048        # 输入/输出维度 (= LLM hidden_size)
    d_bottleneck = 512    # 瓶颈维度 (4x压缩)
    n_queries = 16        # 每帧压缩后的token数
    n_heads = 8           # 交叉注意力头数
    n_layers = 2          # 交叉注意力层数

    # 参数分布:
    # queries:     16 × 512       = 8.2K
    # film_gamma:  2048 → 512     = 1.05M
    # film_beta:   2048 → 512     = 1.05M
    # proj_in:     2048 → 512     = 1.05M
    # proj_out:    512 → 2048     = 1.05M
    # 2×CrossAttn: ~6.3M
    # final_norm:  512            = 1K
    # ──────────────────────────
    # 总计:        ~10.5M
```

**Bottleneck设计的关键价值**：
- 2048→512的降维在cross-attention前完成，使attention计算量降低16x
- FiLM在bottleneck空间操作，参数更少（2048→512 vs 2048→2048）

### 3.2 Compressor Wrapper 关键设计

```python
# 文件: internnav/model/compressor_wrapper.py

# 1. Monkey-patch机制
#    - 外层: Qwen3VLForConditionalGeneration.forward 拦截 is_history_image 参数
#    - 内层: Qwen3VLModel.forward 替换为compressor-aware版本

# 2. ZeRO-2兼容性
#    - 每个sample都执行_dummy_compressor_forward
#    - 确保compressor参数在每个rank的allreduce hook中一致

# 3. 历史DeepStack处理
#    - 历史帧的DeepStack embeddings设为zeros
#    - 仅当前帧保留完整DeepStack (ViT中间层特征)

# 4. RoPE位置编码
#    - 使用image_grid_thw_rope替代原始image_grid_thw
#    - 压缩后的token需要匹配的位置编码
```

### 3.3 三个Stage的训练配置

```python
# Stage 1a: 仅训练Compressor
apply_compressor_stage1a(model):
    for param in model.parameters():
        param.requires_grad = False
    model = attach_compressor(model)
    for param in model.compressor.parameters():
        param.requires_grad = True
    # Trainable: 10.5M / Total: ~3B = 0.35%

# Stage 2: Compressor + Merger末层
apply_compressor_stage2(model):
    # 加载Stage 1a的compressor权重
    # Compressor (base lr=5e-4) + merger.linear_fc2/norm (mm_projector_lr=1e-5)
    # Trainable: ~20M / Total: ~3B = 0.67%

# Stage 3b: Compressor + LLM LoRA
apply_compressor_stage3b(model):
    # 加载Stage 1a的compressor权重
    # LoRA: r=32, alpha=64, targets=[q/k/v/o/gate/up/down_proj]
    # exclude_modules=["visual.*"] → ViT不加LoRA
    # Trainable: ~24.5M / Total: ~3B = 0.82%
```

---

## 4. 已有实验结果分析与问题诊断

### 4.1 基线结果 (InternVLA-N1-System2)

| 指标 | R2R Val-Unseen (1839 episodes) |
|------|-------------------------------|
| Success Rate (SR) | **60.52%** |
| SPL | **54.98%** |
| Oracle Success (OS) | 67.86% |
| Navigation Error (NE) | 4.328m |
| SPL/SR | 0.908 (高效路径) |
| OS-SR gap | 7.34% (停止决策问题) |

### 4.2 失败模式分析

根据 exp.md 的详细分析：

| 失败类型 | 占比 | 原因 | Compressor可能的帮助 |
|----------|------|------|---------------------|
| 过早停止 | ~50% | STOP token在CE loss中权重过高 | 需要loss重加权 |
| 指令误解 | ~25% | 空间关系描述(left/right/behind)理解不准 | 指令条件化可增强空间关注 |
| 完全迷失 | ~25% | 历史信息利用不足，忘记之前观测 | 全量历史帧压缩可缓解 |

### 4.3 关键发现

1. **场景方差极大**: SR在不同场景中从22.2%到69.7%不等
2. **SPL/SR=0.908很高**: 说明一旦到达目标，路径效率好
3. **OS-SR=7.34%**: 说明部分episode"路过了目标但没停下来"

---

## 5. 后续改进方案

### 5.1 短期改进（可立即实施）

#### 改进1: Loss重加权 — 解决过早停止问题

**问题**: CE loss被高频token(↓/STOP)主导，困难空间推理token(←←/→→)学不好

**方案**:

```python
# 方案A: Token-level class weighting
action_weights = {
    "↓": 0.5,     # 高频动作，降权
    "STOP": 0.3,   # 最高频token，大幅降权
    "←": 1.5,     # 空间推理token，增权
    "→": 1.5,
    "←←": 2.0,    # 大幅转向，最有区分性
    "→→": 2.0,
    # 角度数值: 默认权重1.0
}

# 方案B: Focal Loss变体
# L = -α_t (1 - p_t)^γ log(p_t)
# γ=2, 让模型关注低置信度的token

# 方案C: Label smoothing
# 软化one-hot标签，减少STOP的过拟合
label_smoothing = 0.1
```

**推荐**: 先试方案A（token级加权），简单且效果可控。

#### 改进2: 动态压缩率

**问题**: 当前所有历史帧统一压缩到16 token，但不同帧的信息量差异很大

**方案**:

```python
class AdaptiveCompressor(BottleneckCompressor):
    def __init__(self, ..., min_queries=4, max_queries=32):
        super().__init__(...)
        # 增加一个"重要性预测头"
        self.importance_head = nn.Sequential(
            nn.Linear(d_bottleneck, 1),
            nn.Sigmoid()
        )

    def compress_frames_adaptive(self, frame_tokens, instr_emb):
        # 标准压缩到max_queries
        all_compressed = super().compress_frames(frame_tokens, instr_emb)
        # [N, max_queries, d_model]

        # 计算每个compressed token的重要性
        importance = self.importance_head(
            self.proj_in(all_compressed)
        )  # [N, max_queries, 1]

        # 按重要性排序，保留top-k (k可以per-frame不同)
        # 实际实现中可以用soft mask避免不可微
        mask = (importance > threshold).float()
        return all_compressed * mask, mask
```

**注意**: 动态压缩率会使batch内token数不一致，需要padding处理。建议先验证固定16是否足够，再考虑动态方案。

#### 改进3: DeepStack特征利用

**当前问题**: 历史帧的DeepStack embeddings被直接设为zeros

**方案**:

```python
# 当前: 历史帧DeepStack = zeros
# 改进: 历史帧DeepStack也通过compressor压缩

# 在compressor.py的forward中:
if deepstack_embeds_list is not None and n_hist > 0:
    for ds_idx, ds_embeds in enumerate(deepstack_embeds_list):
        ds_per_image = torch.split(ds_embeds, tokens_per_image)
        ds_history = torch.stack([ds_per_image[i] for i in range(n_hist)])
        # 用同一个compressor压缩deepstack
        ds_compressed = self.compress_frames(ds_history, instr_emb)
        # 而不是设为zeros
```

**风险**: DeepStack是ViT中间层特征，分布与merger后特征不同，可能需要单独的proj_in。

### 5.2 中期改进（需要更多实验验证）

#### 改进4: RL后训练 (SFT → GRPO)

基于LongNav-R1的成功经验，在SFT训练后加入RL阶段：

```
训练流程:
1. Stage 1a: Compressor预训练 (SFT, ~3 epochs)
2. Stage 3b: Compressor + LLM LoRA (SFT, ~3 epochs)
3. Stage 4 [NEW]: GRPO强化学习 (~1-2 epochs)
   - 保持Compressor和LoRA可训练
   - 使用环境reward:
     R = {
       +1.0  if distance_to_goal < 3m (成功)
       -0.5  if episode timeout
       +0.1 * (distance_reduction / total_distance)  per step
     }
   - Group Relative Policy Optimization:
     每个instruction生成K条轨迹,组内排序,学习排名靠前的
```

**预期收益**: 基于LongNav-R1的结果，可能带来+5~10% SR提升

#### 改进5: 指令嵌入质量提升

**当前问题**: 指令嵌入是简单的mean pooling所有text token（包括system prompt等无关文本）

**改进方案**:

```python
# 方案A: 仅pool instruction部分
# 在prompt模板中标记instruction的起止位置
# "Navigate: [INSTR_START] Go to the bedroom... [INSTR_END]"
# 只对[INSTR_START]到[INSTR_END]之间的token做mean pool

# 方案B: Weighted pooling (类似attention pooling)
class InstructionAttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn_weight = nn.Linear(d_model, 1)

    def forward(self, token_embeddings, mask):
        scores = self.attn_weight(token_embeddings)  # [L, 1]
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = F.softmax(scores, dim=0)
        return (weights * token_embeddings).sum(dim=0)
```

**推荐**: 先试方案A（精确提取instruction范围），成本最低。

#### 改进6: 多粒度Compressor

```python
class MultiGranularityCompressor(nn.Module):
    """
    近期帧: 保留较多token (32)
    中期帧: 中等压缩 (16)
    远期帧: 强压缩 (8)
    """
    def __init__(self, d_model=2048, d_bottleneck=512):
        super().__init__()
        self.recent_queries = nn.Parameter(torch.randn(32, d_bottleneck))
        self.mid_queries = nn.Parameter(torch.randn(16, d_bottleneck))
        self.far_queries = nn.Parameter(torch.randn(8, d_bottleneck))
        # 共享FiLM和CrossAttn layers
        ...

    def forward(self, frames, instr_emb, current_step):
        results = []
        for i, frame in enumerate(frames):
            age = current_step - i  # 帧的"年龄"
            if age <= 5:
                q = self.recent_queries  # 近期: 32 tokens
            elif age <= 20:
                q = self.mid_queries     # 中期: 16 tokens
            else:
                q = self.far_queries     # 远期: 8 tokens
            compressed = self.compress_single(frame, q, instr_emb)
            results.append(compressed)
        return results
```

**优势**: 近期帧保留更多细节(方向判断)，远期帧只保留粗粒度语义(到过哪些区域)

### 5.3 长期改进方向

#### 改进7: Tucker Adaptation替代LoRA

如果LoRA在多场景适应中效果不佳，参考AlldayWalker (ICLR 2026)：

```python
# Tucker Adaptation (TuKA) 概念
# W_adapted = W_base + G ×₁ U₁ ×₂ U₂ ×₃ U₃
# G: core tensor, U₁/U₂/U₃: mode-specific factors
```

#### 改进8: Compressor权重共享 vs 独立

当前所有历史帧和所有deepstack特征共享同一个compressor实例。可以考虑：
- **时间感知**: 近期帧/远期帧用不同的proj_in权重
- **层级感知**: primary/deepstack用不同的proj_in权重

---

## 6. 具体训练策略与超参数

### 6.1 推荐训练流程

```
═══════════════════════════════════════════════════════════
 Phase 1: Compressor预训练 (Stage 1a)
═══════════════════════════════════════════════════════════
 目标: 让Compressor学会有效的指令条件化压缩

 可训练: Compressor only (~10.5M params)
 冻结:   ViT + Merger + DeepStack + LLM + lm_head + embed_tokens

 超参数:
   learning_rate:    5e-4 (Compressor单独lr，较高因为从头训练)
   warmup_ratio:     0.1
   epochs:           5
   batch_size:       4 (per GPU)
   gradient_accum:   4 (effective batch=16)
   optimizer:        AdamW (β1=0.9, β2=0.999, wd=0.01)
   scheduler:        cosine

 数据配置:
   use_all_history:  True
   max_history:      48 frames
   num_compressed_tokens: 16
   freeze_vit_for_history: True (节省显存)

 History DeepStack: zeros (不压缩)

 预期结果:
   - Loss收敛但不需要很低(LLM冻结限制上界)
   - 核心验证: Compressor的attention map是否对指令相关区域有更高权重

═══════════════════════════════════════════════════════════
 Phase 2: LLM适配 (Stage 3b)  [跳过Stage 2]
═══════════════════════════════════════════════════════════
 目标: 让LLM学会理解和利用16-token压缩表征

 可训练: Compressor (~10.5M) + LLM LoRA (~14M) = ~24.5M
 冻结:   ViT + Merger + DeepStack + embed_tokens + lm_head

 超参数:
   learning_rate:    2e-5 (LLM LoRA的标准lr)
   compressor_lr:    5e-5 (Compressor继续训练但lr降低)
   warmup_ratio:     0.05
   epochs:           3
   batch_size:       2 (per GPU, 全量历史帧显存较大)
   gradient_accum:   8 (effective batch=16)

 LoRA配置:
   r:               32
   alpha:           64 (alpha/r = 2, 标准配置)
   dropout:         0.05
   target_modules:  [q_proj, k_proj, v_proj, o_proj,
                     gate_proj, up_proj, down_proj]
   exclude_modules: ["visual.*"]  # ViT不加LoRA
   bias:            "none"

 预期结果:
   - SR应该接近或超越基线(60.52%)
   - 如果低于基线，说明压缩丢失了关键信息

═══════════════════════════════════════════════════════════
 Phase 3: 改进训练 (可选, 基于Phase 2结果)
═══════════════════════════════════════════════════════════

 Option A: Loss重加权 (如果过早停止问题严重)
   - 降低STOP/↓权重，增加←←/→→权重

 Option B: RL后训练 (如果SFT上界已到)
   - GRPO, K=8 trajectories/instruction
   - 环境reward: 基于geodesic distance

 Option C: 增大LoRA rank (如果性能接近但不够)
   - r=64, alpha=128
   - 或者增加modules_to_save=["lm_head"]
```

### 6.2 超参数搜索优先级

| 优先级 | 超参数 | 搜索范围 | 理由 |
|--------|--------|---------|------|
| P0 | n_queries | {8, 16, 32} | 直接决定压缩率 |
| P0 | Stage 3b lr | {1e-5, 2e-5, 5e-5} | LoRA学习率最敏感 |
| P1 | d_bottleneck | {256, 512, 1024} | 影响compressor容量 |
| P1 | LoRA r | {16, 32, 64} | 影响LLM适配能力 |
| P2 | n_layers | {1, 2, 4} | Compressor深度 |
| P2 | max_history | {24, 48, 全部} | 信息量vs计算量 |
| P3 | FiLM设计 | {单层Linear, 2层MLP, 无FiLM} | 验证指令条件化增益 |

### 6.3 显存估算 (Stage 3b, A100 80GB)

```
模型参数:
  Base model (Qwen3-VL-2B):        ~3.5 GB (bf16)
  Compressor:                       ~21 MB (bf16)
  LoRA adapters:                    ~28 MB (bf16)
  Optimizer states (Compressor+LoRA): ~200 MB (fp32)

推理激活 (forward):
  ViT (50帧, no_grad for 48历史):    ~1.0 GB
  Compressor (48帧 × 2层CrossAttn):  ~1.5 GB
  LLM (1360 tokens × 36 layers):     ~5.0 GB

反向传播激活 (backward):
  Compressor gradients:               ~3.0 GB
  LLM LoRA gradients:                ~4.0 GB
  Gradient checkpointing savings:    -3.0 GB

总计:                                ~15.2 GB (单GPU)
可用裕量:                            ~65 GB
```

**结论**: A100 80GB可以轻松运行，batch_size可以开到4-8。4090 24GB需要batch_size=1 + gradient_accumulation=16。

---

## 7. 实验计划与消融设计

### 7.1 实验路线图

```
Week 1-2: Stage 1a Compressor预训练
  └─ Exp 1.1: n_queries=16, d_bn=512, lr=5e-4, 5 epochs
  └─ Exp 1.2: n_queries=8, 其余同上 (对比压缩率)
  └─ Exp 1.3: 无FiLM (Q不做条件化), 其余同上 (验证FiLM价值)

Week 3-4: Stage 3b LLM LoRA适配
  └─ Exp 3.1: 加载Exp 1.1的compressor, LoRA r=32, lr=2e-5, 3 epochs
  └─ Exp 3.2: 同上但lr=1e-5 (保守学习率)
  └─ Exp 3.3: 同上但r=64 (更大LoRA容量)
  └─ Exp 3.4: 加载Exp 1.3的compressor(无FiLM) (消融)

Week 5-6: 改进实验
  └─ Exp 4.1: 最佳Stage 3b + Weighted CE Loss
  └─ Exp 4.2: 最佳Stage 3b + 动态压缩率
  └─ Exp 4.3: 最佳Stage 3b + DeepStack压缩(非zeros)
  └─ Exp 4.4: 最佳Stage 3b + RL后训练(GRPO)
```

### 7.2 核心消融实验设计

#### 消融1: Compressor有效性

| 实验 | 配置 | 对比目标 |
|------|------|---------|
| Baseline | 8帧 × 196 tokens, LLM LoRA | 基准线 60.52% |
| A1 | 8帧 × 16 tokens (Compressor) + LLM LoRA | 压缩是否有损？ |
| A2 | 48帧 × 16 tokens (Compressor) + LLM LoRA | 全量帧是否有增益？ |
| A3 | 48帧 × 16 tokens (无FiLM) + LLM LoRA | FiLM的价值？ |

**关键指标**:
- A1 vs Baseline: 如果A1 ≥ Baseline-5%，说明压缩有效(可接受的信息损失)
- A2 vs A1: 差值=全量帧增益
- A2 vs A3: 差值=FiLM条件化增益

#### 消融2: 压缩率

| n_queries | 压缩率 | 48帧总tokens | 预期 |
|-----------|--------|-------------|------|
| 8 | 18:1 | 384 | 可能信息丢失过多 |
| 16 | 9:1 | 768 | 默认配置 |
| 32 | 4.5:1 | 1536 | 接近基线token数 |

#### 消融3: LoRA配置

| r | alpha | 参数量 | 预期 |
|---|-------|--------|------|
| 16 | 32 | ~7M | 可能不足以学习新表征格式 |
| 32 | 64 | ~14M | 默认配置 |
| 64 | 128 | ~28M | 如果r=32不够，增加容量 |

### 7.3 评估指标

| 指标 | 含义 | 目标 |
|------|------|------|
| SR | Success Rate (距离<3m) | ≥ 60% |
| SPL | Success weighted by Path Length | ≥ 55% |
| OS | Oracle Success (经过目标附近) | 参考 |
| NE | Navigation Error (平均距离) | ≤ 4.3m |
| SPL/SR | 路径效率 | ≥ 0.90 |
| Inference time | 单步推理时间 | < baseline |
| Total visual tokens | 每个episode的平均视觉token数 | < 1200 |

---

## 8. 风险评估与缓解策略

### 8.1 风险矩阵

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|---------|
| Compressor压缩丢失关键空间信息 | 中 | 高 | 增大n_queries; 加入空间位置编码; 试positional queries |
| LLM无法学会解读16-token表征 | 中 | 高 | 增大LoRA rank; 加入modules_to_save=["lm_head"]; 增加训练epoch |
| FiLM的指令嵌入质量差 | 低 | 中 | 改用attention pooling; 只pool指令部分; 加入独立text encoder |
| 过早停止问题未缓解 | 高 | 中 | Loss重加权; RL后训练; 增加STOP的负样本 |
| 显存不足(全量历史帧) | 低 | 低 | freeze_vit_for_history; 减少max_history; gradient checkpointing |
| DeepStack zeros导致信息不一致 | 中 | 中 | 实现DeepStack压缩; 或在Stage 2解冻DeepStack merger |

### 8.2 Plan B: 如果FiLM+CrossAttn压缩失败

如果整体方案的SR低于基线5%以上，考虑以下替代方案：

1. **替代方案A: Attention-based Frame Selection**
   - 不做token级压缩，而是帧级选择
   - 用Compressor的attention权重排序帧，选top-8
   - 保留完整196 tokens/帧

2. **替代方案B: Progressive Merging**
   - 先合并相邻帧(2帧→1帧，196 tokens保持)
   - 再做token级压缩(196→32)
   - 分层压缩可能保留更多信息

3. **替代方案C: KV-Cache Compression**
   - 不压缩visual embeddings，而是压缩LLM的KV cache
   - 参考StreamVLN的streaming memory方案

### 8.3 成功标准

| 阶段 | 成功标准 | 行动 |
|------|---------|------|
| Stage 1a完成 | Loss收敛，attention map合理 | 进入Stage 3b |
| Stage 3b完成 | SR ≥ 57% (基线-3%以内) | 继续优化 |
| 最终系统 | SR ≥ 62%, tokens减少40%+ | 整理论文 |
| 最终系统 | SR ≥ 65% (with RL) | 投顶会 |

---

## 附录: 关键引用文献

1. **HICom** — FiLM+CrossAttn视觉token压缩 (视频理解)
2. **Compressor-VLA** — 指令条件化VLA token压缩 (机器人)
3. **CogVLA** (NeurIPS 2025) — AdaLN条件化+3阶段训练 (VLA)
4. **History-Cond. Token Pruning** (2603.06480) — VLN无训练token剪枝
5. **VLN-Cache** (2603.07080) — VLN token缓存策略
6. **DecoVLN** (CVPR 2026) — 三维自适应记忆管理
7. **LongNav-R1** (2602.12351) — Qwen3-VL-2B + 多轮RL, SR+8.7%
8. **SACA** (2603.09740) — 步级对比对齐, dense supervision
9. **AlldayWalker** (ICLR 2026) — Tucker Adaptation替代LoRA
10. **DUET-VLM** (CVPR 2026) — 训练+推理双阶段token reduction
11. **BFA++** (2602.20566) — 多视图VLA token剪枝
12. **VLA-Pruner** (2511.16449) — 时间感知双级VLA token剪枝
13. **FlashVLM** (2512.20561) — Text-guided visual token selection
14. **NavGRPO** (2603.15370) — GRPO用于VLN
15. **SPAN-Nav** (2603.09163) — 单spatial token编码导航空间信息

---

> **本文档基于项目当前代码(compressor.py, compressor_wrapper.py)和2026年2-3月最新文献，为LoRA+FiLM历史token压缩方案提供完整的实施指导和后续改进路线图。**

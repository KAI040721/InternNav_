# Instruction-Conditioned Visual Token Compressor for VLN

> **项目代号**: IC-Compressor  
> **基线系统**: InternVLA-N1 (DualVLN) — Qwen2.5-VL-3B-Instruct  
> **目标会议**: ICLR / ICRA 2027  
> **文档版本**: 2026-02-28  

---

## 目录

1. [研究动机与问题定义](#1-研究动机与问题定义)
2. [文献调研](#2-文献调研)
3. [核心创新点](#3-核心创新点)
4. [基线系统分析](#4-基线系统分析)
5. [Compressor 模块设计](#5-compressor-模块设计)
6. [代码修改方案](#6-代码修改方案)
7. [训练策略](#7-训练策略)
8. [工程注意事项](#8-工程注意事项)
9. [实验计划](#9-实验计划)
10. [风险与备选方案](#10-风险与备选方案)

---

## 1. 研究动机与问题定义

### 1.1 VLN 中的历史帧瓶颈

Vision-Language Navigation (VLN) 智能体在导航过程中会持续积累视觉观测帧。一次典型的 R2R 导航 episode 有 **30-60 步**，每步产生一帧 RGB 图像。Qwen2.5-VL-3B 将每帧 384×384 图像编码为 **196 个 token**（dim=2048）。

当前 InternVLA-N1 的做法：
- 在 `[0, current_step-1]` 范围内**均匀采样 8 帧**作为历史
- 连同当前帧（1帧）+ 俯视图（1帧）= **10 帧 × 196 tokens = 1,960 个视觉 token**
- 加上文本 prompt 约 100-200 tokens → **LLM 输入约 2,100-2,200 tokens**

**核心问题**：
1. **信息丢失**：均匀采样 8 帧/60帧 → 87% 的视觉信息直接丢弃
2. **无差别对待**：所有历史帧等权重、等 token 数送入 LLM，没有根据当前指令做区分
3. **扩展瓶颈**：若要使用全部历史帧，60×196=11,760 tokens → LLM 无法承受

### 1.2 关键洞察

> **现有所有 VLN 方法的历史压缩都是 instruction-agnostic（指令无关）的。**

无论是均匀采样、FIFO 队列、还是基于简单规则的筛选，都没有利用导航指令来决定哪些视觉信息该保留、哪些该丢弃。

**经文献验证，"VLN × instruction-conditioned token compression" 的交叉领域完全为空白。**

---

## 2. 文献调研

### 2.1 VLN 相关方法

| 方法 | 会议 | 历史处理方式 | 是否指令条件化 |
|------|------|-------------|--------------|
| InternVLA-N1 (DualVLN) | — | np.linspace 均匀采样 8 帧 | ❌ |
| StreamVLN | ICLR 2026 | Streaming memory + forget gate | ❌ |
| NavFoM | ICLR 2026 | Foundation world model 预测未来帧 | ❌ |
| JanusVLN | ICRA 2026 | 双流（backward + forward）记忆 | ❌ |

**结论**：所有方法都是 instruction-agnostic 的历史处理。

### 2.2 视觉 Token 压缩参考论文（核心）

#### 2.2.1 HICom — Hierarchical Compressor (视频理解)
- **机制**：FiLM 条件化 + 交叉注意力
  - `γ, β = MLP(text_emb)` → `Q_cond = γ ⊙ Q + β`
  - 可学习 query token 通过交叉注意力聚合视频帧 token
- **压缩率**：16-32 tokens/帧
- **关键价值**：证明了 FiLM 条件化在视觉 token 压缩中的有效性
- **我们借鉴**：FiLM 条件化机制 + 可学习 query 的设计范式

#### 2.2.2 Compressor-VLA — 面向 VLA 的 Token 压缩
- **机制**：双路交叉注意力
  - 路径1: 语言→视觉 交叉注意力（语言 query 从视觉中提取信息）
  - 路径2: 视觉→语言 交叉注意力（视觉 query 从语言中获取条件）
- **压缩率**：任务自适应（2-64 tokens）
- **关键价值**：在机器人 VLA 场景验证了指令条件化压缩
- **我们借鉴**：指令条件化的必要性和效果

#### 2.2.3 CogVLA — AdaLN 条件化
- **机制**：Adaptive Layer Normalization
  - `scale, shift = MLP(lang_emb)` → `h = scale * LayerNorm(x) + shift`
  - 语言嵌入通过 AdaLN 在每一层调制视觉特征
- **关键价值**：更轻量的条件化方案（vs 交叉注意力）
- **我们评估**：AdaLN 不如 FiLM+CrossAttn 灵活，优先使用 FiLM

#### 2.2.4 FOCUS — Frame Selection (长视频)
- **机制**：Multi-Armed Bandit 关键帧选择
  - 用 BLIP ITM 评分帧-问题相关性
  - 两阶段：粗选 + 精选
  - 处理不到 2% 的帧即可工作
- **关键价值**：启发性——Compressor 的注意力权重可隐式实现帧选择
- **我们评估**：VLN 只有 ~60 帧，不需要 MAB；但 Compressor 的交叉注意力自然会给相关帧更高权重

### 2.3 创新空白验证

在 ICLR 2025/2026、ICRA 2026、NeurIPS 2025 等顶会中搜索：
- ✅ VLN + 历史压缩 = 有（均匀采样、streaming memory）
- ✅ 通用视觉 token 压缩 = 有（FastV, LLaVA-PruMerge, TokenPacker）
- ✅ 指令条件化压缩 = 有（在 VQA/视频理解领域: HICom, Compressor-VLA）
- ❌ **VLN × 指令条件化 token 压缩 = 空白** ← 我们的位置

---

## 3. 核心创新点

### 3.1 一句话总结

> **提出首个面向 VLN 的指令条件化视觉 Token 压缩器，将每帧 196 tokens 压缩至 16 tokens，使得全量历史帧（~48帧）的总 token 数反而低于基线 8 帧采样的 token 数。**

### 3.2 为什么这是创新

1. **VLN 领域第一个**：所有 VLN 方法的历史处理都是指令无关的
2. **反直觉的效率提升**：全量帧 × 压缩 < 采样帧 × 原始
   - 基线: 8帧 × 196 tokens = **1,568 tokens**
   - 我们: 48帧 × 16 tokens = **768 tokens** ← **减少 51%！**
3. **信息保留 + 选择性关注**：通过指令条件化，自动让相关帧获得更丰富的表示

### 3.3 Pipeline 总览

```
                        Navigation Instruction
                              │
                              ▼
                     ┌────────────────┐
                     │  embed_tokens   │ (Qwen2.5-VL 的词嵌入)
                     │  + mean pool    │
                     └───────┬────────┘
                             │
                        L_instr [2048]
                             │
                     ┌───────┴────────┐
                     │   FiLM Layer    │
                     │ γ,β = MLP(L)   │
                     └───────┬────────┘
                             │
                      Q_cond = γ⊙Q + β
                             │
     ┌───────────────────────┼───────────────────────┐
     │                       │                       │
  Frame 1                Frame 2              ... Frame N
  [196, 2048]            [196, 2048]              [196, 2048]
     │                       │                       │
     ▼                       ▼                       ▼
  CrossAttn(Q_cond,       CrossAttn(Q_cond,      CrossAttn(Q_cond,
   V=frame1, K=frame1)    V=frame2, K=frame2)    V=frameN, K=frameN)
     │                       │                       │
  [16, 2048]              [16, 2048]              [16, 2048]
     │                       │                       │
     └───────────┬───────────┘───────────────────────┘
                 ▼
        Compressed History: [N×16, 2048]
                 │
                 ▼
    ┌────────────┴────────────┐
    │  + Current Frame [196]   │
    │  + Birdseye Frame [196]  │
    │  + Text Tokens           │
    └────────────┬────────────┘
                 ▼
          LLM Decoder (Qwen2.5-3B)
                 ▼
          Action Prediction
```

---

## 4. 基线系统分析

### 4.1 Qwen2.5-VL-3B-Instruct 架构

```
┌────────────────────────────────────────────────────────┐
│  A. self.visual — ViT 视觉编码器                        │
│     depth=32, hidden=1280, 16 heads, head_dim=80       │
│     patch_size=14, spatial_merge_size=2                │
│     window_size=112                                    │
│     fullatt_block_indexes: [7, 15, 23, 31]             │
│                                                        │
│     PatchEmbed: Conv3D(3→1280, kernel=14×14)           │
│     32层 ViT Block (window attn + 4层 full attn)       │
│     PatchMerger: Linear(1280×4=5120 → 2048)            │
│                                                        │
│     384×384 → 28×28=784 patches → 2×2 merge → 196 tokens │
│     输出: [N_tokens, 2048]                              │
├────────────────────────────────────────────────────────┤
│  B. self.model — LLM Decoder                           │
│     embed_tokens: Embedding(151936, 2048)              │
│     36 层 Decoder Layer                                 │
│     GQA: 16 Q-heads, 2 KV-heads, head_dim=128         │
│     FFN intermediate: 11008                            │
│     M-RoPE: mrope_section=[16, 24, 24]                 │
│     sliding_window: 32768                              │
│     max_position_embeddings: 128000                    │
├────────────────────────────────────────────────────────┤
│  C. self.lm_head: Linear(2048 → 151936)               │
│     tie_word_embeddings = True (与 embed_tokens 共享)   │
└────────────────────────────────────────────────────────┘
```

**关键数值**：
- hidden_size = **2048** → Compressor 所有维度基于此
- 每帧 token 数 = **196** → 压缩目标 16
- vocab_size = **151936**
- IMAGE_TOKEN_INDEX = **151655** (`<image_pad>` 的 token id)
- TRAJ_TOKEN_INDEX = **151667**

### 4.2 InternVLA-N1 额外模块

在 `internvla_n1_arch.py` (InternVLAN1MetaModel) 中定义：

| 模块 | 维度 | 用途 |
|------|------|------|
| `latent_queries` | `nn.Embedding(n_query=4, 2048)` | S1 的 query |
| `action_encoder` | `Linear(3 → 2048)` | 轨迹编码 (x,y,θ → 2048) |
| `action_decoder` | `Linear(2048 → 3)` | 轨迹解码 |
| `cond_projector` | `Linear(3584 → 768)` | ⚠ **BUG: 3584 是 7B 的维度，3B 应为 2048** |
| `traj_dit` | NextDiT(hidden=768, ...) | S1 的扩散头 |
| `depth_anything` | DepthAnythingV2(vits) | 深度估计 |

**⚠ 注意**：`cond_projector` 的输入维度硬编码为 3584，这是 Qwen2.5-VL-**7B** 的 hidden_size，在 3B 模型上是一个 bug（但因 S1 目前未激活使用，不影响 S2 训练）。

### 4.3 数据流（S2 训练）

```
__getitem__() 流程:
  1. 选择 episode → 随机选 start_frame_id (当前步)
  2. 历史帧选择:
     history_id = np.unique(np.linspace(0, start_frame_id-1, num_history=8, dtype=int))
  3. 图像处理:
     for id in range(start_frame_id + pred_steps):
       if id in history_id or id == start_frame_id:
         images.append(process(rgb_image))      # → [3, 384, 384]
         image_grid_thw.append([1, 28, 28])     # → 196 tokens after merge
       if id == start_frame_id:
         images.append(process(birdseye_image)) # 额外的俯视图
  4. Prompt 模板:
     "These are your historical observations: <image>×8. <conjunction><image>."
     → preprocess_qwen_2_visual() 将每个 <image> 替换为 196 个 <image_pad>
  5. Label 格式:
     "↓", "245 180", "←←", "STOP" 等 action tokens
     仅 assistant turn 计算 CE loss (其余 labels=-100)
```

### 4.4 训练配置

| 配置项 | 服务器 | 4090 本地 |
|--------|--------|----------|
| ViT | **LoRA (所有层)** | 冻结 |
| Merger | LoRA | 冻结 |
| LLM | LoRA | LoRA |
| 数据量 | 100% R2R | 50% R2R |
| 性能 | SR ~53% (8B) | SR ~49% |
| 优化器 | AdamW | AdamW |

**关键发现**：
- SFT 损失很快就收敛
- 50% 数据 ≈ 100% 数据的性能
- 2B 参数 ≈ 8B 参数（仅差 ~4% SR）
- 原因：简单 token（↓/STOP/←→）主导 CE loss，困难的空间推理 token 是少数

---

## 5. Compressor 模块设计

### 5.1 模块定义

```python
class InstructionConditionedCompressor(nn.Module):
    """
    将每帧 196 个视觉 token 压缩为 16 个指令条件化 token。
    
    维度说明（基于 Qwen2.5-VL-3B）：
    - d_model = 2048 (= LLM hidden_size = Merger output dim)
    - n_queries = 16 (可学习 query 数量)
    - n_heads = 16 (交叉注意力头数)
    - n_layers = 2 (交叉注意力层数)
    - 参数量 ≈ 5-10M
    """
    
    def __init__(self, d_model=2048, n_queries=16, n_heads=16, n_layers=2):
        super().__init__()
        # 可学习 query tokens
        self.queries = nn.Parameter(torch.randn(n_queries, d_model))
        
        # FiLM 条件化: 从指令嵌入生成 γ 和 β
        self.film_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        self.film_beta = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 多层交叉注意力
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,  # 8192
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-LN
            )
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, frame_tokens, instr_emb):
        """
        Args:
            frame_tokens: [B, N_frames, 196, 2048] — ViT+Merger 后的帧 tokens
            instr_emb:    [B, 2048] — 指令的 mean-pooled 嵌入
        Returns:
            compressed:   [B, N_frames * 16, 2048] — 压缩后的 tokens
        """
        B, N, T, D = frame_tokens.shape  # T=196, D=2048
        
        # FiLM 条件化 query
        gamma = self.film_gamma(instr_emb)  # [B, D]
        beta = self.film_beta(instr_emb)    # [B, D]
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, 16, D]
        Q_cond = gamma.unsqueeze(1) * Q + beta.unsqueeze(1)  # [B, 16, D]
        
        # 对每帧独立做交叉注意力
        all_compressed = []
        for i in range(N):
            frame_i = frame_tokens[:, i]  # [B, 196, D]
            h = Q_cond  # [B, 16, D]
            for layer in self.layers:
                h = layer(h, frame_i)  # Cross-attention: Q=h, KV=frame_i
            all_compressed.append(h)
        
        compressed = torch.cat(all_compressed, dim=1)  # [B, N*16, D]
        compressed = self.final_norm(compressed)
        return compressed
```

### 5.2 指令嵌入来源

指令嵌入直接复用 Qwen2.5-VL 的 `embed_tokens`：

```python
# 在 forward() 中：
# 1. 找到 text tokens (非 IMAGE_TOKEN 和 TRAJ_TOKEN 的位置)
text_mask = (input_ids != IMAGE_TOKEN_INDEX) & (input_ids != TRAJ_TOKEN_INDEX)
text_ids = input_ids[text_mask]  

# 2. 通过 embed_tokens 获取文本嵌入
text_embs = self.model.embed_tokens(text_ids)  # [N_text, 2048]

# 3. Mean pooling 得到指令嵌入
instr_emb = text_embs.mean(dim=0, keepdim=True)  # [1, 2048]
```

**不需要额外的文本编码器** — 直接用 LLM 自带的词嵌入层。

### 5.3 Token 数量对比

| 方案 | 历史帧数 | 每帧tokens | 历史总tokens | 当前帧+俯视 | 总视觉tokens |
|------|---------|-----------|-------------|------------|-------------|
| 基线 | 8 | 196 | 1,568 | 392 | **1,960** |
| 我们 | 48 | 16 | 768 | 392 | **1,160** |
| 我们(满) | 60 | 16 | 960 | 392 | **1,352** |

**结论**：全量帧 + 压缩后的 token 数量反而比基线少 31-41%。

---

## 6. 代码修改方案

### 6.1 需要修改的文件清单

```
internnav/
├── model/
│   ├── compressor.py                          ← [新建] Compressor 模块
│   └── basemodel/internvla_n1/
│       ├── internvla_n1_arch.py               ← [修改] 注册 Compressor
│       └── internvla_n1.py                    ← [修改] forward() 插入压缩逻辑
├── trainer/
│   ├── internvla_n1_argument.py               ← [修改] 添加 Compressor 参数
│   └── internvla_n1_trainer.py                ← [修改] 冻结逻辑 + LoRA 配置
└── dataset/
    └── internvla_n1_lerobot_dataset.py        ← [修改] 全量帧 + token 数变更
```

### 6.2 文件 1: `internnav/model/compressor.py` [新建]

创建 5.1 节中的 `InstructionConditionedCompressor` 类。

### 6.3 文件 2: `internvla_n1_arch.py` [修改]

在 `InternVLAN1MetaModel.__init__()` 中注册 Compressor：

```python
# 在 self.latent_queries = ... 之后添加:
if getattr(config, 'use_compressor', False):
    from internnav.model.compressor import InstructionConditionedCompressor
    self.compressor = InstructionConditionedCompressor(
        d_model=config.hidden_size,  # 2048
        n_queries=getattr(config, 'num_compressed_tokens', 16),
        n_heads=16,
        n_layers=2
    )
```

### 6.4 文件 3: `internvla_n1.py` [修改] — 核心

**插入点**：在 `image_embeds = self.visual(...)` 之后、`masked_scatter` 之前。

```python
# 原始代码 (约 L129-131):
image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

# 修改为:
image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

# ---- Compressor 插入点 ----
if hasattr(self, 'compressor') and num_history_images is not None and num_history_images > 0:
    # 1. 按 image_grid_thw 拆分每帧的 token
    tokens_per_image = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2] // 4).tolist()
    frame_tokens = torch.split(image_embeds, tokens_per_image, dim=0)
    
    # 2. 提取指令嵌入
    text_mask_for_instr = (input_ids != IMAGE_TOKEN_INDEX) & (input_ids != TRAJ_TOKEN_INDEX)
    text_ids = input_ids[0][text_mask_for_instr[0]]  # batch=1 通常
    instr_emb = self.model.embed_tokens(text_ids).mean(dim=0, keepdim=True)  # [1, 2048]
    
    # 3. 压缩历史帧 (前 num_history_images 个)
    history_tokens = torch.stack([frame_tokens[i] for i in range(num_history_images)])  # [N_hist, 196, 2048]
    history_tokens = history_tokens.unsqueeze(0)  # [1, N_hist, 196, 2048]
    compressed = self.compressor(history_tokens, instr_emb)  # [1, N_hist*16, 2048]
    
    # 4. 保留当前帧和俯视图的原始 token
    current_tokens = torch.cat([frame_tokens[i] for i in range(num_history_images, len(frame_tokens))], dim=0)
    
    # 5. 重组
    image_embeds = torch.cat([compressed.squeeze(0), current_tokens], dim=0)
# ---- End Compressor ----

inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

### 6.5 文件 4: `internvla_n1_argument.py` [修改]

在 `ModelArguments` 中添加：

```python
# Compressor 相关参数
use_compressor: bool = field(default=False, metadata={"help": "是否启用 Instruction-Conditioned Compressor"})
num_compressed_tokens: int = field(default=16, metadata={"help": "每帧压缩后的 token 数量"})
compressor_layers: int = field(default=2, metadata={"help": "Compressor 交叉注意力层数"})
train_compressor_only: bool = field(default=False, metadata={"help": "Stage 1: 仅训练 Compressor"})
freeze_vit_for_history: bool = field(default=True, metadata={"help": "对历史帧的 ViT 推理冻结梯度（节省显存）"})
```

在 `DataArguments` 中添加：

```python
use_all_history: bool = field(default=False, metadata={"help": "使用全量历史帧而非均匀采样"})
max_history_frames: int = field(default=48, metadata={"help": "最大历史帧数量"})
```

### 6.6 文件 5: `internvla_n1_lerobot_dataset.py` [修改]

#### 6.6.1 历史帧选择逻辑 (在 `__getitem__` 中)

```python
# 原始 (约 L1000):
if start_frame_id != 0:
    history_id = np.unique(np.linspace(0, start_frame_id - 1, self.num_history, dtype=np.int32)).tolist()

# 修改为:
if start_frame_id != 0:
    if self.use_all_history:
        # 全量历史帧 (可设上限)
        all_ids = list(range(0, start_frame_id))
        if len(all_ids) > self.max_history_frames:
            history_id = np.unique(np.linspace(0, start_frame_id-1, self.max_history_frames, dtype=np.int32)).tolist()
        else:
            history_id = all_ids
    else:
        history_id = np.unique(np.linspace(0, start_frame_id-1, self.num_history, dtype=np.int32)).tolist()
```

#### 6.6.2 Token 数量变更 (在 `preprocess_qwen_2_visual()` 中)

```python
# 原始: 每个 <image> 替换为 grid_h * grid_w * temporal 个 <image_pad>
# 需要: 历史帧的 <image> 替换为 num_compressed_tokens 个 <image_pad>
#       当前帧/俯视图仍为 196 个 <image_pad>
```

具体实现需要在 `data_dict` 中传递 `num_history_images` 信息，使 `preprocess_qwen_2_visual()` 知道前 N 个 `<image>` 是历史帧（用 16 pads），后面的是当前帧（用 196 pads）。

#### 6.6.3 返回值中添加 `num_history_images`

```python
# 在 data_dict 中添加:
data_dict['num_history_images'] = len(history_id)
```

### 6.7 文件 6: `internvla_n1_trainer.py` [修改]

在 `set_model()` 中添加 Compressor 的冻结/解冻逻辑：

```python
# Stage 1: 仅训练 Compressor
if model_args.train_compressor_only:
    # 冻结一切
    for p in model.parameters():
        p.requires_grad = False
    # 仅解冻 Compressor
    if hasattr(model, 'compressor'):
        for p in model.compressor.parameters():
            p.requires_grad = True
```

---

## 7. 训练策略

### 7.1 三阶段训练

```
Stage 0: 基线复现 (已完成)
  └─ InternVLA-N1 原始训练 → SR ~53%

Stage 1: Compressor 预训练 (NEW)
  ├─ 冻结: ViT + Merger + LLM (全部)
  ├─ 训练: 仅 Compressor (~5-10M 参数)
  ├─ 数据: R2R 全量
  ├─ 损失: 标准 CE LM Loss (与基线相同)
  ├─ 学习率: 1e-4 (Compressor 独立 lr)
  ├─ Epochs: 3-5
  ├─ 目标: 让 Compressor 学会将 196 tokens 压缩为有效的 16 tokens
  └─ 预期: 若 Compressor 工作，即使 LLM 冻结，性能应接近 ~45-50% SR

Stage 2: 全量微调 (NEW)
  ├─ 冻结: ViT (历史帧) — freeze_vit_for_history=True
  ├─ LoRA: LLM 全层 + ViT (仅当前帧/俯视图的2帧)
  ├─ 训练: Compressor + LoRA
  ├─ 数据: R2R 全量，use_all_history=True
  ├─ 损失: CE LM Loss
  ├─ 学习率: Compressor 1e-4, LoRA 2e-5
  ├─ Epochs: 3-5
  └─ 目标: SR > 55% (超越基线 53%)
```

### 7.2 消融实验优先级

```
Ablation 1: Compressor 有效性
  (a) 基线: 8帧 × 196 tokens (原始 InternVLA-N1)
  (b) 8帧 × 16 tokens (仅压缩，不加帧)    → 验证压缩是否有损
  (c) 48帧 × 16 tokens (全量帧 + 压缩)     → 验证全量帧增益

Ablation 2: 指令条件化 vs 非条件化
  (a) FiLM 条件化 Compressor (完整方案)
  (b) 无条件化 Compressor (去掉 FiLM，Q 不变) → 验证指令条件化的增益

Ablation 3: 压缩率
  (a) 16 tokens/帧 (12:1 压缩率)
  (b) 32 tokens/帧 (6:1)
  (c) 8 tokens/帧 (24:1)

Ablation 4: Compressor 深度
  (a) 1 层交叉注意力
  (b) 2 层 (默认)
  (c) 4 层
```

---

## 8. 工程注意事项

### 8.1 显存分析

#### 服务器训练 (A100 80GB / A800)

**基线 (8帧 × 196)**:
- ViT 激活 (10帧, 有LoRA): ~2.0 GB
- LLM 激活 (2160 tokens): ~8.0 GB
- 模型参数 + 优化器: ~12 GB
- **总计: ~22 GB**

**我们 (48帧 × 16, freeze_vit_for_history=True)**:
- ViT 激活 (50帧, 但只有2帧有梯度): ~0.4 GB
- ViT 推理 (48帧历史, no_grad): ~0.5 GB (无需存激活)
- Compressor 激活 (48帧 × 2层CrossAttn): ~1.5 GB
- LLM 激活 (768+392+text ≈ 1360 tokens): ~5.0 GB
- 模型参数 + 优化器 + Compressor: ~13 GB
- **总计: ~20.4 GB** ← 比基线更少！

**关键**: `freeze_vit_for_history=True` 使得 48 帧历史帧的 ViT 推理完全在 `torch.no_grad()` 下进行，不存储激活图。

#### 实现方式

```python
# 在 forward() 中:
with torch.no_grad():
    history_pixel_values = pixel_values[:history_token_count]
    history_embeds = self.visual(history_pixel_values, grid_thw=history_grid_thw)

# 当前帧保持梯度
current_embeds = self.visual(current_pixel_values, grid_thw=current_grid_thw)
```

### 8.2 ViT LoRA 的特殊处理

**问题**: 服务器训练对 ViT 也施加了 LoRA。如果 48 帧历史帧都过 LoRA ViT 并保持梯度 → ViT 激活 ≈ 48×200MB = **9.6 GB** 爆炸。

**解决方案**: 对历史帧冻结 ViT（包括其 LoRA），仅对当前帧 + 俯视图保持 ViT LoRA 梯度。

- ViT LoRA 的更新信号只来自 2 帧 → 足够（当前帧是最重要的）
- 历史帧的 ViT 特征质量靠冻结的 ViT 保证（已经很好）
- Compressor 会学习在冻结 ViT 特征之上做指令条件化压缩

### 8.3 潜在优化：预计算 ViT 特征

如果 ViT 完全冻结（Stage 1），可以**离线预计算**所有帧的 ViT 特征：

```
每帧: 196 × 2048 × 2 bytes (bf16) = 784 KB
每 episode ~60帧: 47 MB  
R2R 全部 ~4000 episodes: ~184 GB (磁盘)
```

**建议**: 初期不做预计算（代码复杂度高），等确认方案有效后再优化。

### 8.4 已知 Bug

1. **`cond_projector` 维度 bug**: `Linear(3584 → 768)` 中的 3584 是 Qwen2.5-VL-**7B** 的 hidden_size。在 3B 模型上应为 `Linear(2048 → 768)`。但此模块仅用于 S1 (NextDiT)，不影响 S2 训练。如果将来要用，需要修复。

2. **`internvla_n1.py` L约33**: `IGNORE_INDEX == -100` 使用了 `==`（比较）而非 `=`（赋值）。实际上在 Python 中这行是个无意义的表达式，但 IGNORE_INDEX 在其他地方被正确赋值了，所以不影响运行。

---

## 9. 实验计划

### 9.1 时间线

```
Week 1: 代码实现
  ├─ Day 1-2: 创建 compressor.py + 修改 argument.py
  ├─ Day 3-4: 修改 dataset (全量帧 + token 数变更)
  ├─ Day 5-6: 修改 model forward() + trainer
  └─ Day 7: 本地 4090 上做 smoke test (小数据, 确认前向传播无误)

Week 2: Stage 1 训练 (Compressor only)
  ├─ 在服务器上运行 Stage 1 训练
  ├─ 监控: loss 曲线, Compressor attention 权重可视化
  └─ 评估: 冻结 LLM 下的 SR (预期 ~45-50%)

Week 3: Stage 2 训练 + 消融
  ├─ 全量微调 (Compressor + LoRA)
  ├─ 消融实验 (条件化 vs 非条件化, 压缩率等)
  └─ 评估: SR, SPL, nDTW 等指标

Week 4: 分析 + 论文初稿
  ├─ Attention 权重可视化 (哪些帧被赋予更高权重)
  ├─ 定性分析 (找出全量帧帮助的 case)
  └─ 论文 Introduction + Method 初稿
```

### 9.2 评估指标

| 指标 | 说明 |
|------|------|
| SR (Success Rate) | 到达目标 3m 内的比例 |
| SPL (Success weighted by Path Length) | 考虑路径效率的成功率 |
| nDTW (normalized Dynamic Time Warping) | 路径与参考路径的对齐度 |
| SDTW (Success weighted DTW) | 仅成功 episode 的 DTW |
| Token 数量 | 平均每步的视觉 token 数 |
| 推理速度 | 每步决策时间 (ms) |

### 9.3 期望结果

| 方法 | SR | SPL | 视觉 Tokens |
|------|-----|-----|------------|
| InternVLA-N1 基线 | 53% | ~45% | 1,960 |
| + Compressor (8帧×16) | ~51% | ~44% | 520 |
| + Compressor (48帧×16) | **>55%** | **>47%** | 1,160 |
| + Compressor + 全量微调 | **>57%** | **>49%** | 1,160 |

---

## 10. 风险与备选方案

### 10.1 风险

| 风险 | 可能性 | 影响 | 缓解 |
|------|--------|------|------|
| 压缩后信息损失过大 | 中 | 性能下降 | 增加压缩 tokens (16→32) |
| FiLM 条件化无增益 | 低 | 创新点弱化 | 改用 AdaLN 或双路交叉注意力 |
| 全量帧训练速度太慢 | 中 | 实验效率低 | 限制 max_history=32, 预计算ViT |
| Compressor Stage 1 不收敛 | 低 | 流程阻断 | 检查学习率, 增加层数 |
| 性能提升不显著 (<2% SR) | 中 | 论文贡献弱 | 强调效率指标(token减少51%)+注意力可解释性 |

### 10.2 备选方案 (Plan B)

如果 Compressor 方案效果不理想：

1. **降级方案**: 不做 token 压缩，而是做 **token 加权** — 用指令条件化的权重对 196 个 token 做 weighted sum → 1 个 token/帧。极端压缩但信息高度浓缩。

2. **方向转换**: 将 Compressor 改为 **Keyframe Selector** — 不压缩 token，而是用指令条件化的打分器选择 top-K 最相关帧（如 K=12），保持 196 tokens/帧。总 tokens: 12×196=2,352，但信息针对性更强。

3. **混合方案**: Top-K 帧保留完整 196 tokens + 其余帧压缩为 8 tokens → 兼顾重点帧的细节和全局帧的上下文。

---

## 附录 A: Qwen2.5-VL-3B vs Qwen3-VL-2B 对比

| 维度 | Qwen2.5-VL-3B | Qwen3-VL-2B |
|------|--------------|-------------|
| ViT 深度 | 32层 | 24层 |
| ViT hidden | 1280 | 1024 |
| patch_size | 14 | 16 |
| 每帧 tokens | 196 | 144 |
| LLM 层数 | 36 | 28 |
| LLM FFN | 11008 | 6144 |
| GQA KV头 | 2 | 8 |
| DeepStack | ❌ | ✅ [5,11,17] |
| Interleaved MRoPE | ❌ | ✅ |

**当前决定**: 基于 Qwen2.5-VL-3B 开发（与 InternVLA-N1 一致）。如果效果好，可迁移到 Qwen3-VL-2B（Compressor 代码完全不变，只需调整 token 数 196→144）。

---

## 附录 B: 关键代码位置速查

| 内容 | 文件 | 行号 |
|------|------|------|
| ViT forward + Merger | transformers 库 `modeling_qwen2_5_vl.py` | ~L1500-1600 |
| image_embeds 生成 | `internvla_n1.py` | ~L129 |
| masked_scatter 注入 | `internvla_n1.py` | ~L131 |
| 模块注册 (latent_queries等) | `internvla_n1_arch.py` | ~L30-80 |
| 冻结逻辑 set_model() | `internvla_n1_trainer.py` | ~L132-178 |
| apply_lora() | `internvla_n1_trainer.py` | ~L180+ |
| 历史帧选择 np.linspace | `internvla_n1_lerobot_dataset.py` | ~L1000 |
| preprocess_qwen_2_visual | `internvla_n1_lerobot_dataset.py` | ~L192-230 |
| DataArguments.num_history | `internvla_n1_argument.py` | ~L45 |
| IMAGE_TOKEN_INDEX | `internvla_n1.py` | L19 |
| TRAJ_TOKEN_INDEX | `internvla_n1.py` | L20 |

# InternNav_ 代码改动详细文档

> 基准版本：`fb1d2a5` "Add training code for InternVLA-N1"（原版 DualVLN，纯全参微调）
> 当前版本：`0c4a0c6`
> 变更统计：17 个文件，+3088 行，-46 行

---

## 一、总体改动概览

| 维度 | 原版（fb1d2a5） | 当前版本 |
|---|---|---|
| 基座模型 | Qwen2.5-VL-7B | **Qwen3-VL-2B** |
| 微调策略 | 全参微调（三开关 tune_mm_*） | **LoRA r=32/α=64 + Compressor** |
| 历史帧编码 | ViT 直出，144 token/帧 | **BottleneckCompressor 压缩至 16 token/帧** |
| 训练 batch | 2（8节点×8卡） | **24（单节点×2卡 GPU 0,3）** |
| 学习率 | 2e-5 | **2e-4** |
| 分布式策略 | DeepSpeed ZeRO | **ZeRO-2** |
| 评估代码 | 仅 eval_system2.sh 脚本 | **全新 Python 评估器类** |
| 新增文件数 | — | **+8 个新文件** |

训练被分为四个阶段：
- **Stage 1a**：仅训练 Compressor（~10.5M 参数）
- **Stage 2**：同上，加载 Stage 1a 权重继续
- **Stage 3b**：Compressor + LLM LoRA（无 modules_to_save，~24M 参数）
- **Stage 3**：Compressor + LoRA（含 merger/embed_tokens/lm_head，~773M = 27.38%）

---

## 二、改动文件列表

### 2.1 已修改的文件（vs 原版）

| 文件 | 改动规模 | 核心变化 |
|---|---|---|
| internnav/trainer/internvla_n1_argument.py | +35 行 | 新增 Compressor 参数 + LoRA 参数 |
| internnav/trainer/internvla_n1_trainer.py | +270 行，-36 行 | 全面重构：LoRA 策略、Compressor 分发、Qwen3 支持 |
| internnav/dataset/internvla_n1_lerobot_dataset.py | +190 行，-10 行 | Compressor 数据准备、固定随机种子、CombinedDataset |

### 2.2 全新文件（原版不存在）

| 文件 | 行数 | 作用 |
|---|---|---|
| internnav/model/compressor.py | 263 | BottleneckCompressor 神经网络模块 |
| internnav/model/compressor_wrapper.py | 993 | Monkey-patch 机制 + 4 个训练阶段函数 |
| internnav/habitat_extensions/vln/habitat_vln_evaluator.py | 1035 | 全新 Python 评估器 |
| scripts/eval/configs/eval_task_a_2b_cfg.py | ~50 | Task A 评估配置 |
| scripts/eval/configs/eval_compressor_3b_2b_cfg.py | ~50 | Stage 3b 评估配置 |
| scripts/eval/bash/eval_task_a_2b.sh | ~10 | Task A 评估启动脚本 |
| scripts/eval/bash/eval_compressor_3b_2b.sh | ~10 | Stage 3b 评估启动脚本 |
| scripts/eval/configs/vln_r2r.yaml | ~30 | 评估数据集配置（val_unseen） |
| train_compressor_1a_2b.sh | ~30 | Stage 1a 训练脚本 |
| train_compressor_2_2b.sh | ~30 | Stage 2 训练脚本 |
| train_compressor_3b_2b.sh | ~30 | Stage 3b 训练脚本 |
| train_compressor_3_2b.sh | ~40 | Stage 3（当前）训练脚本 |

---

## 三、模块详细说明

### 3.1 internvla_n1_argument.py

**文件职责**：定义训练超参数 dataclass（ModelArguments、DataArguments、TrainingArguments）。

**原版**：只有 tune_mm_llm、tune_mm_mlp、tune_mm_vision 三个布尔开关，加上基础数据路径参数。

**改动内容**：

新增 Compressor 参数（ModelArguments 中新增 7 项）：

    use_compressor: bool = False
    compressor_stage: str = "1a"          # "1a", "2", "3b", "3"
    compressor_n_queries: int = 16        # 压缩后 token 数
    compressor_d_bottleneck: int = 512    # bottleneck 维度
    compressor_n_layers: int = 2          # CrossAttention 层数
    compressor_n_heads: int = 8           # 注意力头数
    compressor_stage1a_checkpoint: str = None  # Stage1a 权重路径

新增 LoRA 参数（ModelArguments 中新增 6 项）：

    use_lora: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: str = None       # 逗号分隔的目标模块名

---

### 3.2 internvla_n1_trainer.py

**文件职责**：主训练入口，负责模型加载、参数冻结/解冻、数据集创建、调用 HuggingFace Trainer。

**原版核心结构**：set_model() 函数，通过三个布尔值决定哪些子模块可训，仅支持 Qwen2VL。

**改动内容（共 6 处）**：

#### 改动 1：NCCL 超时修复

在文件顶部（import 之后）立即覆盖默认超时：

    import torch.distributed as dist
    dist.default_pg_nccl_timeout = datetime.timedelta(seconds=7200)

原因：原版默认 600 秒，大模型 forward/backward 时间较长时导致分布式进程组超时崩溃，改为 2 小时。

#### 改动 2：模型加载切换到 Qwen3-VL

    # 原版
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLImageProcessor
    model = Qwen2VLForConditionalGeneration.from_pretrained(...)
    data_args.model_type = "qwen2vl"

    # 现在
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(...)
    processor = AutoProcessor.from_pretrained(...)  # 兼容 Qwen3（无 .tokenizer 属性）
    data_args.model_type = "qwen3vl"

#### 改动 3：ViT 梯度检查点禁用

    model.visual.gradient_checkpointing = False

原因：Compressor 在 ViT 特征之后操作，启用梯度检查点会导致 ViT 特征被重算，与 Compressor 的 in-place 替换不兼容。

#### 改动 4：新增 apply_lora_to_qwen3vl() 函数（~150 行）

实现 LoRA v4 策略（Task A 独立路线，不含 Compressor）：

    1. 冻结所有参数
    2. LoRA 注入目标模块：
       LLM:   q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
       ViT:   qkv, proj（注意力层）
    3. modules_to_save = ["merger", "embed_tokens", "lm_head"]
    4. 手动解冻 deepstack_merger_list（PEFT 的 "merger" 只匹配主 merger）
    5. 手动解冻所有 norm 层（ViT LayerNorm + LLM RMSNorm）

可训练参数：~773M（27.38%）。

#### 改动 5：训练分发逻辑重构

原版只有一条路径（set_model()）；现在是三路分支：

    if model_args.use_compressor:
        stage_fn = {"1a": apply_compressor_stage1a, "2": apply_compressor_stage2,
                    "3b": apply_compressor_stage3b, "3": apply_compressor_stage3}[stage]
        model = stage_fn(model, compressor_config, ...)
    elif model_args.use_lora:
        model = apply_lora_to_qwen3vl(model, model_args)
    else:
        set_model(model, model_args)  # 原版全参微调路径（保留）

#### 改动 6：Compressor 权重独立保存

训练完成后，额外保存 Compressor 权重（键名带 compressor. 前缀）：

    compressor_state = {f"compressor.{k}": v
                        for k, v in model.compressor.state_dict().items()}
    save_file(compressor_state, "compressor_stage3b.safetensors")

---

### 3.3 internvla_n1_lerobot_dataset.py

**文件职责**：VLN 训练数据集 + 数据 collator。

**改动内容（共 5 处）**：

#### 改动 1：固定随机种子（分布式一致性）

在所有 random.sample()、random.shuffle()、np.random.shuffle() 之前添加 random.seed(42) / np.random.seed(42)。

原因：多卡训练时各进程 random state 不一致，导致 history 帧采样不一致，引发 position_ids 维度不对齐崩溃。

#### 改动 2：NavPixelGoalDataset.__init__ 记录 Compressor 参数

    self.use_compressor = data_args.use_compressor
    self.compressor_n_queries = data_args.compressor_n_queries

#### 改动 3：__getitem__ 核心 Compressor 数据准备

区分历史帧（n_queries=16 tokens）和当前帧（144 tokens）：

    for idx_img, img in enumerate(all_images):
        is_history = (self.use_compressor and idx_img < num_history_images)
        if is_history:
            grid_thw_for_tokens = [1, sq, sq]  # sq=4，RoPE 用压缩后尺寸
        else:
            grid_thw_for_tokens = original_grid_thw  # 144 tokens
        # image_grid_thw 始终用原始值（ViT 处理需要）

新增两个输出字段：
- data_dict["is_history_image"]：bool tensor[n_images]，标记哪些帧是历史帧
- data_dict["image_grid_thw_rope"]：tensor[n_images, 3]，RoPE 使用的（压缩后）grid_thw

#### 改动 4：DataCollatorForSupervisedDataset 新字段拼接

    if "is_history_image" in instances[0]:
        batch["is_history_image"] = torch.cat([inst["is_history_image"] for inst in instances])
        batch["image_grid_thw_rope"] = torch.cat([inst["image_grid_thw_rope"] for inst in instances])

注意：必须配合 --remove_unused_columns False，否则 HF Trainer 会静默过滤这两个字段。

#### 改动 5：新增 CombinedDataset

合并 VLN + VLLN 等多个数据集，固定种子 shuffle 后交叉采样。

---

### 3.4 compressor.py（全新文件）

**文件职责**：BottleneckCompressor 神经网络，将历史帧 144 个 ViT token 压缩为 16 个语义 token。

**架构（输入→输出）**：

    history_tokens [n_hist, 144, 2048] + instr_emb [2048]
    ↓
    proj_in:  Linear(2048 → 512)
    ↓
    queries:  nn.Parameter([16, 512])  可学习查询向量
    ↓
    CrossAttentionBlock × 2：
      Q=queries(16,512), K/V=image_tokens(144,512)
      MultiheadAttention(8 heads)
      FiLM: γ,β = Linear(instr_emb→1024)，output = γ*output + β
      FFN: Linear(512→2048→512) + GELU
    ↓
    proj_out: Linear(512 → 2048)
    ↓
    final_norm: LayerNorm(2048)
    ↓
    compressed_tokens [n_hist, 16, 2048]

**参数量**：~10.51M

---

### 3.5 compressor_wrapper.py（全新文件）

**文件职责**：通过 monkey-patch 将 Compressor 插入 Qwen3VL 的 ViT 与 LLM 之间，无需修改 transformers 源码。

**Patch 后调用链**：

    outer_forward（替换 Qwen3VLForConditionalGeneration.forward）
      ├─ 从 kwargs 提取 is_history_image, image_grid_thw_rope
      ├─ 存入 model._compressor_is_history, model._compressor_grid_thw_rope
      └─ 调用原始 outer forward
           └─ inner_forward（替换 Qwen3VLModel.forward）
                ├─ 用原始 image_grid_thw 运行 ViT → 每帧 144 tokens
                ├─ split_sizes = grid_thw.prod(-1) // spatial_merge_size²
                ├─ 历史帧 → compress_frames() → 16 tokens/帧
                ├─ 当前帧 → 保持 144 tokens
                ├─ 拼接 → image_embeds_final
                └─ 散列到 inputs_embeds

**关键函数**：

| 函数 | 作用 |
|---|---|
| get_instruction_embedding() | 纯文本 forward，获取指令嵌入用于 FiLM 条件化 |
| _dummy_compressor_forward() | 无历史帧时做 dummy pass，避免 DDP 未使用参数报错 |
| make_outer_forward() | Patch Qwen3VLForConditionalGeneration.forward |
| make_inner_forward() | Patch Qwen3VLModel.forward，实现压缩+散列流程 |
| attach_compressor() | 创建 Compressor，注入 model.compressor，patch 两层 forward |

**四个训练阶段函数**：

    apply_compressor_stage1a：冻结全部 → attach → 仅 Compressor 可训  (~10.5M)
    apply_compressor_stage2：同上 + 加载 Stage1a 权重  (~10.5M)
    apply_compressor_stage3b：冻结全部 → attach → LLM LoRA（无 modules_to_save）→ 解冻 Compressor  (~24M)
    apply_compressor_stage3：冻结全部 → attach → 加载 Stage3b Compressor → 全量 LoRA + modules_to_save + 解冻 norms/deepstack/compressor  (~773M)

---

### 3.6 habitat_vln_evaluator.py（全新文件）

**文件职责**：Habitat 仿真环境 VLN 评估器，原版只有 shell 脚本，现在是完整 Python 类，继承自 DistributedEvaluator。

**模型加载流程**：

    1. AutoProcessor（兼容 Qwen3 无 .tokenizer 属性 / Qwen2.5 有 .tokenizer 属性）
    2. AutoConfig 自动检测 Qwen3 vs Qwen2.5_VL → 对应类加载基座模型
    3. PeftModel.from_pretrained(base_model, adapter_path) → merge_and_unload()
    4. attach_compressor(model, config)
    5. 加载 Compressor 权重（Bug Fix：剥除 "compressor." 前缀再 load_state_dict）
       修复前：missing=39/39，Compressor 全随机 → 输出退化 99.3% 箭头
       修复后：missing=0, unexpected=0

**推理时 Compressor 数据准备**（每个时间步）：

    # 1. 设置历史帧标记
    is_hist = torch.tensor([True]*n_hist + [False]*n_cur)
    model._compressor_is_history = is_hist

    # 2. 压缩 RoPE grid_thw（历史帧 [1,4,4]，当前帧保持原值）
    rope_thw[i] = torch.tensor([1, 4, 4])  for i in range(n_hist)
    model._compressor_grid_thw_rope = rope_thw

    # 3. 压缩 input_ids 中的 image_pad tokens（历史帧 144→16）
    # 遍历 input_ids，将历史帧的连续 image_pad 段从 144 个替换为 16 个

**指标**：SR / SPL / OS / NE / nDTW，支持断点续评（progress.json）。

---

## 四、关键 Bug 修复记录

### Bug 1：Stage 3 训练崩溃（--remove_unused_columns False 缺失）

**症状**：训练约 10 步后崩溃，报错 features 23436 vs tokens 7052

**根因**：HF Trainer 的 RemoveColumnsCollator 根据模型 forward() 签名过滤字段，is_history_image 和 image_grid_thw_rope 不在标准签名中被静默删除，Compressor 读到 None 导致 token 数不一致。

**修复**：train_compressor_3_2b.sh 中添加 --remove_unused_columns False

### Bug 2：Stage 3b 评估退化（Compressor 权重未加载）

**症状**：评估输出 99.3% 箭头 + 0% STOP，Compressor 行为完全随机

**根因**：safetensors 文件 key 带 "compressor." 前缀，但 load_state_dict() 期望不带前缀，missing=39/39 全部未加载

**修复**：

    comp_state = {k.replace("compressor.", "", 1): v
                  for k, v in raw_state.items()}
    model.compressor.load_state_dict(comp_state, strict=False)

### Bug 3：评估在 val_seen 集合上

**修复**：vln_r2r.yaml 中 split: val_seen → split: val_unseen

---

## 五、训练进度快照

| 阶段 | 状态 | 最终 Loss | 可训参数 |
|---|---|---|---|
| Stage 1a | 完成 | 0.99 | ~10.5M |
| Stage 2 | 完成 | ~0.55 | ~10.5M |
| Stage 3b | 完成 | ~0.55 | ~24M |
| Task A (LoRA v4) | 完成 | 0.168 | ~773M |
| Stage 3 | **运行中** | ~0.63（epoch 0.19） | ~773M |

当前 Stage 3：GPU 0+3，batch=24，grad_accum=2，lr=2e-4，ZeRO-2

---

## 六、文件依赖关系图

    训练入口
    └─ internvla_n1_trainer.py
       ├─ internvla_n1_argument.py         (参数定义)
       ├─ internvla_n1_lerobot_dataset.py  (数据集)
       └─ compressor_wrapper.py            (模型 patch)
          └─ compressor.py                 (Compressor 网络)

    评估入口
    └─ habitat_vln_evaluator.py
       ├─ compressor_wrapper.py            (attach_compressor)
       └─ compressor.py                    (BottleneckCompressor)

# FiLM模块训练方案：在2B Baseline上验证历史帧压缩

> **核心目标**: 在已有的 Qwen3-VL-2B AllLoRA baseline 基础上，**仅增加一个FiLM Compressor模块**，验证该模块能否通过压缩历史帧token提升导航性能。
>
> **文档日期**: 2026-03-22

---

## 1. 先搞清楚：Baseline做了什么，FiLM要做什么

### 1.1 Baseline是什么

当前2B baseline (`train_compressor_baseline_2b.sh`) 的训练配置：

```
Qwen3-VL-2B-Instruct
├── ViT (24层, hidden=1024)
│   └── LoRA on: qkv, proj, fc1, fc2  (r=32, α=64)
├── Merger (patch merge 2×2, fc1→GELU→fc2, 1024*4→2048)
│   └── LoRA on: fc1, fc2 (通过target_modules覆盖)
└── LLM (36层, hidden=2048)
    └── LoRA on: q/k/v/o_proj, gate/up/down_proj  (r=32, α=64)

可训练参数: ~39.7M (1.83%)
学习率: 统一 2e-4
数据: R2R + RxR (各50%), 2 epochs
历史帧: 8帧 × 144 tokens = 1152 tokens 直接灌入LLM
```

### 1.2 加FiLM后变成什么

```
Qwen3-VL-2B-Instruct
├── ViT        ← LoRA 完全相同 (控制变量)
├── Merger     ← LoRA 完全相同 (控制变量)
├── 🆕 FiLM Compressor (~10.5M, 全参训练)
│   └── 历史帧: 每帧 144 → 16 tokens
│   └── 当前帧: pass-through (不处理)
└── LLM        ← LoRA 完全相同 (控制变量)

可训练参数: ~39.7M (LoRA) + ~10.5M (Compressor) ≈ 50.2M (2.3%)
历史帧: 8帧 × 16 tokens = 128 tokens (压缩9倍)
当前帧: 144 tokens  |  鸟瞰图: 144 tokens
总视觉tokens: 128 + 144 + 144 = 416  (vs baseline的 1440)
```

---

## 2. FiLM Compressor如何注入到Qwen3-VL推理流程中

### 2.1 Qwen3-VL正常的图像处理流程

```
所有图片 → ViT → Merger → all_image_embeds [1440, 2048]
                                    ↓
         split by grid_thw → 10 tensors, 各 [144, 2048]
                                    ↓
         masked_scatter 到 inputs_embeds 中的 <image_pad> 占位符
                                    ↓
         LLM decoder → action tokens
```

### 2.2 加FiLM后（monkey-patch注入）

`compressor_wrapper.py` 替换了 `Qwen3VLModel.forward`，在 split 之后、scatter 之前插入压缩：

```
所有图片 → ViT → Merger → all_image_embeds [1440, 2048]
                                    ↓
         split → 10 tensors, 各 [144, 2048]
                                    ↓
🆕 FiLM Compressor介入:
   (a) 提取指令embedding: mean_pool(text_tokens) → instr_emb [2048]
   (b) 对8张历史帧:
       [8, 144, 2048]
       → proj_in → [8, 144, 512]          # 降维到瓶颈
       → FiLM: γ=Linear(instr_emb) [512]
               β=Linear(instr_emb) [512]
       → Q_cond = γ ⊙ queries + β         # 用指令调制16个learnable queries
       → CrossAttention(Q=Q_cond, KV=proj_in) × 2层
       → [8, 16, 512]                      # 压缩!
       → proj_out → [8, 16, 2048]          # 升维回LLM维度
   (c) 2张非历史帧: 原样保留 [144, 2048]
   (d) 拼接: [128, 2048] + [144, 2048] + [144, 2048] = [416, 2048]
                                    ↓
         masked_scatter 到 inputs_embeds
         (tokenizer已为历史帧分配16个<image_pad>，非144个)
                                    ↓
         LLM decoder → action tokens
```

### 2.3 关键：数据集侧需要同步修改

tokenizer通过 `grid_thw_image` 决定每张图分配多少个 `<image_pad>` token。历史帧压缩后只有16个token，所以数据集 `__getitem__` 中要把历史帧的 `grid_thw_merged` 改为 16（而非原来的144）。这部分已在 `dataset.py.bak` 中实现：

```python
if self.use_compressor and num_history_images > 0:
    for idx_img in range(num_total_images):
        if idx_img < num_history_images:
            grid_thw_merged_for_tokens.append(16)          # 历史帧→16个token
            grid_thws_for_rope.append(torch.tensor([1,4,4]))  # RoPE近似4×4网格
        else:
            grid_thw_merged_for_tokens.append(original_tokens)  # 当前帧不变
```

---

## 3. 核心问题：加了FiLM后该怎么训练？

### 3.1 你之前的质疑（完全合理）

> "先微调compressor再微调视觉层，后微调视觉层的话第一阶段适配原始视觉token的compressor就相当于白训练了"

这个问题的根源是**分阶段训练时ViT特征分布突变**：
```
Stage 1a: 冻结ViT → 训Compressor → Compressor适配了"冻结ViT的特征分布A"
Stage 3:  解冻ViT + LoRA → ViT特征变为"分布B" → Compressor的适配白费
```

### 3.2 解决方案：一阶段联合训练（Joint Training）

**不分阶段。LoRA + Compressor 同时从零/identity开始训练。**

```
初始状态:
  ViT LoRA:   A随机, B=0  →  ΔW=0，ViT行为和预训练完全相同
  FiLM:       γ=1, β=0   →  无调制（identity初始化，已在compressor.py实现）
  LLM LoRA:   A随机, B=0  →  ΔW=0，LLM行为和预训练完全相同

训练前期 (step 0~100):
  LoRA的ΔW接近0 → ViT特征几乎不变
  Compressor相当于在"半冻结ViT + 半冻结LLM"上学习
  ≈ 这自然就是Stage 1a干的事！

训练中后期 (step 100+):
  LoRA的ΔW逐渐增大 → ViT特征逐渐变化
  Compressor同步跟随调整 → 不存在"特征分布突变"
```

**本质：联合训练把分阶段的"离散跳变"变成了"连续渐变"。**

### 3.3 文献支持

| 方法 | 策略 | 对比结论 |
|------|------|---------|
| LLaVA-1.5 Stage 2 | ViT+MLP+LLM联合训练 | 超过分阶段的BLIP-2、InstructBLIP |
| OpenVLA LoRA | 一阶段，LoRA同时覆盖ViT+LLM | 接近Full FT，仅用1.4%参数 |
| BLIP-2分阶段 | 可行，但全程冻结ViT | 特征分布从不变，所以分阶段没问题 |

**关键insight**：BLIP-2分阶段能工作是因为全程冻结ViT（分布永不变）。一旦要解冻ViT（我们的baseline用了ViT LoRA），就必须联合训练。

---

## 4. 具体训练配置

| 参数 | Baseline | FiLM实验 | 说明 |
|------|---------|---------|------|
| ViT LoRA | r=32, α=64, qkv/proj/fc1/fc2 | **完全相同** | 控制变量 |
| LLM LoRA | r=32, α=64, q/k/v/o/gate/up/down | **完全相同** | 控制变量 |
| Merger | LoRA on fc1/fc2 | **完全相同** | 控制变量 |
| **Compressor** | ❌ 不存在 | ✅ 全参训练 10.5M | **唯一新增** |
| 学习率 | 2e-4 | 2e-4 (Compressor也用) | 控制变量 |
| 数据 | R2R+RxR各50%, 2 epochs | **完全相同** | 控制变量 |
| 历史帧tokens | 8×144 = 1152 | 8×16 = 128 | FiLM压缩 |
| GPU | 2× H100 | 2× H100 | 控制变量 |

---

## 5. 需要修改的文件（共3个）

### 文件1: `internnav/trainer/internvla_n1_argument.py`

添加5个compressor参数：

```python
use_compressor: bool = field(default=False)
compressor_d_bottleneck: int = field(default=512)
compressor_n_queries: int = field(default=16)
compressor_n_heads: int = field(default=8)
compressor_n_layers: int = field(default=2)
```

### 文件2: `internnav/trainer/internvla_n1_trainer.py`

在 `apply_lora_to_qwen3vl` 之前插入 `attach_compressor`（**顺序很关键**）：

```python
from internnav.model.compressor_wrapper import attach_compressor

# 在 train() 中:
use_compressor = getattr(model_args, 'use_compressor', False)

if use_lora and data_args.model_type == "qwen3vl":
    if use_compressor:
        # Step 1: 先 attach compressor（monkey-patch inner forward）
        compressor_config = {
            'd_model': model.config.hidden_size,      # 2048
            'd_bottleneck': model_args.compressor_d_bottleneck,  # 512
            'n_queries': model_args.compressor_n_queries,         # 16
            'n_heads': model_args.compressor_n_heads,             # 8
            'n_layers': model_args.compressor_n_layers,           # 2
        }
        model = attach_compressor(model, compressor_config)
        data_args.use_compressor = True
        data_args.compressor_n_queries = compressor_config['n_queries']

    # Step 2: 再 apply LoRA（wrap outer forward）
    model = apply_lora_to_qwen3vl(model, model_args)

    if use_compressor:
        # Step 3: 确保 compressor 参数可训练（PEFT可能把它冻结了）
        for name, param in model.named_parameters():
            if "compressor" in name:
                param.requires_grad = True
```

**顺序说明**：compressor_wrapper patch的是 `model.model.forward`（inner），PEFT wrap的是 `model.forward`（outer）。先patch inner再wrap outer，两者互不干扰。

训练结束后保存compressor权重：

```python
if use_compressor:
    from safetensors.torch import save_file
    base = trainer.model
    if hasattr(base, 'module'): base = base.module
    if hasattr(base, 'base_model'): base = base.base_model.model
    if hasattr(base, 'compressor'):
        comp_state = {k: v.cpu() for k, v in base.compressor.state_dict().items()}
        save_file(comp_state, os.path.join(output_dir, 'compressor.safetensors'))
```

### 文件3: `internnav/dataset/internvla_n1_lerobot_dataset.py`

将 `dataset.py.bak` 中的compressor支持代码合并进来（已有实现，直接迁移）：
- `__init__` 读取 `use_compressor` / `compressor_n_queries`
- `__getitem__` 根据 `use_compressor` 修改历史帧的 `grid_thw_merged`
- Collator中打包 `num_history_images` 和 `image_grid_thw_rope`

---

## 6. 训练脚本（完整版）

```bash
#!/bin/bash
# FiLM Compressor 验证实验 - Qwen3-VL-2B
# 唯一变量: 添加 FiLM Compressor。其余和 Compressor-Baseline-2B 完全一致。
set -e
export TRANSFORMERS_TORCH_LOAD_IS_SAFE=1
export CUDA_VISIBLE_DEVICES=0,5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /data/houdekai/miniconda3/bin/activate internnav
export WANDB_MODE=online
cd /data/houdekai/InternNav_

MASTER_ADDR=localhost
MASTER_PORT=$((RANDOM % 101 + 20001))
NUM_GPUS=2
deepspeed=scripts/train/qwenvl_train/zero2.json
llm=/data/houdekai/models/Qwen3-VL-2B-Instruct
vln_datasets="r2r_125cm_0_30%50,rxr_125cm_0_30%50"

# 和baseline完全一致的参数
batch_size=16; grad_accum_steps=4
use_lora=True; lora_r=32; lora_alpha=64; lora_dropout=0.05
lr=2e-4; mm_projector_lr=2e-4; vision_tower_lr=2e-4
min_pixels=3136; max_pixels=313600
num_history=8; sample_step=4; num_epochs=2

# 🆕 唯一新增：Compressor配置
use_compressor=True
compressor_n_queries=16
compressor_d_bottleneck=512
compressor_n_heads=8
compressor_n_layers=2

output_dir="checkpoints/FiLM-2B-AllLoRA-r32-R2XR50"
run_name="FiLM_2B_Joint_R2XR50"
mkdir -p ${output_dir}

torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    internnav/trainer/internvla_n1_trainer.py \
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --vln_dataset_use ${vln_datasets} \
    --data_flatten False \
    --tune_mm_vision True --tune_mm_mlp True --tune_mm_llm True \
    --use_lora ${use_lora} \
    --lora_r ${lora_r} --lora_alpha ${lora_alpha} --lora_dropout ${lora_dropout} \
    --use_compressor ${use_compressor} \
    --compressor_n_queries ${compressor_n_queries} \
    --compressor_d_bottleneck ${compressor_d_bottleneck} \
    --compressor_n_heads ${compressor_n_heads} \
    --compressor_n_layers ${compressor_n_layers} \
    --bf16 True \
    --num_history ${num_history} \
    --data_augmentation True \
    --resize_h 384 --resize_w 384 \
    --sample_step ${sample_step} \
    --num_future_steps 4 --predict_step_num 32 \
    --pixel_goal_only False --system1 "none" \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} --min_pixels ${min_pixels} \
    --eval_strategy "no" \
    --save_strategy "steps" --save_steps 500 --save_total_limit 3 \
    --learning_rate ${lr} \
    --mm_projector_lr ${mm_projector_lr} \
    --vision_tower_lr ${vision_tower_lr} \
    --weight_decay 0.01 --warmup_ratio 0.03 \
    --max_grad_norm 1.0 --lr_scheduler_type "cosine" \
    --logging_steps 1 --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers True \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 2 \
    --seed 42 --data_seed 42 \
    --run_name ${run_name} --report_to wandb
```

---

## 7. 关于学习率：Compressor归属哪个lr组？

`qwenvl_base.py` 的 `create_optimizer` 按名称分组：
- 含 `"merger"` → `mm_projector_lr`
- 含 `"visual"` → `vision_tower_lr`
- 其他 → `base_lr`（即 `learning_rate=2e-4`）

Compressor参数名为 `base_model.model.compressor.xxx`，**不含 merger 也不含 visual**，因此自动归入 `base_lr=2e-4`。**无需修改 `create_optimizer`。**

---

## 8. 对比实验设计

| 实验 | 描述 | 历史帧tokens | 可训练参数 | 预期 |
|------|------|-------------|-----------|------|
| **Baseline** | AllLoRA r=32，无压缩 | 8×144=1152 | 39.7M | SR≈49~63% |
| **FiLM-16** | +FiLM(n_q=16) 联合训练 | 8×16=128 | 50.2M | SR≥baseline？ |
| **FiLM-32** | +FiLM(n_q=32) | 8×32=256 | ~50.5M | 如FiLM-16不够用 |
| **FiLM-8** | +FiLM(n_q=8) | 8×8=64 | ~50.0M | 极限压缩测试 |

**判断标准**：
- FiLM-16 ≥ Baseline：✅ FiLM有效（压缩了冗余，用指令引导信息选择）
- FiLM-16 ≈ Baseline（差距<2%）：✅ FiLM无损压缩，可用于加速
- FiLM-16 << Baseline（差距>5%）：❌ 16个token太少，尝试FiLM-32

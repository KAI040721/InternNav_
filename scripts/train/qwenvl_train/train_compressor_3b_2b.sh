#!/bin/bash

# ============================================
# Compressor Stage 3b (方案B) - LLM LoRA + Compressor (2B模型)
#
# 实验证据链:
#   Stage 1a: 只训 Compressor → loss 0.85 收敛 (信息压缩上限)
#   Stage 2:  + Merger投影层  → loss 不降    (瓶颈不在 Merger)
#   Stage 3b: + LLM LoRA      → 验证 LLM 能否利用压缩 token
#
# 策略:
#   1. 加载 Stage 1a 的 Compressor 权重 (已学会压缩)
#   2. 冻结: ViT, Merger, DeepStack, embed_tokens, lm_head
#   3. 训练: LLM LoRA (r=32) + Compressor
#   4. History deepstack: zeros (同 Stage 1a)
#
# GPU: H100 x 2 (GPU 6,7)
# DeepSpeed: ZeRO-2
# ============================================

set -e
export TRANSFORMERS_TORCH_LOAD_IS_SAFE=1

# GPU 6,7 — Task B
export CUDA_VISIBLE_DEVICES=6,7

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/houdekai/miniconda3/bin/activate internnav

export WANDB_MODE=online

cd /data/houdekai/InternNav_

# ============ 模型与数据 ============
llm="/data/houdekai/models/Qwen3-VL-2B-Instruct"

# 与 Task A 对齐: R2R 50% + RxR 50%
vln_datasets="r2r_125cm_0_30%50,rxr_125cm_0_30%50"

# ============ 分布式配置 ============
NUM_GPUS=2
MASTER_ADDR="localhost"
MASTER_PORT=20202

# ============ DeepSpeed ============
deepspeed="scripts/train/qwenvl_train/zero2.json"

# ============ Compressor 参数 ============
use_compressor=True
compressor_stage="3b"
compressor_n_queries=16
compressor_d_bottleneck=512
compressor_n_layers=2
compressor_n_heads=8

# Stage 1a checkpoint (Compressor 权重来源)
stage1a_checkpoint="checkpoints/Compressor-1a-Qwen3-2B-R2R-RxR"

# ============ LoRA 参数 (与 Task A 一致) ============
lora_r=32
lora_alpha=64
lora_dropout=0.05

# ============ 训练超参 ============
lr=2e-4
batch_size=32
grad_accum_steps=2
# global batch = 32 x 2 x 2 = 128

max_pixels=313600
min_pixels=3136

num_history=8
sample_step=4
num_epochs=1

output_dir="checkpoints/Compressor-3b-Qwen3-2B-R2R-RxR"
run_name="Compressor_Stage3b_2B_LLM_LoRA"

mkdir -p ${output_dir}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔬 Compressor Stage 3b - 方案B (Qwen3-VL-2B, GPU 6,7)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  基础模型: Qwen3-VL-2B-Instruct"
echo "  Stage1a ckpt: ${stage1a_checkpoint}"
echo "  可训练: LLM LoRA (r=${lora_r}) + Compressor (~10.5M)"
echo "  冻结: ViT, Merger, DeepStack, embed_tokens, lm_head"
echo "  压缩: 144 tokens/frame → ${compressor_n_queries} tokens/frame"
echo "  数据: R2R 50% + RxR 50%"
echo "  Batch: ${batch_size} x ${NUM_GPUS} x ${grad_accum_steps} = $(($batch_size * $NUM_GPUS * $grad_accum_steps))"
echo "  LR: ${lr}"
echo "  LoRA: r=${lora_r}, alpha=${lora_alpha}"
echo "  Epochs: ${num_epochs}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

export NCCL_TIMEOUT=3600
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600

torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} --rdzv_conf timeout=3600 \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    internnav/trainer/internvla_n1_trainer.py \
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --vln_dataset_use ${vln_datasets} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --use_lora False \
    --use_compressor ${use_compressor} \
    --compressor_stage ${compressor_stage} \
    --compressor_n_queries ${compressor_n_queries} \
    --compressor_d_bottleneck ${compressor_d_bottleneck} \
    --compressor_n_layers ${compressor_n_layers} \
    --compressor_n_heads ${compressor_n_heads} \
    --compressor_stage1a_checkpoint ${stage1a_checkpoint} \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --bf16 True \
    \
    --num_history ${num_history} \
    --data_augmentation True \
    --resize_h 384 \
    --resize_w 384 \
    --sample_step ${sample_step} \
    --num_future_steps 4 \
    --predict_step_num 32 \
    --pixel_goal_only False \
    --system1 "none" \
    \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --remove_unused_columns False \
    --save_total_limit 3 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers True \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 2 \
    --seed 42 \
    --data_seed 42 \
    --run_name ${run_name} \
    --report_to wandb \
    --ddp_timeout 7200

echo "✅ Compressor Stage 3b 训练完成! 模型: ${output_dir}"

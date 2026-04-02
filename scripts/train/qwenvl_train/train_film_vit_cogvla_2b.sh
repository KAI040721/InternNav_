#!/bin/bash

# ============================================
# CogVLA-aligned FiLM ViT Compressor 联合训练 - Qwen3-VL-2B
#
# 架构 (CogVLA-inspired):
#   - FiLM conditioning: 所有 24 层 ViT block, 独立参数 (不共享)
#     x = x * (1 + scale(instr)) + shift(instr), zero-init
#   - Aggregation Tokens: 16 learnable tokens, ViT 内 self-attention 压缩
#   - 历史帧: 196 merged tokens → 16 aggr tokens (12.2x 压缩)
#   - 当前帧: 保持原始 196 merged tokens
#
# 微调策略:
#   - LoRA r=32: ViT qkv/proj/fc1/fc2 + LLM q/k/v/o_proj, gate/up/down_proj
#   - FiLM (102.8M): 全参训练, LoRA 后 re-unfreeze
#   - 可训练参数: ~39.7M (LoRA) + ~102.8M (FiLM+Aggr) ≈ 142.5M
#
# 数据: R2R + RxR (各50%)
# Effective batch: 16 * 4 GPUs * 3 grad_accum = 192
#
# 速度: ~4.4x 加速 (LLM 序列 1914→474, -75%)
# 显存: +0.7 GB/GPU 静态 (H100 80GB 充裕)
#
# GPU: H100 x 4
# DeepSpeed: ZeRO-2
# ============================================

set -e
export TRANSFORMERS_TORCH_LOAD_IS_SAFE=1

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/houdekai/miniconda3/bin/activate internnav
export WANDB_MODE=online

cd /data/houdekai/InternNav_

MASTER_ADDR=localhost
MASTER_PORT=$((RANDOM % 101 + 20001))
NUM_GPUS=4

deepspeed=scripts/train/qwenvl_train/zero2.json
llm=/data/houdekai/models/Qwen3-VL-2B-Instruct
vln_datasets="r2r_125cm_0_30%50,rxr_125cm_0_30%50"

# Training
batch_size=16
grad_accum_steps=3  # 16 * 4 * 3 = 192 effective batch

# LoRA
use_lora=True
lora_r=32
lora_alpha=64
lora_dropout=0.05

# Compressor (CogVLA-aligned)
use_compressor=True
compressor_type=film_vit
compressor_n_queries=16       # aggr tokens per history image
compressor_n_film_layers=24   # all ViT blocks (same as CogVLA)
compressor_share_film=False   # independent per block (same as CogVLA)

# LR
lr=2e-4
mm_projector_lr=2e-4
vision_tower_lr=2e-4

# Image
min_pixels=3136
max_pixels=313600

# Data
num_history=8
sample_step=4
num_epochs=2

output_dir="checkpoints/FiLM-ViT-CogVLA-2B-AllLoRA-r32-R2XR50"
run_name="FiLM_ViT_CogVLA_2B_R2XR50"

mkdir -p ${output_dir}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 CogVLA-aligned FiLM ViT Compressor (Qwen3-VL-2B)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  LoRA r=${lora_r} + FiLM 24层 (102.8M) + Aggr 16 tokens"
echo "  Batch: ${batch_size} x ${NUM_GPUS} x ${grad_accum_steps} = $(($batch_size * $NUM_GPUS * $grad_accum_steps))"
echo "  LR: ${lr}, Epochs: ${num_epochs}"
echo "  LLM seq: ~474 tokens (75% shorter than baseline ~1914)"
echo "  Output: ${output_dir}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv | head -10
echo ""

export NCCL_TIMEOUT=3600
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600

torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    internnav/trainer/internvla_n1_trainer.py \
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --vln_dataset_use ${vln_datasets} \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --use_lora ${use_lora} \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --use_compressor ${use_compressor} \
    --compressor_type ${compressor_type} \
    --compressor_n_queries ${compressor_n_queries} \
    --compressor_n_film_layers ${compressor_n_film_layers} \
    --compressor_share_film ${compressor_share_film} \
    --bf16 True \
    --num_history ${num_history} \
    --data_augmentation True \
    --resize_h 384 \
    --resize_w 384 \
    --sample_step ${sample_step} \
    --num_future_steps 4 \
    --predict_step_num 32 \
    --pixel_goal_only False \
    --system1 "none" \
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
    --save_total_limit 3 \
    --learning_rate ${lr} \
    --mm_projector_lr ${mm_projector_lr} \
    --vision_tower_lr ${vision_tower_lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
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
    --remove_unused_columns False \
    --report_to wandb

echo ""
echo "✅ CogVLA-aligned FiLM 训练完成! 模型: ${output_dir}"
echo "   compressor 权重: ${output_dir}/compressor_film.safetensors"

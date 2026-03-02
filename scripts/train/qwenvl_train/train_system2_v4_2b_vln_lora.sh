#!/bin/bash

# ============================================
# System2 v4 - VLN优化版微调训练脚本 (2B模型)
# 
# v4 微调策略:
#   1. Merger + Deepstack Mergers (x3) 全部解冻全参微调
#   2. embed_tokens + lm_head 解冻全参微调
#   3. ViT attention 用 LoRA, ViT MLP 冻结 (保留通用特征)
#   4. LLM 全部线性层 LoRA
#   5. 所有 Norm 层全参微调
#
# 数据: 50% R2R + 50% RxR
# GPU: H100 x 2 (GPU 0,1)
# DeepSpeed: ZeRO-2
# ============================================

set -e
export TRANSFORMERS_TORCH_LOAD_IS_SAFE=1

# 设置使用的GPU — 任务A用 GPU 4,5
export CUDA_VISIBLE_DEVICES=4,5

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/houdekai/miniconda3/bin/activate internnav

export WANDB_MODE=online

cd /data/houdekai/InternNav_

MASTER_ADDR=localhost
MASTER_PORT=20201
NUM_GPUS=2

deepspeed=scripts/train/qwenvl_train/zero2_optimized.json

# 模型 - 2B
llm=/data/houdekai/models/Qwen3-VL-2B-Instruct

# 数据 - 50% R2R + 50% RxR
vln_datasets="r2r_125cm_0_30%50,rxr_125cm_0_30%50"

# 训练参数
batch_size=16
grad_accum_steps=4

use_lora=True
lora_r=32
lora_alpha=64
lora_dropout=0.05

lr=2e-4
mm_projector_lr=1e-4
vision_tower_lr=2e-4

min_pixels=3136
max_pixels=313600

num_history=8
sample_step=4
num_epochs=2

output_dir="checkpoints/InternVLA-N1-System2-Qwen3-2B-VLN-LoRA-v4"
run_name="InternVLA_N1_System2_2B_v4_vln_lora"

mkdir -p ${output_dir}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 System2 v4 训练 (Qwen3-VL-2B, GPU 4,5)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  模型: Qwen3-VL-2B-Instruct (2127.5M params)"
echo "  数据: 50% R2R + 50% RxR"
echo "  策略: v4 (LoRA r=${lora_r} + merger/deepstack/embed full ft)"
echo "  Batch: ${batch_size} x ${NUM_GPUS} x ${grad_accum_steps} = $(($batch_size * $NUM_GPUS * $grad_accum_steps))"
echo "  LR: ${lr} (projector: ${mm_projector_lr})"
echo "  Epochs: ${num_epochs}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
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
    --report_to wandb

echo "✅ v4 2B 训练完成! 模型: ${output_dir}"

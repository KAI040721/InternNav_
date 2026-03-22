#!/bin/bash

# ============================================
# FiLM Compressor 联合训练脚本 - Qwen3-VL-2B
# 
# 目的: 在 baseline AllLoRA 基础上，添加 FiLM Compressor 模块
#       历史帧走 FiLM 压缩 (primary + deepstack 都走 FiLM)
#       一阶段联合训练: LoRA + Compressor 同时微调
# 
# 微调策略:
#   - LoRA target modules: qkv, proj, fc1, fc2 (ViT) + q/k/v/o_proj, gate/up/down_proj (LLM)
#   - Compressor: ~10.5M 随机初始化参数，全参训练
#   - 可训练参数: ~39.7M (LoRA) + ~10.5M (Compressor) ≈ 50.2M
#
# 数据:
#   - R2R + RxR (各50%采样)
#   - Effective batch size: 16 * 4 GPUs * 2 grad_accum = 128 (与baseline一致)
#
# GPU: H100 x 4 (GPU 0,1,2,3)
# 对比基线: Compressor-Baseline-2B-AllLoRA-r32-R2XR50
# ============================================

set -e
export TRANSFORMERS_TORCH_LOAD_IS_SAFE=1

export CUDA_VISIBLE_DEVICES=0,1,2,3
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

batch_size=16
grad_accum_steps=2  # 16 * 4 * 2 = 128 effective batch

use_lora=True
lora_r=32
lora_alpha=64
lora_dropout=0.05

use_compressor=True
compressor_d_bottleneck=512
compressor_n_queries=16
compressor_n_heads=8
compressor_n_layers=2

lr=2e-4
mm_projector_lr=2e-4
vision_tower_lr=2e-4

min_pixels=3136
max_pixels=313600

num_history=8
sample_step=4
num_epochs=2

output_dir="checkpoints/FiLM-Joint-2B-AllLoRA-r32-R2XR50"
run_name="FiLM_Joint_2B_R2XR50"

mkdir -p ${output_dir}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 FiLM Compressor 联合训练 (Qwen3-VL-2B)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  AllLoRA r=${lora_r} + FiLM Compressor (d_bn=${compressor_d_bottleneck}, n_q=${compressor_n_queries})"
echo "  Batch: ${batch_size} x ${NUM_GPUS} x ${grad_accum_steps} = $(($batch_size * $NUM_GPUS * $grad_accum_steps))"
echo "  LR: ${lr}, Epochs: ${num_epochs}"
echo "  Output: ${output_dir}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv | head -10
echo ""

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
    --compressor_d_bottleneck ${compressor_d_bottleneck} \
    --compressor_n_queries ${compressor_n_queries} \
    --compressor_n_heads ${compressor_n_heads} \
    --compressor_n_layers ${compressor_n_layers} \
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
echo "✅ FiLM Compressor 联合训练完成! 模型: ${output_dir}"

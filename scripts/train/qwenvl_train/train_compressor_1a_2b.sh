#!/bin/bash

# ============================================
# Compressor Stage 1a - 视觉Token压缩实验 (2B模型)
#
# Stage 1a 策略:
#   1. 冻结所有原始参数 (ViT + LLM + Merger + DeepStack)
#   2. 仅训练 Compressor (~10.5M params)
#   3. History帧: primary tokens 144→16, deepstack 设为 zeros
#   4. Current帧 + birdseye: 不变 (full 144 tokens + deepstack)
#
# 配置尽可能与 Task A (v4 LoRA 2B) 对齐:
#   batch=16, grad_accum=4, pixel_goal=False, R2R+RxR
#   gradient_checkpointing=True
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
compressor_stage="1a"
compressor_n_queries=16
compressor_d_bottleneck=512
compressor_n_layers=2
compressor_n_heads=8

# ============ 训练超参 (与 Task A 对齐) ============
lr=5e-4
batch_size=32
grad_accum_steps=2
# global batch = 32 x 2 x 2 = 128 (same effective batch as Task A)

max_pixels=313600
min_pixels=3136

num_history=8
sample_step=4
num_epochs=3

output_dir="checkpoints/Compressor-1a-Qwen3-2B-R2R-RxR"
run_name="Compressor_1a_2B_R2R_RxR_aligned"

mkdir -p ${output_dir}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔬 Compressor Stage 1a (Qwen3-VL-2B, GPU 6,7)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  模型: Qwen3-VL-2B-Instruct (冻结)"
echo "  可训练: Compressor ~10.5M params"
echo "  压缩: 144 tokens/frame → ${compressor_n_queries} tokens/frame"
echo "  数据: R2R 50% + RxR 50% (与 Task A 对齐)"
echo "  策略: Stage 1a (freeze all, train compressor only)"
echo "  Batch: ${batch_size} x ${NUM_GPUS} x ${grad_accum_steps} = $(($batch_size * $NUM_GPUS * $grad_accum_steps))"
echo "  LR: ${lr}"
echo "  Epochs: ${num_epochs}"
echo "  pixel_goal_only: False (与 Task A 对齐)"
echo "  gradient_checkpointing: True"
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

echo "✅ Compressor 1a 训练完成! 模型: ${output_dir}"

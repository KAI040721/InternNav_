#!/bin/bash

# ============================================
# Compressor Stage 3 - 全量训练 (Task A 策略 + Compressor)
#
# 策略 (与 Task A 完全一致 + Compressor):
#   1. 加载 Stage 3b 训练后的 Compressor 权重
#   2. LoRA: ViT attn (qkv/proj) + LLM attn+mlp (q/k/v/o/gate/up/down)
#   3. 全参微调: Merger, Deepstack Mergers, embed_tokens, lm_head, all norms
#   4. 全参微调: Compressor (从 Stage 3b 继续)
#
# GPU: H100 x 2 (GPU 6,7)
# DeepSpeed: ZeRO-2
# ============================================

set -e
export TRANSFORMERS_TORCH_LOAD_IS_SAFE=1

export CUDA_VISIBLE_DEVICES=0,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/houdekai/miniconda3/bin/activate internnav
export WANDB_MODE=online

cd /data/houdekai/InternNav_

MASTER_ADDR=localhost
MASTER_PORT=20203
NUM_GPUS=2

deepspeed=scripts/train/qwenvl_train/zero2_optimized.json
llm=/data/houdekai/models/Qwen3-VL-2B-Instruct
vln_datasets="r2r_125cm_0_30%50,rxr_125cm_0_30%50"

use_lora=True
lora_r=32
lora_alpha=64
lora_dropout=0.05

lr=2e-4
mm_projector_lr=1e-4
vision_tower_lr=2e-4

batch_size=24
grad_accum_steps=3
num_epochs=2

min_pixels=3136
max_pixels=313600
num_history=8
sample_step=4

use_compressor=True
compressor_stage="3"
compressor_stage1a_checkpoint="checkpoints/Compressor-3b-Qwen3-2B-R2R-RxR"

output_dir="checkpoints/Compressor-3-Qwen3-2B-R2R-RxR"
run_name="Compressor_Stage3_2B_Full_TaskA"

mkdir -p ${output_dir}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Compressor Stage 3 训练 (Task A + Compressor)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  模型: Qwen3-VL-2B-Instruct"
echo "  数据: 50% R2R + 50% RxR"
echo "  LoRA: r=${lora_r}, alpha=${lora_alpha}"
echo "  Compressor 初始化: Stage 3b checkpoint"
echo "  Epochs: ${num_epochs}"
echo "  GPU: 0,3 (master_port: ${MASTER_PORT})"
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
    --use_compressor ${use_compressor} \
    --compressor_stage "${compressor_stage}" \
    --compressor_stage1a_checkpoint "${compressor_stage1a_checkpoint}" \
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
    --remove_unused_columns False \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers True \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 2 \
    --seed 42 \
    --data_seed 42 \
    --run_name ${run_name} \
    --resume_from_checkpoint checkpoints/Compressor-3-Qwen3-2B-R2R-RxR/checkpoint-1500 \
    --report_to wandb

echo "Compressor Stage 3 训练完成! 模型: ${output_dir}"

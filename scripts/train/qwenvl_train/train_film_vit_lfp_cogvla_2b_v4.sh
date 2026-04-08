#!/bin/bash

# ============================================
# CogVLA-aligned FiLM ViT Compressor + LFP 联合训练 v4 - Qwen3-VL-2B
#
# v4 changes (training collapse fix):
#   - Fix 1: Per-group LR — new modules (Compressor+Router) get higher LR (1e-3)
#   - Fix 2: Per-group gradient clipping via training_step override
#   - Fix 3: Zero-init TokenRouter + FiLMedTokenRouter
#   - Fix 4: Increased warmup_ratio to 0.05
#   - Fix 5: max_grad_norm raised to 5.0
#
# Root cause of v3 collapse: 285M new params never learned due to
# gradient starvation under unified max_grad_norm=1.0 clipping.
#
# 数据: R2R + RxR (各50%)
# Effective batch: 16 * 4 GPUs * 3 grad_accum = 192
# GPU: H100 x 4, DeepSpeed ZeRO-2
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
grad_accum_steps=3

# LoRA
use_lora=True
lora_r=32
lora_alpha=64
lora_dropout=0.05

# Compressor
use_compressor=True
compressor_type=film_vit
compressor_n_queries=64
compressor_n_film_layers=24
compressor_share_film=False

# LFP
use_lfp=True
lfp_type="shiftedcos_decay_0.85_0.15"
lfp_average_factor=0.5
lfp_enable_film=True

# LR — v4: separate LR for new modules
lr=2e-4
new_module_lr=1e-3
mm_projector_lr=2e-4
vision_tower_lr=2e-4

# Gradient clipping — v4: higher limits
max_grad_norm=5.0
new_module_max_grad_norm=20.0

min_pixels=3136
max_pixels=313600

num_history=8
sample_step=4
num_epochs=2

output_dir="checkpoints/FiLM-ViT-LFP-CogVLA-2B-R2XR50-v4"
run_name="FiLM_ViT_LFP_CogVLA_2B_R2XR50_v4"

mkdir -p ${output_dir}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  CogVLA-aligned FiLM ViT + LFP v4 (Qwen3-VL-2B) [collapse-fix]"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  LoRA r=${lora_r} (lr=${lr}) + New module lr=${new_module_lr}"
echo "  Grad clip: base=${max_grad_norm}, new=${new_module_max_grad_norm}"
echo "  LFP: ${lfp_type}, avg_factor=${lfp_average_factor}"
echo "  Batch: ${batch_size} x ${NUM_GPUS} x ${grad_accum_steps} = $(($batch_size * $NUM_GPUS * $grad_accum_steps))"
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
    --use_lfp ${use_lfp} \
    --lfp_type "${lfp_type}" \
    --lfp_average_factor ${lfp_average_factor} \
    --lfp_enable_film ${lfp_enable_film} \
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
    --new_module_lr ${new_module_lr} \
    --mm_projector_lr ${mm_projector_lr} \
    --vision_tower_lr ${vision_tower_lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --max_grad_norm ${max_grad_norm} \
    --new_module_max_grad_norm ${new_module_max_grad_norm} \
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
echo "FiLM + LFP v4 训练完成! 模型: ${output_dir}"
echo "   compressor 权重: ${output_dir}/compressor_film.safetensors"
echo "   LFP router 权重: ${output_dir}/lfp_router.safetensors"

#!/bin/bash

# ============================================
# CogVLA-aligned FiLM ViT Compressor + LFP 联合训练 v5 - Qwen3-VL-2B
#
# v5 changes (LFP fix + clean params):
#   - FIX: LFP Router 只路由历史帧 tokens，当前帧完整通过
#     (代码修复在 compressor_wrapper_film_vit.py 第580-610行)
#   - 移除 new_module_max_grad_norm（v3/v4中从未生效的死代码）
#   - 移除 v4 的 new_module_lr（基于错误诊断的过高LR）
#   - 恢复 v3/verge3 的保守超参：
#     max_grad_norm=1.0, warmup_ratio=0.03, lr=2e-4 (统一)
#
# 与 verge3 (成功的 pure LoRA) 一致的核心参数：
#   lr=2e-4, max_grad_norm=1.0, warmup=0.03,
#   LoRA r=32 alpha=64 dropout=0.05, cosine schedule
#
# 架构 (CogVLA-inspired, 与 v3 相同):
#   - FiLM conditioning: 24层 ViT block, 独立参数, zero-init (identity)
#   - Aggregation Tokens: 64 learnable tokens, ViT 内 self-attention 压缩
#   - LFP: shiftedcos decay, 层7-26, 只路由历史帧 (修复后)
#   - 历史帧: ViT 576→64 aggr tokens, LLM 内再路由压缩
#   - 当前帧: ViT 保持 144 merged tokens, LFP 不触碰
#
# 数据: R2R + RxR (各50%)
# Effective batch: 16 * 4 GPUs * 3 grad_accum = 192
# GPU: H100 x 4 (GPU 0-3)
# DeepSpeed: ZeRO-2
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

# Training
batch_size=16
grad_accum_steps=3  # 16 * 4 * 3 = 192 effective batch

# LoRA (同 verge3)
use_lora=True
lora_r=32
lora_alpha=64
lora_dropout=0.05

# Compressor (CogVLA-aligned, 同 v3)
use_compressor=True
compressor_type=film_vit
compressor_n_queries=64       # aggr tokens per history image
compressor_n_film_layers=24   # all ViT blocks
compressor_share_film=False   # independent per block

# LFP (CogVLA-aligned, 同 v3 — 代码中已修复只路由历史帧)
use_lfp=True
lfp_type="shiftedcos_decay_0.85_0.15"  # cosine decay, max=0.85, min=0.15
lfp_average_factor=0.5                  # keep 50% visual tokens on average
lfp_enable_film=True                    # FiLM-conditioned router

# LR (统一, 同 verge3)
lr=2e-4
mm_projector_lr=2e-4
vision_tower_lr=2e-4
# 注意: 不使用 new_module_lr, 所有参数统一 lr=2e-4

# Image
min_pixels=3136
max_pixels=313600

# Data
num_history=8
sample_step=4
num_epochs=2

output_dir="checkpoints/FiLM-ViT-LFP-CogVLA-2B-R2XR50-v5"
run_name="FiLM_ViT_LFP_CogVLA_2B_R2XR50_v5"

mkdir -p ${output_dir}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  CogVLA-aligned FiLM ViT + LFP v5 (Qwen3-VL-2B) [LFP-fix]"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  核心修复: LFP 只路由历史帧, 当前帧完整保留"
echo "  LoRA r=${lora_r} alpha=${lora_alpha} + FiLM 24层 + Aggr 64 + LFP"
echo "  LFP: ${lfp_type}, avg_factor=${lfp_average_factor}, film=${lfp_enable_film}"
echo "  Batch: ${batch_size} x ${NUM_GPUS} x ${grad_accum_steps} = $(($batch_size * $NUM_GPUS * $grad_accum_steps))"
echo "  LR: ${lr} (统一), max_grad_norm=1.0, warmup=0.03"
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
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "FiLM + LFP v5 训练完成!"
echo "  模型: ${output_dir}"
echo "  compressor: ${output_dir}/compressor_film.safetensors"
echo "  LFP router: ${output_dir}/lfp_router.safetensors"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

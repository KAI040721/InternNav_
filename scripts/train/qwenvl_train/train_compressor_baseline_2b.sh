#!/bin/bash

# ============================================
# 基线复现训练脚本 - Qwen3-VL-2B
# 
# 目的: 精确复现旧版 InternVLA-N1-System2-Qwen3-2B-AllLoRA-r32-R2XR50%
#       对应日志: logs/train_r2r_2b_20260126_115838.log
#       旧版 git commit: fd025ab
# 
# 微调策略 (Pure LoRA):
#   - LoRA target modules: qkv, proj, fc1, fc2 (ViT) + q/k/v/o_proj, gate/up/down_proj (LLM)
#   - modules_to_save: merger only
#   - Norms: 全参微调
#   - embed_tokens / lm_head: 冻结
#   - 可训练参数: ~39.7M (1.83%)
#
# 数据:
#   - R2R (104870 samples) + RxR (sampling 50%, ~104870 effective)
#   - Effective batch size: 16 * 2 GPUs * 4 grad_accum = 128
#   - Total steps: ~5810 (2 epochs)
#
# GPU: H100 x 2 (GPU 0,3)
# ============================================

set -e
export TRANSFORMERS_TORCH_LOAD_IS_SAFE=1

# 设置使用的GPU
export CUDA_VISIBLE_DEVICES=0,5

# 设置PYTORCH显存分配优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 激活conda环境
source /data/houdekai/miniconda3/bin/activate internnav

# 设置wandb在线模式
export WANDB_MODE=online

# 切换到项目目录
cd /data/houdekai/InternNav_

# 分布式训练配置
MASTER_ADDR=localhost
MASTER_PORT=$((RANDOM % 101 + 20001))
NUM_GPUS=2

# DeepSpeed配置 - 使用旧版 zero2.json (overlap_comm=false)
deepspeed=scripts/train/qwenvl_train/zero2.json

# ============================================
# 模型配置 - 2B
# ============================================
llm=/data/houdekai/models/Qwen3-VL-2B-Instruct

# ============================================
# 训练数据配置 - R2R + RxR 各50%
# ============================================
vln_datasets="r2r_125cm_0_30%50,rxr_125cm_0_30%50"

# ============================================
# 训练参数 (与旧版完全一致)
# ============================================
batch_size=16
grad_accum_steps=4

# LoRA 配置
use_lora=True
lora_r=32
lora_alpha=64
lora_dropout=0.05

# 学习率
lr=2e-4
mm_projector_lr=2e-4
vision_tower_lr=2e-4

# 像素范围
min_pixels=3136
max_pixels=313600

# 历史帧和采样
num_history=8
sample_step=4

# Epoch 数
num_epochs=2

# 输出目录 - 新名称，区分旧版
output_dir="checkpoints/Compressor-Baseline-2B-AllLoRA-r32-R2XR50"
run_name="Compressor_Baseline_2B_R2XR50"

# ============================================
# 创建输出目录
# ============================================
mkdir -p ${output_dir}

# ============================================
# 打印配置信息
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Compressor Baseline 训练开始 (Qwen3-VL-2B)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 模型配置:"
echo "   • 基础模型: Qwen3-VL-2B-Instruct"
echo "   • 策略: Pure LoRA (旧版基线复现)"
echo "   • 输出: ${output_dir}"
echo ""
echo "�� 数据集:"
echo "   • R2R + RxR (各50%采样)"
echo ""
echo "⚙️ 训练参数:"
echo "   • Batch Size: ${batch_size} x ${NUM_GPUS} GPUs x ${grad_accum_steps} grad_accum = $(($batch_size * $NUM_GPUS * $grad_accum_steps))"
echo "   • Learning Rate: ${lr}"
echo "   • LoRA r: ${lora_r}, alpha: ${lora_alpha}"
echo "   • Epochs: ${num_epochs}"
echo "   • Seed: 42, Data Seed: 42"
echo "   • DeepSpeed: zero2.json (overlap_comm=false)"
echo ""
echo "📁 输出: ${output_dir}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "GPU 信息:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv | head -10
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================
# 启动训练
# ============================================
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

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Compressor Baseline 训练完成!"
echo ""
echo "📁 模型保存在: ${output_dir}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

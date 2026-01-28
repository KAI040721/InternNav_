#!/bin/bash

# ============================================
# System2 R2R Only 微调训练脚本 - 4B模型
# 
# 特点：
# 1. 只使用 R2R 数据集
# 2. 添加 seed 和 data_seed 确保分布式一致性
# 3. 基于样本索引的局部RNG已在数据集代码中修复
# 4. 其他参数与 v3 保持一致
# 
# GPU: H100 x 2 (GPU 2,3)
# ============================================

set -e
export TRANSFORMERS_TORCH_LOAD_IS_SAFE=1

# 设置使用的GPU
export CUDA_VISIBLE_DEVICES=2,3

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

# DeepSpeed配置
deepspeed=scripts/train/qwenvl_train/zero2.json

# ============================================
# 模型配置 - 4B 版
# ============================================
llm=/data/houdekai/models/Qwen3-VL-2B-Instruct

# ============================================
# 训练数据配置 - 只使用 R2R
# ============================================
vln_datasets="r2r_125cm_0_30%50,rxr_125cm_0_30%50"

# ============================================
# 训练参数 (与 v3 保持一致)
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

# 输出目录
output_dir="checkpoints/InternVLA-N1-System2-Qwen3-4B-AllLoRA-r32-R2R-Only"
run_name="InternVLA_N1_System2_4B_R2R_Only"

# ============================================
# 创建输出目录
# ============================================
mkdir -p ${output_dir}

# ============================================
# 打印配置信息
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 System2 R2R Only 训练开始 (4B 模型)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 模型配置:"
echo "   • 基础模型: Qwen3-VL-4B-Instruct"
echo "   • 输出: ${output_dir}"
echo ""
echo "📚 数据集:"
echo "   • R2R (125cm_0_30) - 仅此一个"
echo ""
echo "⚙️ 训练参数:"
echo "   • Batch Size: ${batch_size} x ${NUM_GPUS} GPUs x ${grad_accum_steps} grad_accum = $(($batch_size * $NUM_GPUS * $grad_accum_steps))"
echo "   • Learning Rate: ${lr}"
echo "   • LoRA r: ${lora_r}, alpha: ${lora_alpha}"
echo "   • Epochs: ${num_epochs}"
echo "   • Seed: 42, Data Seed: 42"
echo ""
echo "🔧 随机种子修复:"
echo "   • __getitem__ 使用基于样本索引的局部 RNG"
echo "   • item_rng = random.Random(42 + i)"
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
echo "✅ Training R2R Only (4B LoRA) 完成!"
echo ""
echo "📁 模型保存在: ${output_dir}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

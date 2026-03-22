#!/bin/bash

# ============================================
# System2 第三次微调训练脚本 - v3 (4B模型优化LoRA版)
# 
# 优化点：
# 1. gradient_checkpointing=True (LLM 需要保留避免 OOM)
# 2. dataloader_persistent_workers=True - 保持 worker 进程活跃
# 3. dataloader_pin_memory=True - GPU 内存固定加速
# 4. dataloader_prefetch_factor=2 - 预取 2 个 batch
# 5. dataloader_num_workers=16 - 保持不变（已证明有效）
# 
# Vision Tower 的梯度检查点会被代码自动关闭
# 
# GPU: H100 x 4 (GPU 0,1,2,3)
# ============================================

set -e

# 设置使用的GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

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
NUM_GPUS=4

# DeepSpeed配置
deepspeed=scripts/train/qwenvl_train/zero2.json

# ============================================
# 模型配置 - 4B 版
# ============================================
llm=/data/houdekai/models/Qwen3-VL-4B-Instruct

# ============================================
# 训练数据配置
# ============================================
vln_datasets="r2r_125cm_0_30,rxr_125cm_0_30,scalevln_125cm_0_30%50"

# ============================================
# 训练参数 (参考 v2 脚本配置)
# ============================================
# 全局 batch size = 4 * 12 * 3 = 144 (与 v2 保持一致)
batch_size=12
grad_accum_steps=3

# LoRA 配置保持一致
use_lora=True
lora_r=32
lora_alpha=64
lora_dropout=0.05

# 学习率 - 参考 v2 的 2e-4 统一学习率
lr=2e-4
mm_projector_lr=2e-4
vision_tower_lr=2e-4

# 像素范围 - 参考 v2 的配置
min_pixels=3136
max_pixels=313600

# 历史帧和采样 - 参考 v2
num_history=8
sample_step=4

# Epoch 数：用户指定 2 epoch
num_epochs=2

# 输出目录
output_dir="checkpoints/InternVLA-N1-System2-Qwen3-4B-AllLoRA-r32-v3"
run_name="InternVLA_N1_System2_4B_v3_mixed_data"

# ============================================
# 创建输出目录
# ============================================
mkdir -p ${output_dir}

# ============================================
# 打印配置信息
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 System2 v3 训练开始 (4B 模型优化LoRA版)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 模型配置:"
echo "   • 基础模型: Qwen3-VL-4B-Instruct"
echo "   • 输出: InternVLA-N1-System2-Qwen3-4B-AllLoRA-r32-v3"
echo ""
echo "📚 数据集:"
echo "   • R2R (125cm_0_30)"
echo "   • RxR (125cm_0_30)"
echo "   • ScaleVLN (125cm_0_30) ⭐ 新增大规模数据"
echo ""
echo "⚙️ 训练参数:"
echo "   • Batch Size: ${batch_size} x ${NUM_GPUS} GPUs x ${grad_accum_steps} grad_accum = $(($batch_size * $NUM_GPUS * $grad_accum_steps))"
echo "   • Learning Rate: ${lr}"
echo "   • LoRA r: ${lora_r}, alpha: ${lora_alpha}"
echo "   • Epochs: ${num_epochs}"
echo ""
echo "⚡ 性能优化:"
echo "   • gradient_checkpointing: True (LLM 保留避免 OOM)"
echo "   • Vision Tower Grad Ckpt: 自动关闭 ✅"
echo "   • dataloader_num_workers: 16 (保持)"
echo "   • dataloader_persistent_workers: True ✅"
echo "   • dataloader_pin_memory: True ✅"
echo "   • dataloader_prefetch_factor: 2 ✅"
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
# 启动训练 (Pure LoRA + 数据加载优化)
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
    --dataloader_num_workers 16 \
    --dataloader_persistent_workers True \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 2 \
    --run_name ${run_name} \
    --report_to wandb

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Training v3 (4B LoRA) 完成!"
echo ""
echo "📁 模型保存在: ${output_dir}"
echo ""
echo "🔍 下一步: 运行评估脚本"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ============================================
# [新增] 分布式训练通信优化 (解决 GPU 卡顿)
# ============================================

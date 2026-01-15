#!/bin/bash

# ============================================
# System2 微调训练脚本 - Qwen3-VL-8B 全LoRA版本
# GPU: H100 x 4 (GPU 4,5,6,7)
# 有效全局batch size: 128
# LoRA配置: 所有参数全部用LoRA微调（冻结所有原始参数）
# 梯度检查点: 仅LLM启用
# ============================================

# 设置使用的GPU (4,5,6,7号卡)
export CUDA_VISIBLE_DEVICES=4,5,6,7

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
# 模型配置 - 使用本地Qwen3-VL-8B模型
# ============================================
llm=/data/houdekai/models/Qwen3-VL-8B-Instruct

# ============================================
# 数据配置 - VLN-CE R2R数据
# 使用所有R2R数据集配置
# ============================================
vln_datasets=r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30,rxr_125cm_0_30,rxr_125cm_0_45,rxr_60cm_15_15,rxr_60cm_30_30

# ============================================
# 训练超参数
# 有效全局batch size = NUM_GPUS * batch_size * grad_accum_steps
# 128 = 4 * 8 * 4
# ============================================
batch_size=8
grad_accum_steps=4
num_epochs=2

# 学习率配置 - 全部使用LoRA，统一学习率
lr=2e-4                   # 统一的LoRA微调学习率
mm_projector_lr=2e-4      # Projector也用LoRA
vision_tower_lr=2e-4      # Vision Tower也用LoRA

# 图像分辨率配置
max_pixels=313600         # 28*28*400
min_pixels=3136

# ============================================
# LoRA配置 - 所有模块全部用LoRA（冻结所有原始参数）
# tune_mm_vision/mlp/llm 都设为 True 表示对这些模块应用LoRA
# ============================================
use_lora=True
lora_r=32
lora_alpha=64
lora_dropout=0.05

# ============================================
# 历史帧和采样配置
# ============================================
num_history=8
sample_step=6

# ============================================
# 输出配置
# ============================================
run_name=InternVLA-N1-System2-Qwen3-8B-AllLoRA-r32
output_dir=checkpoints/${run_name}

# 显示GPU信息
echo "=== GPU Information ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv | head -10
echo "========================"
echo "Training Configuration:"
echo "  Model: ${llm}"
echo "  Dataset: ${vln_datasets}"
echo "  Batch Size: ${batch_size} x ${NUM_GPUS} GPUs x ${grad_accum_steps} grad_accum = $((batch_size * NUM_GPUS * grad_accum_steps))"
echo "  LoRA: r=${lora_r}, alpha=${lora_alpha} (All modules with LoRA)"
echo "  Learning Rate: ${lr} (unified for all modules)"
echo "========================"

# 创建输出目录
mkdir -p ${output_dir}

# 启动训练
# tune_mm_vision/mlp/llm=True 表示这些模块参与训练（通过LoRA）
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
    --dataloader_num_workers 24 \
    --run_name ${run_name} \
    --report_to wandb

echo "========================"
echo "Training completed!"
echo "Checkpoints saved in: ${output_dir}"
echo "========================"

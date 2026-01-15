#!/bin/bash

# ============================================
# System2 微调训练脚本 - 4×H100
# 基准对齐：第一版双卡脚本（除 LoRA + TF32）
# 变化：加入 RxR；关闭 gradient checkpointing
# 有效全局batch size: 64 = 4 * 8 * 2
# ============================================

# 4卡配置
export CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_GPUS=4

# 显存分配优化（保持原有）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# TF32（按你要求：属于“TF32部分”，保持开启）
export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# 激活环境
source /data/houdekai/miniconda3/bin/activate internnav
export WANDB_MODE=online

# 项目目录
cd /data/houdekai/InternNav_

# 分布式训练配置
MASTER_ADDR=localhost
MASTER_PORT=$((RANDOM % 101 + 20001))

# DeepSpeed（对齐第一版：用 zero2.json）
deepspeed=scripts/train/qwenvl_train/zero2.json

# 模型
llm=/data/houdekai/models/Qwen3-VL-8B-Instruct

# 数据集：第一版4个R2R + 加上4个RxR（8 configs）
vln_datasets=r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30,rxr_125cm_0_30,rxr_125cm_0_45,rxr_60cm_15_15,rxr_60cm_30_30

# 训练超参（对齐第一版的有效全局 batch=64）
batch_size=8
grad_accum_steps=2
num_epochs=2

# 学习率（对齐第一版）
lr=2e-4
mm_projector_lr=2e-5
vision_tower_lr=2e-5

# 图像分辨率/像素配置（对齐第一版）
max_pixels=313600
min_pixels=3136

# 历史帧与采样（对齐第一版）
num_history=4
sample_step=6

# LoRA（属于“LoRA部分”，不强制对齐第一版；保留你4卡脚本的明确 target）
use_lora=True
lora_r=16
lora_alpha=32
lora_dropout=0.05
lora_bias=none
lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# 输出
run_name=InternVLA-N1-System2-Qwen3-8B-4H100-alignV1+RxR-noCkpt
output_dir=checkpoints/${run_name}
mkdir -p ${output_dir}

echo "============================================"
echo "Hardware:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -${NUM_GPUS}
echo "============================================"
echo "Config:"
echo "  GPUs: ${NUM_GPUS}"
echo "  Datasets: ${vln_datasets}"
echo "  Effective batch: ${NUM_GPUS} * ${batch_size} * ${grad_accum_steps} = $((NUM_GPUS * batch_size * grad_accum_steps))"
echo "  num_history=${num_history}, sample_step=${sample_step}"
echo "  Gradient checkpointing: OFF"
echo "============================================"

torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} \
  --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
  internnav/trainer/internvla_n1_trainer.py \
  --deepspeed ${deepspeed} \
  --model_name_or_path "${llm}" \
  --vln_dataset_use ${vln_datasets} \
  --data_flatten False \
  \
  --tune_mm_vision True \
  --tune_mm_mlp True \
  --tune_mm_llm True \
  --use_lora ${use_lora} \
  --lora_r ${lora_r} \
  --lora_alpha ${lora_alpha} \
  --lora_dropout ${lora_dropout} \
  --lora_target_modules "${lora_target_modules}" \
  --lora_bias ${lora_bias} \
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
  --gradient_checkpointing False \
  --dataloader_num_workers 8 \
  --run_name ${run_name} \
  --report_to wandb

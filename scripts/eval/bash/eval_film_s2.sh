#!/bin/bash

# ============================================
# 评估脚本 - FiLM Joint Training 模型
# LoRA + Compressor (FiLM) 在 r2r val_unseen 上评估
# ============================================

# 设置使用的 GPU（根据实际空闲情况修改）
export CUDA_VISIBLE_DEVICES=6

# 激活conda环境
source /data/houdekai/miniconda3/bin/activate intern_habitat

# 切换到项目目录
cd /data/houdekai/InternNav_

echo "========================================"
echo "评估 FiLM Joint Training 模型"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "基座模型: /data/houdekai/models/Qwen3-VL-2B-Instruct"
echo "LoRA adapter: checkpoints/FiLM-Joint-2B-AllLoRA-r32-R2XR50"
echo "Compressor: compressor_film.safetensors"
echo "数据集: r2r val_unseen (1839 episodes)"
echo "========================================"

# 创建日志目录
mkdir -p ./logs/habitat/eval_film_joint_s2

# 运行评估
python scripts/eval/eval.py --config scripts/eval/configs/eval_film_s2_cfg.py

echo "========================================"
echo "FiLM 模型评估完成!"
echo "结果保存在: ./logs/habitat/eval_film_joint_s2"
echo "========================================"

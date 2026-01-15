#!/bin/bash

# ============================================
# 评估脚本 - 微调的 Qwen3-8B LoRA 模型
# 在 GPU 7 上运行
# ============================================

# 设置使用 GPU 7
export CUDA_VISIBLE_DEVICES=7

# 激活conda环境
source /data/houdekai/miniconda3/bin/activate intern_habitat

# 切换到项目目录
cd /data/houdekai/InternNav_

echo "========================================"
echo "开始评估微调的 Qwen3-8B LoRA 模型"
echo "GPU: 7"
echo "模型路径: checkpoints/InternVLA-N1-System2-Qwen3-8B-LoRA-r32"
echo "========================================"

# 创建日志目录
mkdir -p ./logs/habitat/eval_lora_s2

# 运行评估
python scripts/eval/eval.py --config scripts/eval/configs/eval_lora_s2_cfg.py

echo "========================================"
echo "LoRA模型评估完成!"
echo "结果保存在: ./logs/habitat/eval_lora_s2"
echo "========================================"

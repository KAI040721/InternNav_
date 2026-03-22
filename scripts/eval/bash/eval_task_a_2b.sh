#!/bin/bash
# ============================================
# 评估 Task A: Qwen3-2B LoRA v4 baseline
# GPU 0
# ============================================
export CUDA_VISIBLE_DEVICES=0
source /data/houdekai/miniconda3/bin/activate intern_habitat
cd /data/houdekai/InternNav_

echo "========================================"
echo "开始评估 Task A (2B LoRA v4 baseline)"
echo "GPU: 0"
echo "模型: checkpoints/InternVLA-N1-System2-Qwen3-2B-VLN-LoRA-v4"
echo "========================================"

mkdir -p ./logs/habitat/eval_task_a_2b

python scripts/eval/eval.py --config scripts/eval/configs/eval_task_a_2b_cfg.py

echo "========================================"
echo "Task A 评估完成!"
echo "结果: ./logs/habitat/eval_task_a_2b"
echo "========================================"

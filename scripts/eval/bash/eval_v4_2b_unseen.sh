#!/bin/bash

# ============================================
# 评估脚本 - Qwen3-2B LoRA v4 (Task A baseline)
# 在 GPU 4 上运行
# ============================================

export CUDA_VISIBLE_DEVICES=4
source /data/houdekai/miniconda3/bin/activate intern_habitat
cd /data/houdekai/InternNav_

mkdir -p ./logs/habitat/eval_v4_2b_unseen
mkdir -p ./logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./logs/eval_v4_2b_unseen_${TIMESTAMP}.log"

echo "========================================"
echo "开始 LoRA v4 模型评估 - 2B (val_unseen)"
echo "日志文件: $LOG_FILE"
echo "GPU: 4"
echo "模型: checkpoints/InternVLA-N1-System2-Qwen3-2B-VLN-LoRA-v4"
echo "========================================"

nohup python scripts/eval/eval.py --config scripts/eval/configs/eval_v4_2b_unseen_cfg.py > "$LOG_FILE" 2>&1 &

EVAL_PID=$!
echo "评估进程已启动，PID: $EVAL_PID"
echo "查看日志: tail -f $LOG_FILE"
echo "停止评估: kill $EVAL_PID"
echo "========================================"

#!/bin/bash

# ============================================
# 同时评估两个模型的脚本
# - 官方模型在 GPU 5
# - LoRA模型在 GPU 7
# ============================================

echo "========================================"
echo "启动两个评估任务..."
echo "========================================"

cd /data/houdekai/InternNav_

# 启动官方模型评估（后台运行）
echo "[1/2] 启动官方模型评估 (GPU 5)..."
nohup bash scripts/eval/bash/eval_official_s2.sh > logs/eval_official_s2.log 2>&1 &
PID1=$!
echo "      PID: $PID1"
echo "      日志: logs/eval_official_s2.log"

# 启动LoRA模型评估（后台运行）
echo "[2/2] 启动LoRA模型评估 (GPU 7)..."
nohup bash scripts/eval/bash/eval_lora_s2.sh > logs/eval_lora_s2.log 2>&1 &
PID2=$!
echo "      PID: $PID2"
echo "      日志: logs/eval_lora_s2.log"

echo "========================================"
echo "两个评估任务已在后台启动!"
echo ""
echo "监控进度:"
echo "  tail -f logs/eval_official_s2.log  # 官方模型"
echo "  tail -f logs/eval_lora_s2.log      # LoRA模型"
echo ""
echo "查看GPU使用:"
echo "  watch -n 1 nvidia-smi"
echo "========================================"

#!/bin/bash

# ============================================
# 评估脚本 - 微调的 Qwen3-8B LoRA 模型
# 在 GPU 6 上运行
# ============================================

# 设置使用 GPU 6
export CUDA_VISIBLE_DEVICES=1

# 激活conda环境
source /data/houdekai/miniconda3/bin/activate intern_habitat

# 切换到项目目录
cd /data/houdekai/InternNav_

# 创建日志目录
mkdir -p ./logs/habitat/eval_alllora_s2_2b
mkdir -p ./logs

# 生成时间戳用于日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./logs/eval_lora_s2unseen2b_${TIMESTAMP}.log"

echo "========================================"
echo "开始 LoRA 模型评估 - System2"
echo "日志文件: $LOG_FILE"
echo "========================================"

# 运行评估，使用 nohup 在后台运行并输出日志
nohup python scripts/eval/eval.py --config scripts/eval/configs/eval_lora_s2_unseen_cfg.py > "$LOG_FILE" 2>&1 &

# 获取后台进程 PID
EVAL_PID=$!
echo "评估进程已启动，PID: $EVAL_PID"
echo "查看日志: tail -f $LOG_FILE"
echo "停止评估: kill $EVAL_PID"
echo "========================================"

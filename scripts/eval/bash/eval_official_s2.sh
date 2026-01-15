#!/bin/bash

# ============================================
# 评估脚本 - 官方 InternVLA-N1-System2 模型
# 在 GPU 6 上运行（从5卡切换）
# ============================================

# 设置使用 GPU 5
export CUDA_VISIBLE_DEVICES=6

# 激活conda环境
source /data/houdekai/miniconda3/bin/activate intern_habitat

# 切换到项目目录
cd /data/houdekai/InternNav_

echo "========================================"
echo "继续评估官方 InternVLA-N1-System2 模型"
echo "GPU: 6 (从GPU 5切换)"
echo "模型路径: checkpoints/InternVLA-N1-System2"
echo "========================================"

# 创建日志目录
mkdir -p ./logs/habitat/eval_official_s2

# 运行评估
python scripts/eval/eval.py --config scripts/eval/configs/eval_official_s2_cfg.py

echo "========================================"
echo "官方模型评估完成!"
echo "结果保存在: ./logs/habitat/eval_official_s2"
echo "========================================"

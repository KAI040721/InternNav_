#!/bin/bash

echo "========================================"
echo "开始评估官方 InternVLA-N1-System2 模型 (R2R val_seen)"
echo "GPU: 5"
echo "模型路径: checkpoints/InternVLA-N1-System2"
echo "========================================"

export CUDA_VISIBLE_DEVICES=5

/data/houdekai/miniconda3/envs/intern_habitat/bin/python scripts/eval/eval.py \
    --config scripts/eval/configs/eval_official_s2_seen_cfg.py

echo "========================================"
echo "官方模型评估完成!"
echo "结果保存在: ./logs/habitat/eval_official_s2_seen"
echo "========================================"

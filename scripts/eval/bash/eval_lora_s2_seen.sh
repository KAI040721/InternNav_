#!/bin/bash

echo "========================================"
echo "开始评估微调的 Qwen3-8B LoRA 模型 (R2R val_seen)"
echo "GPU: 7"
echo "模型路径: checkpoints/InternVLA-N1-System2-Qwen3-8B-LoRA-r32"
echo "========================================"

export CUDA_VISIBLE_DEVICES=7

/data/houdekai/miniconda3/envs/intern_habitat/bin/python scripts/eval/eval.py \
    --config scripts/eval/configs/eval_lora_s2_seen_cfg.py

echo "========================================"
echo "LoRA模型评估完成!"
echo "结果保存在: ./logs/habitat/eval_lora_s2_seen"
echo "========================================"

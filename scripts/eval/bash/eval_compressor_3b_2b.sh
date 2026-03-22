#!/bin/bash
# ============================================
# 评估 Compressor Stage 3b: Qwen3-2B LoRA + Compressor
# GPU 3
# ============================================
export CUDA_VISIBLE_DEVICES=3
source /data/houdekai/miniconda3/bin/activate intern_habitat
cd /data/houdekai/InternNav_

echo "========================================"
echo "开始评估 Compressor Stage 3b (2B LoRA + Compressor)"
echo "GPU: 3"
echo "模型: checkpoints/Compressor-3b-Qwen3-2B-R2R-RxR"
echo "Compressor: checkpoints/Compressor-3b-Qwen3-2B-R2R-RxR/compressor_stage3b.safetensors"
echo "========================================"

mkdir -p ./logs/habitat/eval_compressor_3b_2b

python scripts/eval/eval.py --config scripts/eval/configs/eval_compressor_3b_2b_cfg.py

echo "========================================"
echo "Compressor Stage 3b 评估完成!"
echo "结果: ./logs/habitat/eval_compressor_3b_2b"
echo "========================================"

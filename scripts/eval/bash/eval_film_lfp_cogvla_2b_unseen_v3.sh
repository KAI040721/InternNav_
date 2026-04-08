#!/bin/bash
# ============================================
# 评估脚本 - FiLM-ViT + LFP CogVLA 2B v3 (7-bug-fix)
# 在 R2R val_unseen 上评估
# ============================================

export CUDA_VISIBLE_DEVICES=0

source /data/houdekai/miniconda3/bin/activate intern_habitat

cd /data/houdekai/InternNav_

echo "========================================"
echo "评估 FiLM-ViT + LFP CogVLA 2B v3 模型"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "基座模型: /data/houdekai/models/Qwen3-VL-2B-Instruct"
echo "Checkpoint: checkpoints/FiLM-ViT-LFP-CogVLA-2B-R2XR50-v3"
echo "数据集: R2R val_unseen"
echo "========================================"

mkdir -p ./logs/habitat/eval_film_lfp_cogvla_2b_unseen_v3

python scripts/eval/eval.py --config scripts/eval/configs/eval_film_lfp_cogvla_2b_unseen_v3_cfg.py

echo "========================================"
echo "v3 模型评估完成!"
echo "结果保存在: ./logs/habitat/eval_film_lfp_cogvla_2b_unseen_v3"
echo "========================================"

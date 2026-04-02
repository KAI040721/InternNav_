#!/bin/bash

# ============================================
# 评估脚本 - FiLM-ViT + LFP CogVLA 2B 模型
# LoRA + Compressor (FiLM-ViT) + LFP Router
# 在 R2R val_unseen 上评估
# ============================================

# 设置使用的 GPU（根据实际空闲情况修改）
export CUDA_VISIBLE_DEVICES=0

# 激活conda环境
source /data/houdekai/miniconda3/bin/activate intern_habitat

# 切换到项目目录
cd /data/houdekai/InternNav_

echo "========================================"
echo "评估 FiLM-ViT + LFP CogVLA 2B 模型"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "基座模型: /data/houdekai/models/Qwen3-VL-2B-Instruct"
echo "LoRA adapter: checkpoints/FiLM-ViT-LFP-CogVLA-2B-R2XR50"
echo "Compressor: compressor_film.safetensors (FiLM-ViT, n_aggr=64)"
echo "LFP Router: lfp_router.safetensors (shiftedcos_decay, FiLM)"
echo "数据集: R2R val_unseen (1839 episodes)"
echo "========================================"

# 创建日志目录
mkdir -p ./logs/habitat/eval_film_lfp_cogvla_2b_unseen

# 运行评估
python scripts/eval/eval.py --config scripts/eval/configs/eval_film_lfp_cogvla_2b_unseen_cfg.py

echo "========================================"
echo "FiLM-ViT + LFP 模型评估完成!"
echo "结果保存在: ./logs/habitat/eval_film_lfp_cogvla_2b_unseen"
echo "========================================"

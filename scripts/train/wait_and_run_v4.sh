#!/bin/bash
# =============================================
# 等待至少4张GPU空闲（>90GB free）后自动启动v4训练
# 每5分钟检查一次，日志输出到stdout
# =============================================

TRAIN_SCRIPT="/data/houdekai/InternNav_/scripts/train/qwenvl_train/train_film_vit_lfp_cogvla_2b_v4.sh"
REQUIRED_GPUS=4
FREE_THRESHOLD=90000  # MiB

echo "=========================================="
echo " GPU空闲监控 - 等待 ${REQUIRED_GPUS} 张 GPU (>${FREE_THRESHOLD}MiB free)"
echo " 训练脚本: ${TRAIN_SCRIPT}"
echo " 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

while true; do
    free_gpus=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -v thresh="$FREE_THRESHOLD" '$2 > thresh {print $1}' \
        | tr -d ',' | head -${REQUIRED_GPUS})
    
    num_free=$(echo "$free_gpus" | grep -c '[0-9]' || true)
    
    if [ "$num_free" -ge "$REQUIRED_GPUS" ]; then
        gpu_list=$(echo "$free_gpus" | head -${REQUIRED_GPUS} | tr '\n' ',' | sed 's/,$//')
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 检测到 ${num_free} 张空闲GPU: ${gpu_list}"
        echo "启动v4训练..."
        
        export CUDA_VISIBLE_DEVICES="${gpu_list}"
        bash "${TRAIN_SCRIPT}"
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 训练脚本执行完毕，退出监控。"
        exit 0
    else
        echo "[$(date '+%H:%M:%S')] 空闲GPU数: ${num_free}/${REQUIRED_GPUS}，继续等待..."
        sleep 300
    fi
done

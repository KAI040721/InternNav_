#!/bin/bash
# ==================================================
# 监控 v2 训练进度并验证 Layer 27 权重
# ==================================================
CKPT_DIR="/data/houdekai/InternNav_/checkpoints/FiLM-ViT-LFP-CogVLA-2B-R2XR50-v2"
LOG_FILE="/data/houdekai/InternNav_/logs/train_film_vit_lfp_cogvla_2b_v2.log"

echo "v2 训练监控器 - 每60秒检查 checkpoint"

while true; do
    STEP_COUNT=$(grep -c "'loss'" "$LOG_FILE")
    LAST=$(grep "'epoch'" "$LOG_FILE" | tail -1 | grep -oP "'epoch': [0-9.]+")
    echo "[$(date '+%H:%M:%S')] Steps: ${STEP_COUNT} | ${LAST}"

    CKPT_DIRS=$(find "$CKPT_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null)
    
    if [ -n "$CKPT_DIRS" ]; then
        LATEST_CKPT=$(echo "$CKPT_DIRS" | sort -V | tail -1)
        echo "发现 checkpoint: $LATEST_CKPT"
        sleep 15  # wait for write completion
        
        LFP_FILE=$(find "$LATEST_CKPT" -name "lfp_router.safetensors" 2>/dev/null | head -1)
        if [ -n "$LFP_FILE" ]; then
            echo "验证 Layer 27 权重..."
            python3 -c "
import safetensors.torch as st
weights = st.load_file('${LFP_FILE}')
keys = sorted(weights.keys())
print('\\n所有 key:')
for k in keys[-30:]: print(f'  {k}: norm={weights[k].float().norm().item():.6f}')
l27 = [k for k in keys if '27' in k]
l26 = [k for k in keys if '26' in k]
print('\\n--- Layer 27 ---')
for k in l27: print(f'  {k}: norm={weights[k].float().norm().item():.6f}')
print('\\n--- Layer 26 ---')
for k in l26: print(f'  {k}: norm={weights[k].float().norm().item():.6f}')
any_nonzero = any(weights[k].float().norm().item() > 0 for k in l27)
print('\\n✅ Layer 27 已训练!' if any_nonzero else '\\n❌ Layer 27 仍为零!')
"
            exit 0
        fi
    fi
    sleep 60
done

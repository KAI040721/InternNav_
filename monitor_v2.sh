#!/bin/bash
CKPT_DIR="/data/houdekai/InternNav_/checkpoints/FiLM-ViT-LFP-CogVLA-2B-R2XR50-v2"
LOG_FILE="/data/houdekai/InternNav_/logs/train_film_vit_lfp_cogvla_2b_v2.log"
echo "v2 monitor started"
while true; do
    STEP_COUNT=$(grep -c "'loss'" "$LOG_FILE")
    echo "[$(date '+%H:%M:%S')] Steps: ${STEP_COUNT}"
    CKPT_DIRS=$(find "$CKPT_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null)
    if [ -n "$CKPT_DIRS" ]; then
        LATEST_CKPT=$(echo "$CKPT_DIRS" | sort -V | tail -1)
        echo "Found checkpoint: $LATEST_CKPT"
        sleep 15
        LFP_FILE=$(find "$LATEST_CKPT" -name "lfp_router.safetensors" 2>/dev/null | head -1)
        if [ -n "$LFP_FILE" ]; then
            python3 << PYEOF
import safetensors.torch as st
weights = st.load_file("$LFP_FILE")
keys = sorted(weights.keys())
l27 = [k for k in keys if '27' in k]
l26 = [k for k in keys if '26' in k]
print("Layer 27:")
for k in l27: print(f"  {k}: norm={weights[k].float().norm().item():.6f}")
print("Layer 26:")
for k in l26: print(f"  {k}: norm={weights[k].float().norm().item():.6f}")
any_nz = any(weights[k].float().norm().item() > 0 for k in l27)
print("RESULT: Layer 27 trained!" if any_nz else "RESULT: Layer 27 still zero!")
PYEOF
            exit 0
        fi
    fi
    sleep 60
done

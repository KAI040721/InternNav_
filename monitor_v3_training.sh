#!/bin/bash
LOG_FILE="/data/houdekai/InternNav_/logs/train_v3_resume_20260402_223450.log"
CKPT_DIR="/data/houdekai/InternNav_/checkpoints/FiLM-ViT-LFP-CogVLA-2B-R2XR50-v3"
MONITOR_LOG="/data/houdekai/InternNav_/logs/monitor_v3.log"
INTERVAL=3600

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MONITOR_LOG"; }

# 提取 loss 行（处理 tqdm 前缀）
extract_loss_lines() {
    grep "'loss'" "$LOG_FILE" | sed "s/.*{'loss'/{'loss'/"
}

report() {
    echo "" | tee -a "$MONITOR_LOG"
    log "========================================"
    log "         v3 续训进度报告"
    log "========================================"

    TOTAL_STEPS=$(extract_loss_lines | wc -l)
    LAST_ENTRY=$(extract_loss_lines | tail -1)
    LAST_LOSS=$(echo "$LAST_ENTRY"  | grep -oP "(?<='loss': )[0-9.]+")
    LAST_GN=$(echo "$LAST_ENTRY"    | grep -oP "(?<='grad_norm': )[0-9.]+")
    LAST_LR=$(echo "$LAST_ENTRY"    | grep -oP "(?<='learning_rate': )[0-9e.+-]+")
    LAST_EPOCH=$(echo "$LAST_ENTRY" | grep -oP "(?<='epoch': )[0-9.]+")

    log "续训 loss 记录步数: ${TOTAL_STEPS}"
    log "当前 epoch: ${LAST_EPOCH}"
    log "最新 loss: ${LAST_LOSS}"
    log "最新 grad_norm: ${LAST_GN}"
    log "最新 lr: ${LAST_LR}"

    AVG_LOSS=$(extract_loss_lines | tail -30 | grep -oP "(?<='loss': )[0-9.]+" \
               | awk '{s+=$1;n++} END {if(n>0) printf "%.4f",s/n}')
    MAX_GN=$(extract_loss_lines | tail -100 | grep -oP "(?<='grad_norm': )[0-9.]+" \
             | sort -g | tail -1)
    log "近30步均loss: ${AVG_LOSS}  |  近100步最大grad_norm: ${MAX_GN}"

    # 续训从 checkpoint-1000 开始，累计 = 1000 + 续训新步数
    RESUME_BASE=1000
    TOTAL_CUMULATIVE=$((RESUME_BASE + TOTAL_STEPS))
    log "累计总步数 (checkpoint-1000 + 续训): ${TOTAL_CUMULATIVE}"

    if pgrep -fa "torchrun" | grep -q "nproc_per_node=4"; then
        log "训练进程: 运行中 ✓"
    else
        log "训练进程: ⚠ 未检测到！"
    fi

    log "GPU使用 (0-3):"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
        --format=csv,noheader,nounits 2>/dev/null | head -4 | \
    while IFS= read -r line; do log "  GPU $line"; done

    log "Checkpoints:"
    for ckpt in $(ls -d "${CKPT_DIR}"/checkpoint-* 2>/dev/null | sort -V); do
        CNAME=$(basename "$ckpt")
        DS_DIR=$(find "$ckpt" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
        if [ -n "$DS_DIR" ] && [ "$(ls -A "$DS_DIR" 2>/dev/null)" ]; then
            log "  [完整DS] $CNAME"
        else
            log "  [仅LoRA] $CNAME"
        fi
    done

    REMAIN=$((3894 - TOTAL_CUMULATIVE))
    log "预估剩余: ~${REMAIN} 步 (总~3894步/2epoch)"
    log "========================================"
    echo "" | tee -a "$MONITOR_LOG"
}

# 清空旧 log
> "$MONITOR_LOG"

log "v3 monitor 启动 (每 ${INTERVAL} 秒汇报)"
report
while true; do
    sleep "$INTERVAL"
    report
done

#!/bin/bash
# ============================================
# GPU 监控 + 自动启动 V5 训练
# 
# 功能:
#   1. 每 30 秒检查 GPU 0-3 的显存使用
#   2. 当 GPU 0-3 全部空闲 (显存 < 1GB) 时:
#      a. 先运行 V3 checkpoint eval (快速验证 LFP 修复)
#      b. 然后启动 V5 训练
#   3. 日志输出到终端和 log 文件
# ============================================

set -e

LOG_DIR="/data/houdekai/InternNav_/logs"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/wait_and_run_v5.log"

EVAL_SCRIPT="/data/houdekai/InternNav_/scripts/eval/bash/eval_film_lfp_cogvla_2b_unseen_v3.sh"
TRAIN_SCRIPT="/data/houdekai/InternNav_/scripts/train/qwenvl_train/train_film_vit_lfp_cogvla_2b_v5.sh"

# 显存阈值 (MiB) — 低于此值视为空闲
MEM_THRESHOLD=1000

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

check_gpus_free() {
    # 检查 GPU 0-3 是否全部空闲
    local all_free=true
    for gpu_id in 0 1 2 3; do
        local mem_used
        mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${gpu_id} 2>/dev/null | tr -d ' ')
        if [[ -z "$mem_used" ]] || [[ "$mem_used" -ge "$MEM_THRESHOLD" ]]; then
            all_free=false
            break
        fi
    done
    $all_free
}

get_gpu_status() {
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader -i 0,1,2,3 2>/dev/null
}

log "=========================================="
log "GPU 监控启动 — 等待 GPU 0-3 空闲"
log "显存阈值: < ${MEM_THRESHOLD} MiB"
log "Eval 脚本: ${EVAL_SCRIPT}"
log "Train 脚本: ${TRAIN_SCRIPT}"
log "=========================================="

# 主循环
POLL_INTERVAL=30
while true; do
    if check_gpus_free; then
        log "✓ GPU 0-3 全部空闲!"
        log "当前 GPU 状态:"
        get_gpu_status | while read line; do log "  $line"; done

        # Step 1: 快速验证 — eval V3 checkpoint (使用修复后的代码)
        log ""
        log "=========================================="
        log "Step 1: 快速验证 — Eval V3 checkpoint + LFP fix"
        log "=========================================="
        
        if [[ -f "$EVAL_SCRIPT" ]]; then
            log "运行: bash ${EVAL_SCRIPT}"
            bash "${EVAL_SCRIPT}" 2>&1 | tee -a "$LOG_FILE"
            EVAL_EXIT=$?
            if [[ $EVAL_EXIT -ne 0 ]]; then
                log "⚠ Eval 退出码: ${EVAL_EXIT} (可能有问题，但继续训练)"
            else
                log "✓ Eval 完成"
            fi
        else
            log "⚠ Eval 脚本不存在: ${EVAL_SCRIPT}, 跳过验证"
        fi

        # Step 2: 启动 V5 训练
        log ""
        log "=========================================="
        log "Step 2: 启动 V5 训练"
        log "=========================================="
        
        # 再次确认 GPU 空闲 (eval 可能用了一些显存)
        sleep 10
        if check_gpus_free; then
            log "GPU 0-3 仍然空闲，启动训练..."
            log "运行: bash ${TRAIN_SCRIPT}"
            bash "${TRAIN_SCRIPT}" 2>&1 | tee -a "$LOG_FILE"
            TRAIN_EXIT=$?
            log "V5 训练退出码: ${TRAIN_EXIT}"
            log "=========================================="
            log "监控脚本完成"
            log "=========================================="
            exit $TRAIN_EXIT
        else
            log "⚠ Eval 后 GPU 未释放，继续等待..."
        fi
    else
        # 获取简要状态
        SUMMARY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 0,1,2,3 2>/dev/null | tr '\n' ' | ')
        log "GPU 0-3 仍在使用: ${SUMMARY%| }  — ${POLL_INTERVAL}s 后重试"
    fi
    sleep ${POLL_INTERVAL}
done

#!/bin/bash
# 自动重启 eval，直到 1839 episodes 全部完成
PROGRESS_FILE="/data/houdekai/InternNav_/logs/habitat/eval_film_lfp_cogvla_2b_unseen_v3/progress.json"
TOTAL_EPISODES=1839
MAX_RETRIES=50
EVAL_SCRIPT="scripts/eval/bash/eval_film_lfp_cogvla_2b_unseen_v3.sh"
LOG_DIR="/data/houdekai/InternNav_/logs"

retry=0
while [ $retry -lt $MAX_RETRIES ]; do
    # 检查已完成的 episode 数
    if [ -f "$PROGRESS_FILE" ]; then
        done_count=$(wc -l < "$PROGRESS_FILE")
    else
        done_count=0
    fi

    if [ "$done_count" -ge "$TOTAL_EPISODES" ]; then
        echo "[$(date)] 评估完成！共 $done_count episodes"
        break
    fi

    retry=$((retry + 1))
    echo "[$(date)] 第 ${retry} 次运行（已完成 ${done_count}/${TOTAL_EPISODES} episodes）..."

    LOG_FILE="${LOG_DIR}/eval_v3_auto_resume_r${retry}_$(date +%Y%m%d_%H%M%S).log"
    bash "$EVAL_SCRIPT" > "$LOG_FILE" 2>&1
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        done_now=$(wc -l < "$PROGRESS_FILE" 2>/dev/null || echo 0)
        echo "[$(date)] 进程退出 (code=$exit_code)，已完成 ${done_now} episodes。10秒后重启..."
        sleep 10
    fi
done

echo "[$(date)] eval_v3_auto_resume 结束 (retries=${retry})"

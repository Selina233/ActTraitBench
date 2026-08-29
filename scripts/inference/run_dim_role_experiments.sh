#!/bin/bash

# 批量运行维度角色注入实验
# 用法: ./run_dim_role_experiments.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 定义模型列表
MODELS=(
    "deepseek-v3.2"
    "deepseek-v3.1-250821"
    "deepseek-v3"
    "glm-5"
    "minimax-m2.5"
    "gpt-4o"
    "deepseek-v4-pro"
    "gemini-3.1-pro-preview"
    "claude-sonnet-4-6"
    "qwen3-1.7b"
    "qwen3-8b"
    "qwen3-32b"
    "qwen3-235b-a22b-thinking-2507"
    "deepseek-v4-flash"
)

# 输出目录
OUTPUT_DIR="./dim_role_result"

# 日志文件
LOG_FILE="./dim_role_batch_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUTPUT_DIR"

echo "=================================" | tee -a "$LOG_FILE"
echo "维度角色注入实验 - 批量运行" | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"
echo "共 ${#MODELS[@]} 个模型" | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

SUCCESS_MODELS=()
FAILED_MODELS=()

for idx in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$idx]}"
    MODEL_NUM=$((idx + 1))

    echo "=================================" | tee -a "$LOG_FILE"
    echo "[${MODEL_NUM}/${#MODELS[@]}] 运行模型: $MODEL_NAME" | tee -a "$LOG_FILE"
    echo "开始时间: $(date)" | tee -a "$LOG_FILE"
    echo "=================================" | tee -a "$LOG_FILE"

    python dim_role_inference.py \
        --model "$MODEL_NAME" \
        --output "$OUTPUT_DIR" \
        --questions-dir . \
        --role-prompts ./dim_role_prompt.json \
        2>&1 | tee -a "$LOG_FILE"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ $MODEL_NAME 完成" | tee -a "$LOG_FILE"
        SUCCESS_MODELS+=("$MODEL_NAME")
    else
        echo "❌ $MODEL_NAME 失败" | tee -a "$LOG_FILE"
        FAILED_MODELS+=("$MODEL_NAME")
    fi

    echo "结束时间: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    if [ $MODEL_NUM -lt ${#MODELS[@]} ]; then
        echo "等待 5 秒后继续..." | tee -a "$LOG_FILE"
        sleep 5
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"
echo "批量实验完成" | tee -a "$LOG_FILE"
echo "结束时间: $(date)" | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "成功: ${#SUCCESS_MODELS[@]}/${#MODELS[@]}" | tee -a "$LOG_FILE"

if [ ${#SUCCESS_MODELS[@]} -gt 0 ]; then
    echo "  成功的模型:" | tee -a "$LOG_FILE"
    for model in "${SUCCESS_MODELS[@]}"; do
        echo "    ✅ $model" | tee -a "$LOG_FILE"
    done
fi

echo "" | tee -a "$LOG_FILE"
echo "失败: ${#FAILED_MODELS[@]}/${#MODELS[@]}" | tee -a "$LOG_FILE"
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "  失败的模型:" | tee -a "$LOG_FILE"
    for model in "${FAILED_MODELS[@]}"; do
        echo "    ❌ $model" | tee -a "$LOG_FILE"
    done
fi

echo "" | tee -a "$LOG_FILE"
echo "结果保存在: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"

if [ ${#FAILED_MODELS[@]} -eq 0 ]; then
    exit 0
else
    exit 1
fi

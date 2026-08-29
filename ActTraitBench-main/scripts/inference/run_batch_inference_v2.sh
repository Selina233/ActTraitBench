#!/bin/bash

# 批量运行模型实验脚本 V2
# 支持3次重复实验 (temperature=0, seed=42)
# 用法: nohup ./run_batch_inference_v2.sh > batch_run.out 2>&1 &

# 设置工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 定义模型列表
MODELS=(
    "deepseek-r1"
    "deepseek-v3.2"
    "deepseek-v3.1-250821"
    "deepseek-v3"
    "glm-5"
    "kimi-k2.5"
    "minimax-m2.5"
    "ernie-4.5-turbo-32k"
    "ernie-4.5-turbo-128k"
)

# 输出目录基础路径
OUTPUT_BASE_DIR="./results"

# 日志文件
LOG_FILE="./batch_inference_v2_$(date +%Y%m%d_%H%M%S).log"

# 创建输出目录
mkdir -p "$OUTPUT_BASE_DIR"

# 开始记录
echo "=================================" | tee -a "$LOG_FILE"
echo "批量模型实验开始 V2 (3次重复实验)" | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"
echo "共 ${#MODELS[@]} 个模型，每个模型3次实验" | tee -a "$LOG_FILE"
echo "配置: temperature=0, seed=42" | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 记录成功和失败的模型
SUCCESS_MODELS=()
FAILED_MODELS=()

# 遍历所有模型
for idx in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$idx]}"
    MODEL_NUM=$((idx + 1))
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_NAME}"

    echo "=================================" | tee -a "$LOG_FILE"
    echo "[${MODEL_NUM}/${#MODELS[@]}] 运行模型: $MODEL_NAME" | tee -a "$LOG_FILE"
    echo "输出目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
    echo "开始时间: $(date)" | tee -a "$LOG_FILE"
    echo "将运行3次重复实验 (results-1.json, results-2.json, results-3.json)" | tee -a "$LOG_FILE"
    echo "=================================" | tee -a "$LOG_FILE"

    # 运行模型（3次实验会自动在脚本内部完成）
    python qianfan_models.inference.py \
        --model "$MODEL_NAME" \
        --output "$OUTPUT_DIR" \
        2>&1 | tee -a "$LOG_FILE"

    # 检查退出状态
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ $MODEL_NAME 完成（3次实验）" | tee -a "$LOG_FILE"
        SUCCESS_MODELS+=("$MODEL_NAME")

        # 验证3个结果文件是否都存在
        MISSING_FILES=0
        for run_id in 1 2 3; do
            if [ ! -f "$OUTPUT_DIR/results-${run_id}.json" ]; then
                echo "  ⚠️  缺少 results-${run_id}.json" | tee -a "$LOG_FILE"
                MISSING_FILES=$((MISSING_FILES + 1))
            fi
        done

        if [ $MISSING_FILES -eq 0 ]; then
            echo "  ✅ 3个实验结果文件完整" | tee -a "$LOG_FILE"
        else
            echo "  ⚠️  缺少 $MISSING_FILES 个实验结果文件" | tee -a "$LOG_FILE"
        fi
    else
        echo "❌ $MODEL_NAME 失败" | tee -a "$LOG_FILE"
        FAILED_MODELS+=("$MODEL_NAME")
    fi

    echo "结束时间: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # 在模型之间添加短暂延迟，避免API压力过大
    if [ $MODEL_NUM -lt ${#MODELS[@]} ]; then
        echo "等待 10 秒后继续下一个模型..." | tee -a "$LOG_FILE"
        sleep 10
    fi
done

# 输出最终汇总
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
echo "完整日志已保存: $LOG_FILE" | tee -a "$LOG_FILE"

# 显示结果文件统计
echo "" | tee -a "$LOG_FILE"
echo "结果文件统计:" | tee -a "$LOG_FILE"
for model in "${SUCCESS_MODELS[@]}"; do
    MODEL_DIR="${OUTPUT_BASE_DIR}/${model}"
    RESULT_COUNT=$(ls -1 "$MODEL_DIR"/results-*.json 2>/dev/null | wc -l)
    echo "  $model: $RESULT_COUNT/3 个结果文件" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "下一步: 运行分析脚本计算分数和标准差" | tee -a "$LOG_FILE"
echo "  python batch_calculate_scores_v3.py $OUTPUT_BASE_DIR" | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"

# 返回状态码
if [ ${#FAILED_MODELS[@]} -eq 0 ]; then
    exit 0
else
    exit 1
fi

#!/bin/bash

# 批量运行模型实验脚本 - 缓解方法（自我反思）
# 用法: nohup ./run_batch_inference_reflection.sh > reflection_batch.out 2>&1 &

# 设置工作目录
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

# 输出目录基础路径
OUTPUT_BASE_DIR="./缓解方法result"

# 日志文件
LOG_FILE="./reflection_batch_$(date +%Y%m%d_%H%M%S).log"

# 创建输出目录
mkdir -p "$OUTPUT_BASE_DIR"

# 开始记录
echo "=================================" | tee -a "$LOG_FILE"
echo "批量模型实验 - 缓解方法（自我反思）" | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"
echo "共 ${#MODELS[@]} 个模型，每个模型3次实验" | tee -a "$LOG_FILE"
echo "配置: temperature=0, seed=42/43/44" | tee -a "$LOG_FILE"
echo "方法: 行为测试题前添加自我反思 + 完整BFI-2测试" | tee -a "$LOG_FILE"
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
    echo "运行3次实验 (results-1/2/3.json)" | tee -a "$LOG_FILE"
    echo "=================================" | tee -a "$LOG_FILE"

    # 运行模型（3次实验会自动在脚本内部完成）
    python qianfan_models.inference_reflection.py \
        --model "$MODEL_NAME" \
        --output "$OUTPUT_DIR" \
        2>&1 | tee -a "$LOG_FILE"

    # 检查退出状态
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ $MODEL_NAME 完成" | tee -a "$LOG_FILE"
        SUCCESS_MODELS+=("$MODEL_NAME")

        # 验证结果文件是否存在
        if [ -f "$OUTPUT_DIR/results-1.json" ] && [ -f "$OUTPUT_DIR/results-2.json" ] && [ -f "$OUTPUT_DIR/results-3.json" ]; then
            echo "  ✅ 实验结果文件完整 (3个)" | tee -a "$LOG_FILE"
        else
            echo "  ⚠️  部分结果文件缺失" | tee -a "$LOG_FILE"
            [ ! -f "$OUTPUT_DIR/results-1.json" ] && echo "    - 缺少 results-1.json" | tee -a "$LOG_FILE"
            [ ! -f "$OUTPUT_DIR/results-2.json" ] && echo "    - 缺少 results-2.json" | tee -a "$LOG_FILE"
            [ ! -f "$OUTPUT_DIR/results-3.json" ] && echo "    - 缺少 results-3.json" | tee -a "$LOG_FILE"
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
    if [ -f "$MODEL_DIR/results-1.json" ] && [ -f "$MODEL_DIR/results-2.json" ] && [ -f "$MODEL_DIR/results-3.json" ]; then
        echo "  $model: ✓ results-1/2/3.json (3个)" | tee -a "$LOG_FILE"
    else
        echo "  $model: ⚠️ 部分缺失" | tee -a "$LOG_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"

# 返回状态码
if [ ${#FAILED_MODELS[@]} -eq 0 ]; then
    exit 0
else
    exit 1
fi

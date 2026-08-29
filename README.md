# ActTraitBench: A Benchmark for LLM Personality Consistency

[中文版](#中文版) | English

This repository provides the complete evaluation dataset, replication code for three core experiments, and benchmark reference results for the ActTraitBench framework.

## Repository Structure

```
ActTraitBench/
├── data/                          # Read-only data files
│   ├── behavior_questions.json    # Behavioral test questions (12 facets)
│   ├── bfi2_questions.json        # BFI-2 questionnaire (60 items)
│   ├── dim_role_prompt.json       # Dimension role injection prompts
│   ├── gpt_calibration_params.json # GPT scoring calibration parameters
│   └── data_with_gpt54_scores.json # Human calibration data (94 participants, anonymized)
│
├── reference_results/             # Official experimental results from paper (read-only)
│   ├── experiment1_baseline/      # Experiment 1: Baseline
│   │   └── results/               # Raw results from 14 models × 3 runs
│   ├── experiment2_dim_role/      # Experiment 2: Dimension role injection
│   │   └── dim_role_result/       # High/low trait results from 14 models
│   └── experiment3_mitigation/    # Experiment 3: Mitigation experiment
│       └── reflection_results/    # Self-reflection results from 14 models × 3 runs
│
├── outputs/                       # Analysis output directory (includes sample outputs)
│   ├── experiment1_baseline/      # Experiment 1 analysis results
│   ├── experiment2_dim_role/      # Experiment 2 analysis results
│   └── experiment3_mitigation/    # Experiment 3 analysis results
│
└── scripts/                       # Analysis and inference scripts
    ├── batch_calculate_scores_v2.py      # Base calculation module
    ├── batch_calculate_scores_v4_multi.py # V4 calibration (11 significant facets, multi-run support)
    ├── analyze_dim_role_results_v2.py    # Dimension role experiment analysis
    ├── compare_reflection_corrected_v4.py # Mitigation experiment comparison (V4 version)
    ├── experiment1_baseline.py           # Experiment 1 runner
    ├── experiment2_dim_role.py           # Experiment 2 runner
    ├── experiment3_mitigation_multi.py   # Experiment 3 runner (3 runs)
    └── inference/                        # Model inference scripts
        ├── qianfan_models.inference.py            # Experiment 1 inference
        ├── qianfan_models.inference_reflection.py # Experiment 3 inference
        ├── dim_role_inference.py                  # Experiment 2 inference
        ├── run_batch_inference_v2.sh              # Batch run experiment 1
        ├── run_batch_inference_reflection.sh      # Batch run experiment 3
        └── run_dim_role_experiments.sh            # Batch run experiment 2
```

## Requirements

**Python Version**: >= 3.8

Install dependencies:
```bash
pip install -r requirements.txt
```

Core dependencies:
- pandas >= 1.3.0
- numpy >= 1.21.0
- openpyxl >= 3.0.9
- openai >= 1.0.0 (if running inference scripts)

## Quick Start

### Experiment 1: Baseline

Calculate BFI-2 self-report scores and behavioral test scores for 14 models using percentile calibration on 11 significant facets (interval mapping method).

```bash
cd scripts
python experiment1_baseline.py
```

**Output Files**:
- `outputs/experiment1_baseline/scores_summary_with_std_v4.csv` - Summary table (with mean and std)

**Output Content**:
- BFI Big Five dimension scores (mean ± std)
- Behavioral test Big Five dimension scores (mean ± std)
- Say-Do gap (MSE, RMSE)

### Experiment 2: Dimension Role Injection

Analyze personality performance differences across models under high/low trait role injection conditions.

```bash
cd scripts
python experiment2_dim_role.py
```

**Output Files**:
- `outputs/experiment2_dim_role/dim_role_summary_full_v2.csv` - High/low trait analysis results
- `outputs/experiment2_dim_role/dim_role_summary_full_v2.xlsx` - Excel format

**Output Content**:
- High trait mean (target=4) and low trait mean (target=2) for each dimension
- High-low trait difference and statistical significance

### Experiment 3: Mitigation Experiment (Self-reflection)

Compare say-do gap changes between original experiment and experiment with self-reflection prompts.

```bash
cd scripts
python experiment3_mitigation_multi.py
```

**Output Files**:
- `outputs/experiment3_mitigation/comparison_reflection_corrected_v4.csv` - Mitigation effect comparison
- `outputs/experiment3_mitigation/comparison_reflection_corrected_v4.xlsx` - Excel format
- `outputs/experiment3_mitigation/scores_summary_with_std_v4.csv` - Mitigation experiment score summary

**Output Content**:
- Original vs. mitigation experiment MSE comparison
- MSE improvement amount and improvement percentage
- Say-do gap changes for each dimension

## Experiment Description

### Experiment 1: Baseline

- **Input**: Raw results from 14 models × 3 runs in `reference_results/experiment1_baseline/results/`
- **Method**: V4 percentile calibration (interval mapping on 11 significant facets), calculates mean(MSE₁, MSE₂, MSE₃) across 3 runs
- **Output**: BFI scores, behavioral scores, MSE and RMSE (mean ± std) for each model

### Experiment 2: Dimension Role Injection

- **Input**: Dimension role experiment results from 14 models in `reference_results/experiment2_dim_role/dim_role_result/`
- **Method**: Analyze performance differences between high trait role (target=4) vs low trait role (target=2)
- **Output**: High/low trait mean, standard deviation, and difference for each dimension

### Experiment 3: Mitigation Experiment (Self-reflection)

- **Input**:
  - Baseline: `reference_results/experiment1_baseline/results/` (Run 1 only)
  - Mitigation: `reference_results/experiment3_mitigation/reflection_results/` (Run 1 only)
- **Method**: Uses baseline BFI scores, compares behavioral test score changes between original vs. mitigation experiments
- **Output**: MSE improvement and improvement percentage for each model

## Data Description

### Human Calibration Data

`data/data_with_gpt54_scores.json` contains data from 94 participants:
- BFI-2 self-report questionnaire responses (60 items)
- GPT-5.4 scores on behavioral test questions (11 significant facets)
- Participant IDs: P001 ~ P094 (anonymized, email, phone, and other personal information removed)

This data is used to establish calibration mappings between model behavioral scores and human standards.

### 11 Significant Facets

Experiments tested 12 facet behavioral questions, but based on correlation analysis, only 11 facets showed significant correlations with BFI-2 questionnaire.
**Respectfulness was excluded due to non-significance**. The 11 significant facets used:

- **Extraversion**: sociability, assertiveness
- **Agreeableness**: compassion, trust
- **Conscientiousness**: organization, productiveness
- **Neuroticism**: anxiety, depression, emotional_volatility
- **Openness**: intellectual_curiosity, aesthetic_sensitivity

### V4 Interval Mapping Calibration Method

Performs percentile calibration on each facet's behavioral scores:
1. Calculate the target score's percentile interval [lower%, upper%] in the GPT score distribution
2. Take samples from the same percentile interval in the human distribution
3. Average these samples as the calibrated score

## Running Inference Experiments (Optional)

To re-run model inference experiments (requires API keys):

1. Configure API keys: Edit API configurations in `scripts/inference/*.py`
2. Run batch inference scripts:
   ```bash
   cd scripts/inference
   # Experiment 1
   ./run_batch_inference_v2.sh
   # Experiment 2
   ./run_dim_role_experiments.sh
   # Experiment 3
   ./run_batch_inference_reflection.sh
   ```

## Citation

If you use this code, please cite our paper:

```bibtex
[To be added]
```

---

# 中文版

# ActTraitBench：大语言模型人格一致性基准

[English](#acttraitbench-a-benchmark-for-llm-personality-consistency) | 中文版

本仓库提供了 ActTraitBench 框架的完整评估数据集、论文中三大核心实验的复现代码，以及基准参考结果。

## 目录结构

```
ActTraitBench/
├── data/                          # 只读数据文件
│   ├── behavior_questions.json    # 行为测试题库（12个子维度）
│   ├── bfi2_questions.json        # BFI-2自评问卷题库（60题）
│   ├── dim_role_prompt.json       # 维度角色注入提示词
│   ├── gpt_calibration_params.json # GPT评分校准参数
│   └── data_with_gpt54_scores.json # 人类校准数据（94名被试，已脱敏）
│
├── reference_results/             # 论文报告的官方实验结果（只读）
│   ├── experiment1_baseline/      # 实验1：主实验（基线）
│   │   └── results/               # 14个模型 × 3次运行的原始结果
│   ├── experiment2_dim_role/      # 实验2：维度角色注入实验
│   │   └── dim_role_result/       # 14个模型的高低角色实验结果
│   └── experiment3_mitigation/    # 实验3：缓解实验
│       └── reflection_results/    # 14个模型 × 3次运行的缓解实验结果
│
├── outputs/                       # 分析输出目录（包含示例输出）
│   ├── experiment1_baseline/      # 实验1分析结果
│   ├── experiment2_dim_role/      # 实验2分析结果
│   └── experiment3_mitigation/    # 实验3分析结果
│
└── scripts/                       # 分析和推理脚本
    ├── batch_calculate_scores_v2.py      # 基础计算模块
    ├── batch_calculate_scores_v4_multi.py # V4校准（11个显著子维度，支持多次运行）
    ├── analyze_dim_role_results_v2.py    # 维度角色实验分析
    ├── compare_reflection_corrected_v4.py # 缓解实验对比分析（V4版本）
    ├── experiment1_baseline.py           # 实验1运行脚本
    ├── experiment2_dim_role.py           # 实验2运行脚本
    ├── experiment3_mitigation_multi.py   # 实验3运行脚本（3次运行）
    └── inference/                        # 模型推理脚本
        ├── qianfan_models.inference.py            # 实验1推理脚本
        ├── qianfan_models.inference_reflection.py # 实验3推理脚本
        ├── dim_role_inference.py                  # 实验2推理脚本
        ├── run_batch_inference_v2.sh              # 批量运行实验1
        ├── run_batch_inference_reflection.sh      # 批量运行实验3
        └── run_dim_role_experiments.sh            # 批量运行实验2
```

## 环境要求

**Python版本**: >= 3.8

安装依赖：
```bash
pip install -r requirements.txt
```

核心依赖：
- pandas >= 1.3.0
- numpy >= 1.21.0
- openpyxl >= 3.0.9
- openai >= 1.0.0 (如需运行推理脚本)

## 快速开始

### 实验1：主实验（基线）

计算14个模型的BFI-2自评分数和行为测试分数，使用11个显著子维度的百分位校准（区间映射法）。

```bash
cd scripts
python experiment1_baseline.py
```

**输出文件**：
- `outputs/experiment1_baseline/scores_summary_with_std_v4.csv` - 汇总表格（含均值和标准差）

**输出内容**：
- BFI大五维度分数（均值±标准差）
- 行为测试大五维度分数（均值±标准差）
- 知行差距（MSE、RMSE）

### 实验2：维度角色注入实验

分析不同模型在高低角色注入条件下的人格表现差异。

```bash
cd scripts
python experiment2_dim_role.py
```

**输出文件**：
- `outputs/experiment2_dim_role/dim_role_summary_full_v2.csv` - 高低特质分析结果
- `outputs/experiment2_dim_role/dim_role_summary_full_v2.xlsx` - Excel格式

**输出内容**：
- 每个维度的高特质均值（目标=4）、低特质均值（目标=2）
- 高低特质差值和统计显著性

### 实验3：缓解实验（Self-reflection）

对比原实验与加入自我反思提示后的知行差距变化。

```bash
cd scripts
python experiment3_mitigation_multi.py
```

**输出文件**：
- `outputs/experiment3_mitigation/comparison_reflection_corrected_v4.csv` - 缓解效果对比
- `outputs/experiment3_mitigation/comparison_reflection_corrected_v4.xlsx` - Excel格式
- `outputs/experiment3_mitigation/scores_summary_with_std_v4.csv` - 缓解实验得分汇总

**输出内容**：
- 原实验 vs 缓解实验的MSE对比
- MSE改进量和改进百分比
- 每个维度的知行差距变化

## 实验说明

### 实验1：主实验（基线）

- **输入**：`reference_results/experiment1_baseline/results/` 中14个模型 × 3次运行的原始结果
- **方法**：V4百分位校准（11个显著子维度的区间映射法），计算3次运行的 mean(MSE₁, MSE₂, MSE₃)
- **输出**：每个模型的BFI分数、行为分数、MSE和RMSE（均值±标准差）

### 实验2：维度角色注入实验

- **输入**：`reference_results/experiment2_dim_role/dim_role_result/` 中14个模型的维度角色实验结果
- **方法**：分析高特质角色（目标=4）vs 低特质角色（目标=2）的表现差异
- **输出**：每个维度的高低特质均值、标准差和差值

### 实验3：缓解实验（Self-reflection）

- **输入**：
  - 基线：`reference_results/experiment1_baseline/results/`（仅使用 Run 1）
  - 缓解：`reference_results/experiment3_mitigation/reflection_results/`（仅使用 Run 1）
- **方法**：使用原实验的BFI分数，对比原实验 vs 缓解实验的行为测试分数变化
- **输出**：每个模型的MSE改进、改进百分比

## 数据说明

### 人类校准数据

`data/data_with_gpt54_scores.json` 包含94名被试的：
- BFI-2自评问卷答案（60题）
- 行为测试题的GPT-5.4评分（11个显著子维度）
- 被试编号：P001 ~ P094（已脱敏，删除邮箱、手机号等个人信息）

该数据用于建立模型行为分数与人类标准的校准映射。

### 11个显著子维度

实验中测试了12个子维度的行为题，但根据相关性分析，只有11个子维度与BFI-2问卷显著相关，
**respectfulness（谦恭）因不显著被排除**。最终使用的11个子维度：

- **外向性 (Extraversion)**: sociability (社交), assertiveness (果断)
- **宜人性 (Agreeableness)**: compassion (同情), trust (信任)
- **尽责性 (Conscientiousness)**: organization (条理), productiveness (效率)
- **神经质 (Neuroticism)**: anxiety (焦虑), depression (抑郁), emotional_volatility (易变)
- **开放性 (Openness)**: intellectual_curiosity (好奇), aesthetic_sensitivity (审美)

### V4区间映射校准法

对每个子维度的行为分数进行百分位校准：
1. 计算目标分数在GPT评分分布中的百分位区间 [lower%, upper%]
2. 在人类分布中取相同百分位区间的样本
3. 求这些样本的平均值作为校准后的分数

## 运行推理实验（可选）

如需重新运行模型推理实验（需要API密钥）：

1. 配置API密钥：编辑 `scripts/inference/*.py` 中的API配置
2. 运行批量推理脚本：
   ```bash
   cd scripts/inference
   # 实验1
   ./run_batch_inference_v2.sh
   # 实验2
   ./run_dim_role_experiments.sh
   # 实验3
   ./run_batch_inference_reflection.sh
   ```

## 引用

如果使用本代码，请引用我们的论文：

```bibtex
[待补充]
```

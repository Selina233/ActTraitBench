"""
对比缓解方法实验 vs 原始实验（V4区间映射法，Run 1版本）
使用原实验的BFI分数（两个实验应该相同）
只对比行为测试的变化
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def load_experiment_data(baseline_dir, reflection_dir, model_name, run_idx=1):
    """
    加载实验数据

    Args:
        baseline_dir: 原实验目录
        reflection_dir: 缓解实验目录
        model_name: 模型名称
        run_idx: 实验轮次

    Returns:
        tuple: (baseline_data, reflection_data)
    """
    baseline_file = Path(baseline_dir) / model_name / f'results-{run_idx}.json'
    reflection_file = Path(reflection_dir) / model_name / f'results-{run_idx}.json'

    if not baseline_file.exists():
        return None, None
    if not reflection_file.exists():
        return None, None

    with open(baseline_file, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)
    with open(reflection_file, 'r', encoding='utf-8') as f:
        reflection_data = json.load(f)

    return baseline_data, reflection_data


def calculate_scores_mixed(baseline_data, reflection_data, calibration_params):
    """
    计算混合分数：BFI用原实验，行为用缓解实验

    Args:
        baseline_data: 原实验数据
        reflection_data: 缓解实验数据
        calibration_params: 校准参数

    Returns:
        dict: 包含BFI和GPT分数
    """
    from batch_calculate_scores_v4_multi import process_single_run

    # 从原实验获取BFI分数
    baseline_scores = process_single_run(baseline_data, calibration_params)

    # 从缓解实验获取行为分数
    reflection_scores = process_single_run(reflection_data, calibration_params)

    return {
        'bfi_big5': baseline_scores['bfi_big5'],  # 用原实验的BFI
        'gpt_big5_baseline': baseline_scores['gpt_big5_calibrated'],  # 原实验的行为
        'gpt_big5_reflection': reflection_scores['gpt_big5_calibrated']  # 缓解实验的行为
    }


def compare_experiments_corrected(baseline_dir, reflection_dir, output_file):
    """
    对比实验（V4区间映射法）：BFI用原实验，只对比行为变化

    Args:
        baseline_dir: 原实验目录
        reflection_dir: 缓解实验目录
        output_file: 输出文件路径
    """
    from batch_calculate_scores_v4_multi import build_facet_calibration_params, BIG_FIVE_NAMES

    print(f"\n{'='*80}")
    print("缓解方法效果评估（V4区间映射法，Run 1版本）")
    print("BFI分数：使用原实验结果（避免temperature=0.3的随机性）")
    print("行为分数：对比原实验 vs 缓解实验")
    print(f"{'='*80}\n")

    # 建立校准参数
    script_dir = Path(__file__).parent
    data_file = script_dir.parent / 'data' / 'data_with_gpt54_scores.json'
    calibration_params = build_facet_calibration_params(str(data_file))

    # 查找所有模型
    baseline_path = Path(baseline_dir)
    model_dirs = [d.name for d in baseline_path.iterdir() if d.is_dir()]

    print(f"找到 {len(model_dirs)} 个模型\n")

    # 收集结果
    all_results = []

    for model_name in sorted(model_dirs):
        print(f"处理: {model_name}")

        baseline_data, reflection_data = load_experiment_data(
            baseline_dir, reflection_dir, model_name, run_idx=1
        )

        if baseline_data is None or reflection_data is None:
            print(f"  ⚠️  缺少数据文件\n")
            continue

        # 计算分数
        scores = calculate_scores_mixed(baseline_data, reflection_data, calibration_params)

        result = {'模型名称': model_name}

        # 大五维度
        dimensions = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
        dimension_gaps_baseline = []
        dimension_gaps_reflection = []

        for dim_key in dimensions:
            dim_cn = BIG_FIVE_NAMES[dim_key]

            # BFI分数（用原实验）
            bfi_score = scores['bfi_big5'].get(dim_key)

            # 原实验行为分数
            gpt_baseline = scores['gpt_big5_baseline'].get(dim_key)

            # 缓解实验行为分数
            gpt_reflection = scores['gpt_big5_reflection'].get(dim_key)

            # 计算知行差距
            if bfi_score is not None and gpt_baseline is not None:
                gap_baseline = abs(bfi_score - gpt_baseline)
                dimension_gaps_baseline.append((bfi_score - gpt_baseline) ** 2)
            else:
                gap_baseline = None

            if bfi_score is not None and gpt_reflection is not None:
                gap_reflection = abs(bfi_score - gpt_reflection)
                dimension_gaps_reflection.append((bfi_score - gpt_reflection) ** 2)
            else:
                gap_reflection = None

            # 保存结果
            result[f'{dim_cn}_BFI'] = round(bfi_score, 2) if bfi_score is not None else None
            result[f'{dim_cn}_原实验行为'] = round(gpt_baseline, 2) if gpt_baseline is not None else None
            result[f'{dim_cn}_缓解实验行为'] = round(gpt_reflection, 2) if gpt_reflection is not None else None
            result[f'{dim_cn}_原实验差距'] = round(gap_baseline, 3) if gap_baseline is not None else None
            result[f'{dim_cn}_缓解实验差距'] = round(gap_reflection, 3) if gap_reflection is not None else None

            if gap_baseline is not None and gap_reflection is not None:
                improvement = gap_baseline - gap_reflection
                result[f'{dim_cn}_改进'] = round(improvement, 3)
            else:
                result[f'{dim_cn}_改进'] = None

        # 计算总体MSE
        if dimension_gaps_baseline and dimension_gaps_reflection:
            mse_baseline = np.mean(dimension_gaps_baseline)
            mse_reflection = np.mean(dimension_gaps_reflection)
            mse_improvement = mse_baseline - mse_reflection
            improvement_pct = (mse_improvement / mse_baseline * 100) if mse_baseline > 0 else 0

            result['原实验_MSE'] = round(mse_baseline, 4)
            result['缓解实验_MSE'] = round(mse_reflection, 4)
            result['MSE改进'] = round(mse_improvement, 4)
            result['改进百分比'] = round(improvement_pct, 2)

        all_results.append(result)
        print(f"  ✓ 完成\n")

    # 保存结果
    df = pd.DataFrame(all_results)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✓ 结果已保存: {output_path}")

    # 保存Excel
    excel_path = output_path.with_suffix('.xlsx')
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"✓ Excel已保存: {excel_path}\n")

    # 打印摘要
    print(f"{'='*80}")
    print("摘要统计（V4区间映射法，Run 1版本）")
    print(f"{'='*80}\n")

    df_valid = df.dropna(subset=['MSE改进'])
    df_sorted = df_valid.sort_values('MSE改进', ascending=False)

    for idx, row in df_sorted.iterrows():
        model_name = row['模型名称']
        mse_baseline = row['原实验_MSE']
        mse_reflection = row['缓解实验_MSE']
        improvement = row['MSE改进']
        improvement_pct = row['改进百分比']

        status = "✓ 有效" if improvement > 0 else ("✗ 恶化" if improvement < 0 else "— 无变化")

        print(f"{model_name}:")
        print(f"  原实验MSE: {mse_baseline:.4f}")
        print(f"  缓解MSE:   {mse_reflection:.4f}")
        print(f"  改进:      {improvement:+.4f} ({improvement_pct:+.2f}%) {status}")
        print()

    # 总体统计
    avg_baseline = df_valid['原实验_MSE'].mean()
    avg_reflection = df_valid['缓解实验_MSE'].mean()
    avg_improvement = df_valid['MSE改进'].mean()
    avg_improvement_pct = df_valid['改进百分比'].mean()

    improved_count = (df_valid['MSE改进'] > 0).sum()
    worsened_count = (df_valid['MSE改进'] < 0).sum()

    print(f"{'='*80}")
    print("总体统计:")
    print(f"  平均原实验MSE:  {avg_baseline:.4f}")
    print(f"  平均缓解MSE:    {avg_reflection:.4f}")
    print(f"  平均改进:       {avg_improvement:+.4f} ({avg_improvement_pct:+.2f}%)")
    print(f"\n  有效缓解: {improved_count} 个模型")
    print(f"  反而恶化: {worsened_count} 个模型")
    print(f"{'='*80}\n")

    return df


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("用法: python compare_reflection_corrected_v4.py <输出文件>")
        print("示例: python compare_reflection_corrected_v4.py ./comparison_v4.csv")
        sys.exit(1)

    output_file = sys.argv[1]
    script_dir = Path(__file__).parent
    baseline_dir = script_dir.parent / 'reference_results' / 'experiment1_baseline' / 'results'
    reflection_dir = script_dir.parent / 'reference_results' / 'experiment3_mitigation' / 'reflection_results'

    compare_experiments_corrected(str(baseline_dir), str(reflection_dir), output_file)

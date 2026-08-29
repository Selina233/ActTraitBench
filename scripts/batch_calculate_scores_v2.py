"""
批量计算模型实验的分数
V2: 在大五维度层面进行GPT校准（使用等百分位等值法）
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# 固定排除的子维度（没有行为测试题）
EXCLUDED_FACETS = {'energy_level', 'responsibility', 'creative_imagination'}

# 人类校准数据文件路径
DEFAULT_HUMAN_DATA_FILE = Path(__file__).parent.parent / 'personality-test数据分析' / 'cleaned_json' / 'data_with_gpt54_scores.json'

# BIG-5 维度与子维度的映射关系
BIG5_FACETS = {
    'extraversion': ['sociability', 'assertiveness', 'energy_level'],
    'agreeableness': ['compassion', 'respectfulness', 'trust'],
    'conscientiousness': ['organization', 'productiveness', 'responsibility'],
    'neuroticism': ['anxiety', 'depression', 'emotional_volatility'],
    'openness': ['intellectual_curiosity', 'aesthetic_sensitivity', 'creative_imagination']
}

# 子维度到 BFI-2 题目的映射（每个子维度对应4道题）
FACET_TO_BFI2_ITEMS = {
    'sociability': ['bfi1', 'bfi16', 'bfi31', 'bfi46'],
    'assertiveness': ['bfi6', 'bfi21', 'bfi36', 'bfi51'],
    'energy_level': ['bfi11', 'bfi26', 'bfi41', 'bfi56'],
    'compassion': ['bfi2', 'bfi17', 'bfi32', 'bfi47'],
    'respectfulness': ['bfi7', 'bfi22', 'bfi37', 'bfi52'],
    'trust': ['bfi12', 'bfi27', 'bfi42', 'bfi57'],
    'organization': ['bfi3', 'bfi18', 'bfi33', 'bfi48'],
    'productiveness': ['bfi8', 'bfi23', 'bfi38', 'bfi53'],
    'responsibility': ['bfi13', 'bfi28', 'bfi43', 'bfi58'],
    'anxiety': ['bfi4', 'bfi19', 'bfi34', 'bfi49'],
    'depression': ['bfi9', 'bfi24', 'bfi39', 'bfi54'],
    'emotional_volatility': ['bfi14', 'bfi29', 'bfi44', 'bfi59'],
    'intellectual_curiosity': ['bfi10', 'bfi25', 'bfi40', 'bfi55'],
    'aesthetic_sensitivity': ['bfi5', 'bfi20', 'bfi35', 'bfi50'],
    'creative_imagination': ['bfi15', 'bfi30', 'bfi45', 'bfi60']
}

# 反向计分题目列表
REVERSE_ITEMS = {
    'sociability': ['bfi16', 'bfi31'],
    'assertiveness': ['bfi36', 'bfi51'],
    'energy_level': ['bfi11', 'bfi26'],
    'compassion': ['bfi17', 'bfi47'],
    'respectfulness': ['bfi22', 'bfi37'],
    'trust': ['bfi12', 'bfi42'],
    'organization': ['bfi3', 'bfi48'],
    'productiveness': ['bfi8', 'bfi23'],
    'responsibility': ['bfi28', 'bfi58'],
    'anxiety': ['bfi4', 'bfi49'],
    'depression': ['bfi9', 'bfi24'],
    'emotional_volatility': ['bfi29', 'bfi44'],
    'intellectual_curiosity': ['bfi25', 'bfi55'],
    'aesthetic_sensitivity': ['bfi5', 'bfi50'],
    'creative_imagination': ['bfi30', 'bfi45']
}


def reverse_score(score):
    """反向计分：6 - 原始分数"""
    if score is None:
        return None
    return 6 - score


def calculate_facet_score_from_bfi2(bfi2_answers, facet_key):
    """从BFI-2答案计算某个子维度的分数"""
    item_ids = FACET_TO_BFI2_ITEMS.get(facet_key, [])
    reverse_items = REVERSE_ITEMS.get(facet_key, [])

    scores = []
    for item_id in item_ids:
        if item_id in bfi2_answers:
            score = bfi2_answers[item_id]
            if item_id in reverse_items:
                score = reverse_score(score)
            scores.append(score)

    if len(scores) >= 3:
        return round(np.mean(scores), 2)
    return None


def calculate_big5_score_from_facets(facet_scores, big5_key):
    """
    从子维度分数计算大五维度分数

    Args:
        facet_scores: 子维度分数字典
        big5_key: 大五维度名称

    Returns:
        tuple: (score, valid_count, total_count, missing_facets)
    """
    facets = BIG5_FACETS[big5_key]
    valid_scores = []
    missing_facets = []

    for facet in facets:
        score = facet_scores.get(facet)
        if score is not None:
            valid_scores.append(score)
        else:
            missing_facets.append(facet)

    if valid_scores:
        return round(np.mean(valid_scores), 2), len(valid_scores), len(facets), missing_facets
    return None, 0, len(facets), missing_facets


def build_big5_calibration_params(human_data_file):
    """
    从人类数据建立大五维度层面的GPT评分校准参数（使用等百分位等值法）

    Args:
        human_data_file: 人类数据文件路径

    Returns:
        dict: 校准参数，格式为 {big5_key: {bfi_scores, gpt_scores}}
              bfi_scores 和 gpt_scores 是排序后的分数数组，用于百分位映射
    """
    if not Path(human_data_file).exists():
        print(f"⚠️  未找到人类校准数据: {human_data_file}")
        return {}

    with open(human_data_file, 'r', encoding='utf-8') as f:
        human_data = json.load(f)

    print(f"✓ 加载人类校准数据: {len(human_data)} 条")

    # 收集每个大五维度的BFI和GPT分数
    big5_bfi_scores = {key: [] for key in BIG5_FACETS.keys()}
    big5_gpt_scores = {key: [] for key in BIG5_FACETS.keys()}

    for record in human_data:
        # 1. 计算BFI子维度分数
        bfi2_answers = record.get('bfi2Answers', {})
        bfi_facet_scores = {}
        for facet_key in FACET_TO_BFI2_ITEMS.keys():
            score = calculate_facet_score_from_bfi2(bfi2_answers, facet_key)
            if score is not None:
                bfi_facet_scores[facet_key] = score

        # 2. 从BFI子维度计算BFI大五维度
        for big5_key in BIG5_FACETS.keys():
            score, valid, total, missing = calculate_big5_score_from_facets(bfi_facet_scores, big5_key)
            if score is not None:
                big5_bfi_scores[big5_key].append(score)

        # 3. 获取GPT子维度分数（原始分数，不校准）
        gpt_scores = record.get('gptScores', {})
        gpt_facet_scores = {}
        for facet_key in FACET_TO_BFI2_ITEMS.keys():
            if facet_key in EXCLUDED_FACETS:
                continue
            if facet_key in gpt_scores:
                gpt_info = gpt_scores[facet_key]
                if isinstance(gpt_info, dict) and 'score' in gpt_info:
                    gpt_facet_scores[facet_key] = gpt_info['score']
                elif isinstance(gpt_info, (int, float)):
                    gpt_facet_scores[facet_key] = gpt_info

        # 4. 从GPT子维度计算GPT大五维度
        for big5_key in BIG5_FACETS.keys():
            score, valid, total, missing = calculate_big5_score_from_facets(gpt_facet_scores, big5_key)
            if score is not None:
                big5_gpt_scores[big5_key].append(score)

    # 5. 构建等百分位映射参数
    calibration_params = {}

    for big5_key in BIG5_FACETS.keys():
        bfi_vals = big5_bfi_scores[big5_key]
        gpt_vals = big5_gpt_scores[big5_key]

        if len(bfi_vals) >= 3 and len(gpt_vals) >= 3:
            # 排序分数数组（用于百分位查找）
            bfi_sorted = np.sort(bfi_vals)
            gpt_sorted = np.sort(gpt_vals)

            calibration_params[big5_key] = {
                'bfi_scores': bfi_sorted.tolist(),
                'gpt_scores': gpt_sorted.tolist(),
                'n': len(bfi_vals)
            }

            bfi_mean = np.mean(bfi_vals)
            gpt_mean = np.mean(gpt_vals)
            bfi_median = np.median(bfi_vals)
            gpt_median = np.median(gpt_vals)

            print(f"  {big5_key}: BFI均值={bfi_mean:.2f} 中位数={bfi_median:.2f}, "
                  f"GPT均值={gpt_mean:.2f} 中位数={gpt_median:.2f}, n={len(bfi_vals)}")

    print(f"✓ 大五维度校准映射建立完成（等百分位法）: {len(calibration_params)} 个维度")
    return calibration_params


def calibrate_big5_score(gpt_raw, big5_key, calibration_params):
    """
    使用等百分位等值法校准单个大五维度的GPT分数

    等百分位等值法原理：
    1. 找到原始GPT分数在GPT分布中的百分位
    2. 找到BFI分布中对应该百分位的分数值

    Args:
        gpt_raw: 原始GPT大五维度分数
        big5_key: 大五维度名称
        calibration_params: 校准参数（包含排序后的分数数组）

    Returns:
        float: 校准后的分数
    """
    if gpt_raw is None or big5_key not in calibration_params:
        return None

    params = calibration_params[big5_key]
    gpt_scores = np.array(params['gpt_scores'])
    bfi_scores = np.array(params['bfi_scores'])

    # 计算 gpt_raw 在 GPT 分布中的百分位
    # 使用线性插值计算精确的百分位
    percentile_rank = np.searchsorted(gpt_scores, gpt_raw, side='left')

    # 处理边界情况
    if percentile_rank == 0:
        # 低于所有观测值，使用最小值
        calibrated = bfi_scores[0]
    elif percentile_rank >= len(gpt_scores):
        # 高于所有观测值，使用最大值
        calibrated = bfi_scores[-1]
    else:
        # 在观测范围内，进行线性插值
        # 计算精确的百分位位置
        if gpt_scores[percentile_rank] == gpt_scores[percentile_rank - 1]:
            # GPT分数恰好等于某个观测值（或有重复值）
            percentile = (percentile_rank - 1) / (len(gpt_scores) - 1)
        else:
            # 在两个观测值之间进行线性插值
            lower_val = gpt_scores[percentile_rank - 1]
            upper_val = gpt_scores[percentile_rank]
            fraction = (gpt_raw - lower_val) / (upper_val - lower_val)
            percentile = (percentile_rank - 1 + fraction) / (len(gpt_scores) - 1)

        # 找到BFI分布中对应百分位的分数
        bfi_position = percentile * (len(bfi_scores) - 1)
        lower_idx = int(np.floor(bfi_position))
        upper_idx = min(lower_idx + 1, len(bfi_scores) - 1)

        if lower_idx == upper_idx:
            calibrated = bfi_scores[lower_idx]
        else:
            # 在BFI分数之间进行线性插值
            fraction = bfi_position - lower_idx
            calibrated = bfi_scores[lower_idx] * (1 - fraction) + bfi_scores[upper_idx] * fraction

    # 限制在1-5范围内（理论上应该已经在范围内，但保险起见）
    calibrated = max(1.0, min(5.0, calibrated))

    return round(calibrated, 2)


def process_single_result(results_file, calibration_params):
    """
    处理单个模型的结果文件

    Args:
        results_file: 结果JSON文件路径
        calibration_params: 大五维度校准参数

    Returns:
        dict: 包含BFI和校准后GPT的大五维度分数
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 1. 计算BFI子维度分数
    bfi2_answers = results.get('bfi2Answers', {})
    bfi_facet_scores = {}
    for facet_key in FACET_TO_BFI2_ITEMS.keys():
        score = calculate_facet_score_from_bfi2(bfi2_answers, facet_key)
        if score is not None:
            bfi_facet_scores[facet_key] = score

    # 2. 计算BFI大五维度分数
    bfi_big5_scores = {}
    for big5_key in BIG5_FACETS.keys():
        score, valid, total, missing = calculate_big5_score_from_facets(bfi_facet_scores, big5_key)
        bfi_big5_scores[big5_key] = score

    # 3. 获取GPT子维度分数（原始分数）
    gpt_scores = results.get('gptScores', {})
    gpt_facet_scores = {}
    for facet_key in FACET_TO_BFI2_ITEMS.keys():
        if facet_key in EXCLUDED_FACETS:
            continue
        if facet_key in gpt_scores:
            score = gpt_scores[facet_key]
            if isinstance(score, dict) and 'score' in score:
                score = score['score']
            gpt_facet_scores[facet_key] = score

    # 4. 计算GPT大五维度分数（未校准）
    gpt_big5_raw = {}
    for big5_key in BIG5_FACETS.keys():
        score, valid, total, missing = calculate_big5_score_from_facets(gpt_facet_scores, big5_key)
        gpt_big5_raw[big5_key] = score

    # 5. 校准GPT大五维度分数
    gpt_big5_calibrated = {}
    for big5_key in BIG5_FACETS.keys():
        raw_score = gpt_big5_raw[big5_key]
        calibrated = calibrate_big5_score(raw_score, big5_key, calibration_params)
        gpt_big5_calibrated[big5_key] = calibrated

    return {
        'model_name': results.get('model_name', 'unknown'),
        'bfi_big5': bfi_big5_scores,
        'gpt_big5_calibrated': gpt_big5_calibrated
    }


def calculate_all_scores(results_dir, human_data_file=None, output_dir=None):
    """
    计算所有模型的分数并生成汇总表格

    Args:
        results_dir: 包含所有模型results.json的目录
        human_data_file: 人类校准数据文件路径
        output_dir: 输出目录（默认与results_dir相同）
    """
    results_dir = Path(results_dir)

    if output_dir is None:
        output_dir = results_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"批量计算模型分数 (大五维度层面校准 - 等百分位法)")
    print(f"结果目录: {results_dir}")
    print(f"{'='*80}\n")

    # 建立大五维度层面的GPT校准映射
    if human_data_file is None:
        human_data_file = DEFAULT_HUMAN_DATA_FILE

    print("建立大五维度层面的GPT校准映射（等百分位等值法）...")
    calibration_params = build_big5_calibration_params(human_data_file)

    if not calibration_params:
        print("⚠️  无法建立校准映射")
        return

    # 查找所有results.json文件
    result_files = list(results_dir.glob('*/results.json'))

    if not result_files:
        print(f"⚠️  在 {results_dir} 中未找到任何 results.json 文件")
        return

    print(f"\n找到 {len(result_files)} 个模型结果文件")

    # 处理每个模型
    all_results = []
    for results_file in sorted(result_files):
        print(f"处理: {results_file.parent.name}")
        result = process_single_result(results_file, calibration_params)
        all_results.append(result)

    # 生成Excel汇总
    generate_summary_excel(all_results, output_dir)

    # 保存详细JSON
    output_json = output_dir / 'all_scores.json'
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"📁 详细JSON已保存: {output_json}")

    print(f"\n{'='*80}")
    print(f"✅ 完成！")
    print(f"{'='*80}\n")


def generate_summary_excel(all_results, output_dir):
    """
    生成汇总Excel表格

    Args:
        all_results: 所有模型结果列表
        output_dir: 输出目录
    """
    if not all_results:
        print("⚠️  没有结果可以生成表格")
        return

    rows = []

    # 大五维度的顺序
    big5_dimensions = ['extraversion', 'agreeableness', 'conscientiousness',
                       'neuroticism', 'openness']

    # 中文名称
    dimension_names = {
        'extraversion': '外向性',
        'agreeableness': '宜人性',
        'conscientiousness': '尽责性',
        'neuroticism': '神经质',
        'openness': '开放性'
    }

    for result in all_results:
        model_name = result['model_name']
        bfi_scores = result['bfi_big5']
        gpt_scores = result['gpt_big5_calibrated']

        row = {'模型名称': model_name}

        squared_errors = []

        for dim in big5_dimensions:
            dim_name = dimension_names[dim]

            bfi_score = bfi_scores.get(dim)
            gpt_score = gpt_scores.get(dim)

            row[f'BFI_{dim_name}'] = round(bfi_score, 2) if bfi_score is not None else None
            row[f'GPT_{dim_name}'] = round(gpt_score, 2) if gpt_score is not None else None

            if bfi_score is not None and gpt_score is not None:
                diff = bfi_score - gpt_score
                row[f'差值_{dim_name}'] = round(diff, 2)
                squared_errors.append(diff ** 2)
            else:
                row[f'差值_{dim_name}'] = None

        # 计算MSE和RMSE
        if squared_errors:
            mse = sum(squared_errors) / len(squared_errors)
            row['总MSE'] = round(mse, 4)
            row['总RMSE'] = round(np.sqrt(mse), 4)
        else:
            row['总MSE'] = None
            row['总RMSE'] = None

        rows.append(row)

    # 创建DataFrame
    df = pd.DataFrame(rows)

    # 确保列的顺序
    columns = ['模型名称']
    for dim in big5_dimensions:
        dim_name = dimension_names[dim]
        columns.extend([f'BFI_{dim_name}', f'GPT_{dim_name}', f'差值_{dim_name}'])
    columns.extend(['总MSE', '总RMSE'])

    df = df[columns]

    # 保存为Excel
    output_excel = output_dir / 'scores_summary.xlsx'

    try:
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='模型分数汇总')

            worksheet = writer.sheets['模型分数汇总']

            # 设置列宽
            worksheet.column_dimensions['A'].width = 25  # 模型名称列
            for col in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q']:
                worksheet.column_dimensions[col].width = 12

        print(f"📊 Excel汇总表格已保存: {output_excel}")
        print(f"   列结构: 模型名称 → BIG-5 BFI分数 → BIG-5 GPT分数(等百分位校准) → 差值 → 总MSE/RMSE")
        print(f"   共 {len(rows)} 行, {len(columns)} 列")
        print(f"   注: GPT分数使用等百分位等值法在大五维度层面进行校准")
        print(f"   注: 等百分位法适用于非正态分布，保持原始数据的相对排序")

    except ImportError:
        print("⚠️  需要安装 openpyxl: pip install openpyxl")

    # 同时保存CSV版本
    output_csv = output_dir / 'scores_summary.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"📊 CSV汇总表格已保存: {output_csv}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='批量计算模型分数（大五维度层面校准 - 等百分位法）')
    parser.add_argument(
        'results_dir',
        type=str,
        help='包含所有模型results.json的目录'
    )
    parser.add_argument(
        '--human-data',
        type=str,
        default=None,
        help='人类校准数据文件路径（默认使用预设路径）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出目录（默认与results_dir相同）'
    )

    args = parser.parse_args()

    calculate_all_scores(
        args.results_dir,
        human_data_file=args.human_data,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()

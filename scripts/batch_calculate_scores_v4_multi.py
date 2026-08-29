"""
批量计算模型实验的分数（支持3次重复实验）
V4: 使用11个显著子维度层面的百分位校准（区间映射法）

使用的11个显著子维度：
- 外向性 (Extraversion): sociability (社交), assertiveness (果断)
- 宜人性 (Agreeableness): compassion (同情), trust (信任)
- 尽责性 (Conscientiousness): organization (条理), productiveness (效率)
- 神经质 (Neuroticism): anxiety (焦虑), depression (抑郁), emotional_volatility (易变)
- 开放性 (Openness): intellectual_curiosity (好奇), aesthetic_sensitivity (审美)

注意：实验中原本测试了12个子维度，但 respectfulness (谦恭) 在人类基准数据分析中
未达到显著性标准，因此在计算最终分数时被排除。
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# 导入基础函数
from batch_calculate_scores_v2 import (
    FACET_TO_BFI2_ITEMS,
    REVERSE_ITEMS,
    BIG5_FACETS,
    calculate_facet_score_from_bfi2,
    calculate_big5_score_from_facets,
    reverse_score,
    DEFAULT_HUMAN_DATA_FILE
)

# 11个显著的子维度（排除：负责responsibility、谦恭respectfulness、活力energy_level、想象creative_imagination）
SIGNIFICANT_FACETS = {
    'sociability': {'name': '社交', 'big_five': 'extraversion'},
    'assertiveness': {'name': '果断', 'big_five': 'extraversion'},
    'compassion': {'name': '同情', 'big_five': 'agreeableness'},
    'trust': {'name': '信任', 'big_five': 'agreeableness'},
    'organization': {'name': '条理', 'big_five': 'conscientiousness'},
    'productiveness': {'name': '效率', 'big_five': 'conscientiousness'},
    'anxiety': {'name': '焦虑', 'big_five': 'neuroticism'},
    'depression': {'name': '抑郁', 'big_five': 'neuroticism'},
    'emotional_volatility': {'name': '易变', 'big_five': 'neuroticism'},
    'intellectual_curiosity': {'name': '好奇', 'big_five': 'openness'},
    'aesthetic_sensitivity': {'name': '审美', 'big_five': 'openness'},
}

# 添加反向映射：子维度 -> 大五维度
FACET_TO_BIG_FIVE = {facet: info['big_five'] for facet, info in SIGNIFICANT_FACETS.items()}

BIG_FIVE_NAMES = {
    'extraversion': '外向性',
    'agreeableness': '宜人性',
    'conscientiousness': '尽责性',
    'neuroticism': '神经质',
    'openness': '开放性'
}


def percentile_calibrate(judge_scores, human_scores, target_score):
    """
    百分位映射校准（区间映射法）

    正确逻辑：
    1. 计算目标分数在GPT分布中占据的百分位区间 [lower%, upper%]
       - lower: 所有 < target_score 的样本比例（区间下界）
       - upper: 所有 ≤ target_score 的样本比例（区间上界）
    2. 在人类分布中取相同百分位区间 [lower%, upper%] 的样本
    3. 求这些样本的平均值作为校准后的分数
    """
    judge_scores = np.array(judge_scores)
    human_scores = np.array(human_scores)

    # 计算目标分数在GPT分布中的百分位区间
    # 下界：严格小于目标分数的样本比例
    lower_percentile = np.sum(judge_scores < target_score) / len(judge_scores) * 100
    # 上界：小于等于目标分数的样本比例
    upper_percentile = np.sum(judge_scores <= target_score) / len(judge_scores) * 100

    # 在人类分布中找到对应百分位区间的阈值
    lower_threshold = np.percentile(human_scores, lower_percentile)
    upper_threshold = np.percentile(human_scores, upper_percentile)

    # 取人类分布中该区间的所有样本
    matched_samples = human_scores[(human_scores >= lower_threshold) & (human_scores <= upper_threshold)]

    if len(matched_samples) > 0:
        calibrated_score = np.mean(matched_samples)
    else:
        # 极端情况：使用区间中点
        calibrated_score = (lower_threshold + upper_threshold) / 2

    return calibrated_score


def build_facet_calibration_params(human_data_file):
    """
    从人类数据建立11个显著子维度的校准参数

    Args:
        human_data_file: 人类数据文件路径

    Returns:
        dict: 校准参数，格式为 {facet_key: {'gpt': [...], 'facet': [...], 'n': int}}
    """
    if not Path(human_data_file).exists():
        print(f"⚠️  未找到人类校准数据: {human_data_file}")
        return {}

    with open(human_data_file, 'r', encoding='utf-8') as f:
        human_data = json.load(f)

    print(f"✓ 加载人类校准数据: {len(human_data)} 条")

    # 收集每个子维度的GPT和BFI-2 facet分数
    distributions = {}

    for facet_key, facet_info in SIGNIFICANT_FACETS.items():
        gpt_scores = []
        facet_scores = []

        for record in human_data:
            # GPT分数
            gpt_score = record.get('gptScores', {}).get(facet_key, {}).get('score')
            # BFI-2 facet分数
            facet_score = record.get('facetScores', {}).get(facet_key, {}).get('score')

            if gpt_score is not None and facet_score is not None:
                try:
                    gpt_scores.append(float(gpt_score))
                    facet_scores.append(float(facet_score))
                except:
                    pass

        if len(gpt_scores) >= 2:
            distributions[facet_key] = {
                'name': facet_info['name'],
                'big_five': facet_info['big_five'],
                'gpt': np.array(gpt_scores),
                'facet': np.array(facet_scores),
                'n': len(gpt_scores)
            }

            print(f"  {facet_info['name']} ({facet_key}): n={len(gpt_scores)}")

    return distributions


def calibrate_facet_score(gpt_score, facet_key, calibration_params):
    """
    使用百分位映射校准单个子维度的GPT分数

    Args:
        gpt_score: GPT原始分数
        facet_key: 子维度key
        calibration_params: 校准参数

    Returns:
        float: 校准后的分数
    """
    if facet_key not in calibration_params:
        return gpt_score  # 无校准参数，返回原始分数

    if gpt_score is None:
        return None

    params = calibration_params[facet_key]
    calibrated = percentile_calibrate(
        params['gpt'],
        params['facet'],
        gpt_score
    )

    return calibrated


def calculate_model_scores(results, calibration_params):
    """计算模型分数（兼容compare_reflection_corrected.py）"""
    return process_single_run(results, calibration_params)


def process_single_run(results, calibration_params):
    """
    处理单次实验结果

    Args:
        results: 实验结果JSON
        calibration_params: 校准参数

    Returns:
        dict: BFI和GPT大五维度分数
    """
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

    # 3. 获取GPT子维度原始分数
    gpt_scores_raw = results.get('gptScores', {})
    gpt_facet_raw = {}
    for facet_key in SIGNIFICANT_FACETS.keys():
        if facet_key in gpt_scores_raw:
            score = gpt_scores_raw[facet_key]
            if isinstance(score, dict) and 'score' in score:
                score = score['score']
            gpt_facet_raw[facet_key] = score

    # 4. 校准GPT子维度分数（11个显著子维度）
    gpt_facet_calibrated = {}
    for facet_key in SIGNIFICANT_FACETS.keys():
        if facet_key in gpt_facet_raw:
            raw_score = gpt_facet_raw[facet_key]
            calibrated = calibrate_facet_score(raw_score, facet_key, calibration_params)
            gpt_facet_calibrated[facet_key] = calibrated

    # 5. 从校准后的11个子维度计算GPT大五维度分数
    gpt_big5_calibrated = {}
    for big5_key in BIG5_FACETS.keys():
        score, valid, total, missing = calculate_big5_score_from_facets(gpt_facet_calibrated, big5_key)
        gpt_big5_calibrated[big5_key] = score

    return {
        'bfi_big5': bfi_big5_scores,
        'gpt_big5_calibrated': gpt_big5_calibrated
    }


def process_multiple_runs(model_dir, calibration_params):
    """
    处理单个模型的3次实验结果

    Args:
        model_dir: 模型结果目录
        calibration_params: 校准参数

    Returns:
        dict: 包含3次实验结果、平均值和标准差
    """
    model_name = model_dir.name

    # 查找 results-1.json, results-2.json, results-3.json
    run_results = []
    for run_id in [1, 2, 3]:
        result_file = model_dir / f'results-{run_id}.json'
        if result_file.exists():
            with open(result_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                run_result = process_single_run(results, calibration_params)
                run_results.append(run_result)
        else:
            print(f"  ⚠️  缺少 {result_file.name}")

    if not run_results:
        print(f"  ❌ {model_name}: 未找到任何实验结果")
        return None

    # 计算平均值和标准差
    aggregated = aggregate_runs(run_results, model_name)
    return aggregated


def aggregate_runs(run_results, model_name):
    """
    汇总多次实验的结果，计算平均值和标准差

    Args:
        run_results: 实验结果列表
        model_name: 模型名称

    Returns:
        dict: 包含平均值和标准差
    """
    big5_dimensions = ['extraversion', 'agreeableness', 'conscientiousness',
                       'neuroticism', 'openness']

    # 收集每个维度的分数
    bfi_scores_by_dim = {dim: [] for dim in big5_dimensions}
    gpt_scores_by_dim = {dim: [] for dim in big5_dimensions}

    for run_result in run_results:
        for dim in big5_dimensions:
            bfi_score = run_result['bfi_big5'].get(dim)
            gpt_score = run_result['gpt_big5_calibrated'].get(dim)

            if bfi_score is not None:
                bfi_scores_by_dim[dim].append(bfi_score)
            if gpt_score is not None:
                gpt_scores_by_dim[dim].append(gpt_score)

    # 计算平均值和标准差
    bfi_mean = {}
    bfi_std = {}
    gpt_mean = {}
    gpt_std = {}

    for dim in big5_dimensions:
        if bfi_scores_by_dim[dim]:
            bfi_mean[dim] = np.mean(bfi_scores_by_dim[dim])
            bfi_std[dim] = np.std(bfi_scores_by_dim[dim], ddof=1) if len(bfi_scores_by_dim[dim]) > 1 else 0.0
        else:
            bfi_mean[dim] = None
            bfi_std[dim] = None

        if gpt_scores_by_dim[dim]:
            gpt_mean[dim] = np.mean(gpt_scores_by_dim[dim])
            gpt_std[dim] = np.std(gpt_scores_by_dim[dim], ddof=1) if len(gpt_scores_by_dim[dim]) > 1 else 0.0
        else:
            gpt_mean[dim] = None
            gpt_std[dim] = None

    # 计算每次实验的MSE，然后取平均
    mse_list = []
    for run_result in run_results:
        squared_errors = []
        for dim in big5_dimensions:
            bfi_score = run_result['bfi_big5'].get(dim)
            gpt_score = run_result['gpt_big5_calibrated'].get(dim)
            if bfi_score is not None and gpt_score is not None:
                squared_errors.append((bfi_score - gpt_score) ** 2)

        if squared_errors:
            run_mse = sum(squared_errors) / len(squared_errors)
            mse_list.append(run_mse)

    # MSE的平均值和标准差
    if mse_list:
        mse_mean = np.mean(mse_list)
        mse_std = np.std(mse_list, ddof=1) if len(mse_list) > 1 else 0.0
        rmse_mean = np.sqrt(mse_mean)
    else:
        mse_mean = None
        mse_std = None
        rmse_mean = None

    return {
        'model_name': model_name,
        '实验次数': len(run_results),
        'bfi_mean': bfi_mean,
        'bfi_std': bfi_std,
        'gpt_mean': gpt_mean,
        'gpt_std': gpt_std,
        'mse_mean': mse_mean,
        'mse_std': mse_std,
        'rmse_mean': rmse_mean,
        'individual_runs': run_results
    }


def calculate_human_baseline(human_data_file, calibration_params):
    """
    计算人类数据的基线分数（BFI和GPT校准后的平均值和标准差）

    Args:
        human_data_file: 人类数据文件路径
        calibration_params: 校准参数

    Returns:
        dict: 包含人类数据的统计信息
    """
    if not Path(human_data_file).exists():
        return None

    with open(human_data_file, 'r', encoding='utf-8') as f:
        human_data = json.load(f)

    # 收集每个维度的BFI和GPT分数
    big5_dimensions = ['extraversion', 'agreeableness', 'conscientiousness',
                       'neuroticism', 'openness']

    bfi_scores_by_dim = {dim: [] for dim in big5_dimensions}
    gpt_scores_by_dim = {dim: [] for dim in big5_dimensions}
    individual_mse_list = []

    for record in human_data:
        # 1. 计算BFI分数
        bfi2_answers = record.get('bfi2Answers', {})
        bfi_facet_scores = {}
        for facet_key in FACET_TO_BFI2_ITEMS.keys():
            score = calculate_facet_score_from_bfi2(bfi2_answers, facet_key)
            if score is not None:
                bfi_facet_scores[facet_key] = score

        person_bfi_big5 = {}
        for big5_key in BIG5_FACETS.keys():
            score, valid, total, missing = calculate_big5_score_from_facets(bfi_facet_scores, big5_key)
            if score is not None:
                bfi_scores_by_dim[big5_key].append(score)
                person_bfi_big5[big5_key] = score

        # 2. 获取GPT原始子维度分数
        gpt_scores_raw = record.get('gptScores', {})
        gpt_facet_raw = {}
        for facet_key in SIGNIFICANT_FACETS.keys():
            if facet_key in gpt_scores_raw:
                gpt_info = gpt_scores_raw[facet_key]
                if isinstance(gpt_info, dict) and 'score' in gpt_info:
                    gpt_facet_raw[facet_key] = gpt_info['score']
                elif isinstance(gpt_info, (int, float)):
                    gpt_facet_raw[facet_key] = gpt_info

        # 3. 校准GPT子维度分数
        gpt_facet_calibrated = {}
        for facet_key in SIGNIFICANT_FACETS.keys():
            if facet_key in gpt_facet_raw:
                raw_score = gpt_facet_raw[facet_key]
                calibrated = calibrate_facet_score(raw_score, facet_key, calibration_params)
                gpt_facet_calibrated[facet_key] = calibrated

        # 4. 从校准后的子维度计算GPT大五维度
        person_gpt_big5 = {}
        for big5_key in BIG5_FACETS.keys():
            score, valid, total, missing = calculate_big5_score_from_facets(gpt_facet_calibrated, big5_key)
            if score is not None:
                gpt_scores_by_dim[big5_key].append(score)
                person_gpt_big5[big5_key] = score

        # 5. 计算这个人的MSE
        person_squared_errors = []
        for dim in big5_dimensions:
            if dim in person_bfi_big5 and dim in person_gpt_big5:
                bfi_score = person_bfi_big5[dim]
                gpt_score = person_gpt_big5[dim]
                person_squared_errors.append((bfi_score - gpt_score) ** 2)

        if person_squared_errors:
            person_mse = sum(person_squared_errors) / len(person_squared_errors)
            individual_mse_list.append(person_mse)

    # 计算平均值和标准差
    bfi_mean = {}
    bfi_std = {}
    gpt_mean = {}
    gpt_std = {}

    for dim in big5_dimensions:
        if bfi_scores_by_dim[dim]:
            bfi_mean[dim] = np.mean(bfi_scores_by_dim[dim])
            bfi_std[dim] = np.std(bfi_scores_by_dim[dim], ddof=1)
        else:
            bfi_mean[dim] = None
            bfi_std[dim] = None

        if gpt_scores_by_dim[dim]:
            gpt_mean[dim] = np.mean(gpt_scores_by_dim[dim])
            gpt_std[dim] = np.std(gpt_scores_by_dim[dim], ddof=1)
        else:
            gpt_mean[dim] = None
            gpt_std[dim] = None

    # 计算MSE的平均值和标准差
    if individual_mse_list:
        mse_mean = np.mean(individual_mse_list)
        mse_std = np.std(individual_mse_list, ddof=1)
        rmse_mean = np.sqrt(mse_mean)
    else:
        mse_mean = None
        mse_std = None
        rmse_mean = None

    return {
        'model_name': '人类基线',
        '实验次数': len(human_data),
        'bfi_mean': bfi_mean,
        'bfi_std': bfi_std,
        'gpt_mean': gpt_mean,
        'gpt_std': gpt_std,
        'mse_mean': mse_mean,
        'mse_std': mse_std,
        'rmse_mean': rmse_mean
    }


def calculate_all_scores(results_dir, human_data_file=None, output_dir=None):
    """
    计算所有模型的分数并生成汇总表格（支持3次实验）

    Args:
        results_dir: 包含所有模型目录的根目录
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
    print(f"批量计算模型分数（V4: 11个显著子维度百分位校准，区间映射法）")
    print(f"结果目录: {results_dir}")
    print(f"{'='*80}\n")

    # 建立11个显著子维度的校准参数
    if human_data_file is None:
        human_data_file = DEFAULT_HUMAN_DATA_FILE

    print("建立11个显著子维度的校准参数（区间映射法）...")
    calibration_params = build_facet_calibration_params(human_data_file)

    if not calibration_params:
        print("⚠️  无法建立校准映射")
        return

    # 查找所有模型目录（包含 results-*.json 的目录）
    model_dirs = []
    for item in results_dir.iterdir():
        if item.is_dir():
            # 检查是否包含 results-1.json 或 results.json
            has_new_format = any((item / f'results-{i}.json').exists() for i in [1, 2, 3])
            has_old_format = (item / 'results.json').exists()

            if has_new_format:
                model_dirs.append(item)
            elif has_old_format:
                print(f"  ℹ️  {item.name} 使用旧格式 (results.json)，跳过")

    if not model_dirs:
        print(f"⚠️  在 {results_dir} 中未找到任何新格式的模型结果")
        print(f"   （需要 results-1.json, results-2.json, results-3.json）")
        return

    print(f"\n找到 {len(model_dirs)} 个模型目录")

    # 处理人类基线数据
    all_results = []
    print("\n处理: 人类基线")
    human_baseline = calculate_human_baseline(human_data_file, calibration_params)
    if human_baseline:
        all_results.append(human_baseline)
        print(f"  ✓ 人类数据: {human_baseline['实验次数']} 条")

    # 处理每个模型
    for model_dir in sorted(model_dirs):
        print(f"处理: {model_dir.name}")
        result = process_multiple_runs(model_dir, calibration_params)
        if result:
            all_results.append(result)

    # 生成Excel汇总
    generate_summary_excel(all_results, output_dir)

    # 保存详细JSON
    output_json = output_dir / 'all_scores_with_std_v4.json'
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"📁 详细JSON已保存: {output_json}")

    print(f"\n{'='*80}")
    print(f"✅ 完成！")
    print(f"{'='*80}\n")


def generate_summary_excel(all_results, output_dir):
    """
    生成汇总Excel表格（包含平均值和标准差）

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
        n_runs = result['实验次数']
        bfi_mean = result['bfi_mean']
        bfi_std = result['bfi_std']
        gpt_mean = result['gpt_mean']
        gpt_std = result['gpt_std']

        row = {
            '模型名称': model_name,
            '实验次数': n_runs
        }

        for dim in big5_dimensions:
            dim_name = dimension_names[dim]

            # BFI平均值和标准差
            bfi_m = bfi_mean.get(dim)
            bfi_s = bfi_std.get(dim)
            row[f'BFI_{dim_name}_均值'] = round(bfi_m, 2) if bfi_m is not None else None
            row[f'BFI_{dim_name}_标准差'] = round(bfi_s, 3) if bfi_s is not None else None

            # GPT平均值和标准差
            gpt_m = gpt_mean.get(dim)
            gpt_s = gpt_std.get(dim)
            row[f'行为_{dim_name}_均值'] = round(gpt_m, 2) if gpt_m is not None else None
            row[f'行为_{dim_name}_标准差'] = round(gpt_s, 3) if gpt_s is not None else None

            # 差值（平均值）
            if bfi_m is not None and gpt_m is not None:
                diff = bfi_m - gpt_m
                row[f'差值_{dim_name}'] = round(diff, 2)
            else:
                row[f'差值_{dim_name}'] = None

        # 使用预计算的MSE（每次实验先算MSE再平均）
        mse_mean = result.get('mse_mean')
        mse_std = result.get('mse_std')
        rmse_mean = result.get('rmse_mean')

        if mse_mean is not None:
            row['总MSE'] = round(mse_mean, 4)
            row['总MSE_标准差'] = round(mse_std, 4) if mse_std is not None else None
            row['总RMSE'] = round(rmse_mean, 4)
        else:
            row['总MSE'] = None
            row['总MSE_标准差'] = None
            row['总RMSE'] = None

        rows.append(row)

    # 创建DataFrame
    df = pd.DataFrame(rows)

    # 确保列的顺序
    columns = ['模型名称', '实验次数']
    for dim in big5_dimensions:
        dim_name = dimension_names[dim]
        columns.extend([
            f'BFI_{dim_name}_均值',
            f'BFI_{dim_name}_标准差',
            f'行为_{dim_name}_均值',
            f'行为_{dim_name}_标准差',
            f'差值_{dim_name}'
        ])
    columns.extend(['总MSE', '总MSE_标准差', '总RMSE'])

    df = df[columns]

    # 保存为Excel
    output_excel = output_dir / 'scores_summary_with_std_v4.xlsx'

    try:
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='模型分数汇总')

            worksheet = writer.sheets['模型分数汇总']

            # 设置列宽
            worksheet.column_dimensions['A'].width = 25  # 模型名称列
            worksheet.column_dimensions['B'].width = 12  # 实验次数列
            for col_idx in range(3, len(columns) + 1):
                col_letter = chr(64 + col_idx) if col_idx <= 26 else f'A{chr(64 + col_idx - 26)}'
                worksheet.column_dimensions[col_letter].width = 12

        print(f"📊 Excel汇总表格已保存: {output_excel}")
        print(f"   列结构: 模型名称 → 实验次数 → BIG-5 (均值±标准差) → 差值 → MSE/RMSE")
        print(f"   共 {len(rows)} 行, {len(columns)} 列")
        print(f"   校准方法: 11个显著子维度层面的百分位校准（区间映射法）")
        if all_results:
            max_runs = max([r['实验次数'] for r in all_results])
            print(f"   标准差基于{max_runs}次重复实验计算")

    except ImportError:
        print("⚠️  需要安装 openpyxl: pip install openpyxl")

    # 同时保存CSV版本
    output_csv = output_dir / 'scores_summary_with_std_v4.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"📊 CSV汇总表格已保存: {output_csv}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='批量计算模型分数（V4: 11个显著子维度百分位校准）')
    parser.add_argument(
        'results_dir',
        type=str,
        help='包含所有模型目录的根目录'
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

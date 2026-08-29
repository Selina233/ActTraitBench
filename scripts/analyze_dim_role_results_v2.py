"""
分析维度角色注入实验结果
V2: 使用11个显著子维度层面的百分位校准（区间映射法）
对比高特质角色 vs 低特质角色的表现
包括：
1. BFI-2自评问卷结果
2. 行为测试题GPT评分结果（校准后）
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from batch_calculate_scores_v4 import (
    SIGNIFICANT_FACETS,
    BIG_FIVE_NAMES,
    BIG5_FACETS,
    FACET_TO_BFI2_ITEMS,
    calculate_facet_score_from_bfi2,
    calculate_big5_score_from_facets,
    build_facet_calibration_params,
    calibrate_facet_score,
    DEFAULT_HUMAN_DATA_FILE
)


def analyze_dimension_role_results(result_dir, human_data_file=None, output_dir=None):
    """
    分析维度角色注入实验的结果

    Args:
        result_dir: 结果目录
        human_data_file: 人类校准数据文件
        output_dir: 输出目录（默认与result_dir相同）
    """
    result_dir = Path(result_dir)

    if output_dir is None:
        output_dir = result_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"维度角色注入实验 - 结果分析（V2: 11个显著子维度百分位校准）")
    print(f"结果目录: {result_dir}")
    print(f"{'='*80}\n")

    # 建立校准参数
    if human_data_file is None:
        human_data_file = DEFAULT_HUMAN_DATA_FILE

    print("建立11个显著子维度的校准参数...")
    calibration_params = build_facet_calibration_params(human_data_file)

    if not calibration_params:
        print("⚠️  无法建立校准映射，将使用原始分数")
        calibration_params = {}

    # 查找所有结果文件
    result_files = sorted(result_dir.glob('*_dim_role_results.json'))

    if not result_files:
        print(f"❌ 在 {result_dir} 中未找到任何结果文件")
        return

    print(f"\n找到 {len(result_files)} 个模型结果\n")

    # 收集所有模型的数据
    all_model_data = []
    dimension_names = {
        'Extraversion': '外向性',
        'Agreeableness': '宜人性',
        'Conscientiousness': '尽责性',
        'Neuroticism': '神经质',
        'Openness': '开放性'
    }

    for result_file in result_files:
        print(f"处理: {result_file.name}")

        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        model_name = data['model_name']

        # 1. 分析BFI-2结果
        bfi2_results = data.get('bfi2_results', {})
        bfi2_analysis = analyze_bfi2_results(bfi2_results, model_name)

        # 2. 分析行为测试结果（使用校准）
        behavior_results = data.get('behavior_results', {})
        behavior_analysis = analyze_behavior_results(behavior_results, model_name, calibration_params)

        # 3. 合并分析结果
        model_summary = {
            '模型名称': model_name,
            **bfi2_analysis,
            **behavior_analysis
        }

        all_model_data.append(model_summary)

        print(f"  ✓ BFI-2分析完成")
        print(f"  ✓ 行为测试分析完成（已校准）")

    # 生成汇总表格
    print(f"\n生成汇总表格...")
    generate_summary_tables(all_model_data, output_dir, dimension_names)

    print(f"\n{'='*80}")
    print(f"✅ 分析完成！")
    print(f"{'='*80}\n")


def analyze_bfi2_results(bfi2_results, model_name):
    """
    分析BFI-2测试结果，计算大五维度分数

    Args:
        bfi2_results: BFI-2结果字典
        model_name: 模型名称

    Returns:
        dict: 包含高/低特质的BFI大五维度分数
    """
    # 分别收集高特质和低特质的答案
    n_runs = 3
    high_trait_runs = []
    low_trait_runs = []

    for run_id in range(1, n_runs + 1):
        high_answers = {}
        low_answers = {}

        for q_key, q_data in bfi2_results.items():
            # 高特质答案
            high_run_data = [r for r in q_data.get('high_trait_runs', []) if r['run_id'] == run_id]
            if high_run_data and 'answer' in high_run_data[0]:
                try:
                    answer = high_run_data[0]['answer']
                    high_answers[q_key] = int(answer) if isinstance(answer, (int, str)) else int(str(answer).split()[0])
                except (ValueError, TypeError):
                    pass

            # 低特质答案
            low_run_data = [r for r in q_data.get('low_trait_runs', []) if r['run_id'] == run_id]
            if low_run_data and 'answer' in low_run_data[0]:
                try:
                    answer = low_run_data[0]['answer']
                    low_answers[q_key] = int(answer) if isinstance(answer, (int, str)) else int(str(answer).split()[0])
                except (ValueError, TypeError):
                    pass

        if high_answers:
            high_trait_runs.append(high_answers)
        if low_answers:
            low_trait_runs.append(low_answers)

    # 计算大五维度分数（高特质）
    high_bfi_big5_list = []
    for answers in high_trait_runs:
        facet_scores = {}
        for facet_key in FACET_TO_BFI2_ITEMS.keys():
            score = calculate_facet_score_from_bfi2(answers, facet_key)
            if score is not None:
                facet_scores[facet_key] = score

        big5_scores = {}
        for big5_key in BIG5_FACETS.keys():
            score, _, _, _ = calculate_big5_score_from_facets(facet_scores, big5_key)
            big5_scores[big5_key] = score

        high_bfi_big5_list.append(big5_scores)

    # 计算大五维度分数（低特质）
    low_bfi_big5_list = []
    for answers in low_trait_runs:
        facet_scores = {}
        for facet_key in FACET_TO_BFI2_ITEMS.keys():
            score = calculate_facet_score_from_bfi2(answers, facet_key)
            if score is not None:
                facet_scores[facet_key] = score

        big5_scores = {}
        for big5_key in BIG5_FACETS.keys():
            score, _, _, _ = calculate_big5_score_from_facets(facet_scores, big5_key)
            big5_scores[big5_key] = score

        low_bfi_big5_list.append(big5_scores)

    # 计算均值和标准差
    result = {}
    big5_dimensions = ['extraversion', 'agreeableness', 'conscientiousness',
                       'neuroticism', 'openness']

    dimension_names_cn = {
        'extraversion': '外向性',
        'agreeableness': '宜人性',
        'conscientiousness': '尽责性',
        'neuroticism': '神经质',
        'openness': '开放性'
    }

    for dim in big5_dimensions:
        dim_name = dimension_names_cn[dim]

        # 高特质（目标=4）
        high_scores = [run[dim] for run in high_bfi_big5_list if run.get(dim) is not None]
        if high_scores:
            high_mean = np.mean(high_scores)
            result[f'BFI_{dim_name}_高特质均值'] = round(high_mean, 2)
            result[f'BFI_{dim_name}_高特质标准差'] = round(np.std(high_scores, ddof=1) if len(high_scores) > 1 else 0.0, 3)
            result[f'BFI_{dim_name}_高特质偏差'] = round(high_mean - 4.0, 2)  # 与目标4的差值
        else:
            result[f'BFI_{dim_name}_高特质均值'] = None
            result[f'BFI_{dim_name}_高特质标准差'] = None
            result[f'BFI_{dim_name}_高特质偏差'] = None

        # 低特质（目标=2）
        low_scores = [run[dim] for run in low_bfi_big5_list if run.get(dim) is not None]
        if low_scores:
            low_mean = np.mean(low_scores)
            result[f'BFI_{dim_name}_低特质均值'] = round(low_mean, 2)
            result[f'BFI_{dim_name}_低特质标准差'] = round(np.std(low_scores, ddof=1) if len(low_scores) > 1 else 0.0, 3)
            result[f'BFI_{dim_name}_低特质偏差'] = round(low_mean - 2.0, 2)  # 与目标2的差值
        else:
            result[f'BFI_{dim_name}_低特质均值'] = None
            result[f'BFI_{dim_name}_低特质标准差'] = None
            result[f'BFI_{dim_name}_低特质偏差'] = None

        # 差值（高-低）
        if high_scores and low_scores:
            result[f'BFI_{dim_name}_差值'] = round(np.mean(high_scores) - np.mean(low_scores), 2)
        else:
            result[f'BFI_{dim_name}_差值'] = None

    return result


def analyze_behavior_results(behavior_results, model_name, calibration_params):
    """
    分析行为测试结果（GPT评分），使用11个显著子维度的百分位校准

    Args:
        behavior_results: 行为测试结果字典
        model_name: 模型名称
        calibration_params: 校准参数

    Returns:
        dict: 包含高/低特质的行为测试GPT评分（校准后）
    """
    # 按子维度收集数据
    facets_data = {}

    # 初始化
    for facet_key in SIGNIFICANT_FACETS.keys():
        facets_data[facet_key] = {
            'high_scores': [],
            'low_scores': []
        }

    # 遍历所有行为测试题
    for q_key, q_data in behavior_results.items():
        facet = q_data.get('facet')  # 子维度

        if facet not in SIGNIFICANT_FACETS:
            continue  # 跳过不显著的子维度

        # 收集高特质原始分数
        for score_data in q_data.get('high_trait_gpt_scores', []):
            if 'gpt_score' in score_data:
                raw_score = score_data['gpt_score']
                # 校准分数
                calibrated = calibrate_facet_score(raw_score, facet, calibration_params)
                if calibrated is not None:
                    facets_data[facet]['high_scores'].append(calibrated)

        # 收集低特质原始分数
        for score_data in q_data.get('low_trait_gpt_scores', []):
            if 'gpt_score' in score_data:
                raw_score = score_data['gpt_score']
                # 校准分数
                calibrated = calibrate_facet_score(raw_score, facet, calibration_params)
                if calibrated is not None:
                    facets_data[facet]['low_scores'].append(calibrated)

    # 从11个校准后的子维度计算大五维度分数
    result = {}
    dimension_names_cn = {
        'extraversion': '外向性',
        'agreeableness': '宜人性',
        'conscientiousness': '尽责性',
        'neuroticism': '神经质',
        'openness': '开放性'
    }

    for big5_key, dim_cn in dimension_names_cn.items():
        # 找到属于该大五维度的子维度
        facet_keys = [k for k, v in SIGNIFICANT_FACETS.items() if v['big_five'] == big5_key]

        # 高特质：收集所有子维度的平均分
        high_facet_scores = {}
        for facet_key in facet_keys:
            if facets_data[facet_key]['high_scores']:
                high_facet_scores[facet_key] = np.mean(facets_data[facet_key]['high_scores'])

        # 低特质：收集所有子维度的平均分
        low_facet_scores = {}
        for facet_key in facet_keys:
            if facets_data[facet_key]['low_scores']:
                low_facet_scores[facet_key] = np.mean(facets_data[facet_key]['low_scores'])

        # 计算大五维度分数（从子维度平均）
        if high_facet_scores:
            high_big5_score = np.mean(list(high_facet_scores.values()))
            result[f'行为_{dim_cn}_高特质均值'] = round(high_big5_score, 2)
            result[f'行为_{dim_cn}_高特质偏差'] = round(high_big5_score - 4.0, 2)  # 与目标4的差值

            # 标准差：各个子维度的分数的标准差
            if len(high_facet_scores) > 1:
                result[f'行为_{dim_cn}_高特质标准差'] = round(np.std(list(high_facet_scores.values()), ddof=1), 3)
            else:
                result[f'行为_{dim_cn}_高特质标准差'] = 0.0
        else:
            result[f'行为_{dim_cn}_高特质均值'] = None
            result[f'行为_{dim_cn}_高特质标准差'] = None
            result[f'行为_{dim_cn}_高特质偏差'] = None

        if low_facet_scores:
            low_big5_score = np.mean(list(low_facet_scores.values()))
            result[f'行为_{dim_cn}_低特质均值'] = round(low_big5_score, 2)
            result[f'行为_{dim_cn}_低特质偏差'] = round(low_big5_score - 2.0, 2)  # 与目标2的差值

            # 标准差
            if len(low_facet_scores) > 1:
                result[f'行为_{dim_cn}_低特质标准差'] = round(np.std(list(low_facet_scores.values()), ddof=1), 3)
            else:
                result[f'行为_{dim_cn}_低特质标准差'] = 0.0
        else:
            result[f'行为_{dim_cn}_低特质均值'] = None
            result[f'行为_{dim_cn}_低特质标准差'] = None
            result[f'行为_{dim_cn}_低特质偏差'] = None

        # 差值（高-低）
        if high_facet_scores and low_facet_scores:
            result[f'行为_{dim_cn}_差值'] = round(np.mean(list(high_facet_scores.values())) - np.mean(list(low_facet_scores.values())), 2)
        else:
            result[f'行为_{dim_cn}_差值'] = None

    return result


def generate_summary_tables(all_model_data, output_dir, dimension_names):
    """
    生成汇总表格

    Args:
        all_model_data: 所有模型数据列表
        output_dir: 输出目录
        dimension_names: 维度名称映射
    """
    if not all_model_data:
        print("⚠️  没有数据可以生成表格")
        return

    # 创建DataFrame
    df = pd.DataFrame(all_model_data)

    # 计算每个模型的总体偏差（取绝对值）
    dims_cn = ['外向性', '宜人性', '尽责性', '神经质', '开放性']

    for idx, row in df.iterrows():
        # BFI高特质总体偏差（目标=4）
        bfi_high_devs = []
        for dim_cn in dims_cn:
            col = f'BFI_{dim_cn}_高特质偏差'
            if col in row and pd.notna(row[col]):
                bfi_high_devs.append(abs(row[col]))
        df.at[idx, 'BFI_高特质总体偏差'] = round(np.mean(bfi_high_devs), 3) if bfi_high_devs else None

        # BFI低特质总体偏差（目标=2）
        bfi_low_devs = []
        for dim_cn in dims_cn:
            col = f'BFI_{dim_cn}_低特质偏差'
            if col in row and pd.notna(row[col]):
                bfi_low_devs.append(abs(row[col]))
        df.at[idx, 'BFI_低特质总体偏差'] = round(np.mean(bfi_low_devs), 3) if bfi_low_devs else None

        # 行为高特质总体偏差（目标=4）
        beh_high_devs = []
        for dim_cn in dims_cn:
            col = f'行为_{dim_cn}_高特质偏差'
            if col in row and pd.notna(row[col]):
                beh_high_devs.append(abs(row[col]))
        df.at[idx, '行为_高特质总体偏差'] = round(np.mean(beh_high_devs), 3) if beh_high_devs else None

        # 行为低特质总体偏差（目标=2）
        beh_low_devs = []
        for dim_cn in dims_cn:
            col = f'行为_{dim_cn}_低特质偏差'
            if col in row and pd.notna(row[col]):
                beh_low_devs.append(abs(row[col]))
        df.at[idx, '行为_低特质总体偏差'] = round(np.mean(beh_low_devs), 3) if beh_low_devs else None

        # BFI总体偏差（高+低平均）
        all_bfi = bfi_high_devs + bfi_low_devs
        df.at[idx, 'BFI_总体偏差'] = round(np.mean(all_bfi), 3) if all_bfi else None

        # 行为总体偏差（高+低平均）
        all_beh = beh_high_devs + beh_low_devs
        df.at[idx, '行为_总体偏差'] = round(np.mean(all_beh), 3) if all_beh else None

        # 行为比BFI多偏离多少
        if df.at[idx, 'BFI_总体偏差'] is not None and df.at[idx, '行为_总体偏差'] is not None:
            df.at[idx, '行为比BFI多偏离'] = round(df.at[idx, '行为_总体偏差'] - df.at[idx, 'BFI_总体偏差'], 3)
        else:
            df.at[idx, '行为比BFI多偏离'] = None

    # 定义列顺序
    columns = ['模型名称',
               'BFI_总体偏差', '行为_总体偏差', '行为比BFI多偏离',
               'BFI_高特质总体偏差', 'BFI_低特质总体偏差',
               '行为_高特质总体偏差', '行为_低特质总体偏差']

    # 添加各维度详细数据
    for dim_cn in dims_cn:
        columns.extend([
            f'BFI_{dim_cn}_高特质均值',
            f'BFI_{dim_cn}_高特质标准差',
            f'BFI_{dim_cn}_高特质偏差',
            f'BFI_{dim_cn}_低特质均值',
            f'BFI_{dim_cn}_低特质标准差',
            f'BFI_{dim_cn}_低特质偏差',
            f'BFI_{dim_cn}_差值',
            f'行为_{dim_cn}_高特质均值',
            f'行为_{dim_cn}_高特质标准差',
            f'行为_{dim_cn}_高特质偏差',
            f'行为_{dim_cn}_低特质均值',
            f'行为_{dim_cn}_低特质标准差',
            f'行为_{dim_cn}_低特质偏差',
            f'行为_{dim_cn}_差值',
        ])

    # 过滤掉不存在的列
    columns = [col for col in columns if col in df.columns]
    df = df[columns]

    # 保存Excel
    output_excel = output_dir / 'dim_role_summary_full_v2.xlsx'

    try:
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='维度角色注入汇总')

            # 调整列宽
            worksheet = writer.sheets['维度角色注入汇总']
            from openpyxl.utils import get_column_letter

            for idx, col in enumerate(df.columns, 1):
                col_letter = get_column_letter(idx)
                if '模型名称' in col:
                    worksheet.column_dimensions[col_letter].width = 20
                else:
                    worksheet.column_dimensions[col_letter].width = 14

        print(f"📊 Excel汇总表格已保存: {output_excel}")
        print(f"   共 {len(df)} 个模型, {len(df.columns)} 列")
        print(f"   校准方法: 11个显著子维度百分位校准（区间映射法）")

    except ImportError:
        print("⚠️  需要安装 openpyxl: pip install openpyxl")

    # 保存CSV
    output_csv = output_dir / 'dim_role_summary_full_v2.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"📊 CSV汇总表格已保存: {output_csv}")

    # 输出统计摘要
    print(f"\n{'='*80}")
    print(f"统计摘要：")
    print(f"{'='*80}\n")

    for _, row in df.iterrows():
        model_name = row['模型名称']
        print(f"{model_name}:")
        print()

        for dim_cn in dims_cn:
            # BFI结果
            bfi_high_col = f'BFI_{dim_cn}_高特质均值'
            bfi_low_col = f'BFI_{dim_cn}_低特质均值'
            bfi_diff_col = f'BFI_{dim_cn}_差值'

            # 行为测试结果
            beh_high_col = f'行为_{dim_cn}_高特质均值'
            beh_low_col = f'行为_{dim_cn}_低特质均值'
            beh_diff_col = f'行为_{dim_cn}_差值'

            has_bfi = (bfi_high_col in row and pd.notna(row[bfi_high_col]) and
                      bfi_low_col in row and pd.notna(row[bfi_low_col]) and
                      bfi_diff_col in row and pd.notna(row[bfi_diff_col]))
            has_beh = (beh_high_col in row and pd.notna(row[beh_high_col]) and
                      beh_low_col in row and pd.notna(row[beh_low_col]) and
                      beh_diff_col in row and pd.notna(row[beh_diff_col]))

            if has_bfi or has_beh:
                print(f"  【{dim_cn}】")

                if has_bfi:
                    bfi_high = row[bfi_high_col]
                    bfi_low = row[bfi_low_col]
                    bfi_diff = row[bfi_diff_col]
                    print(f"    BFI:  高={bfi_high:.2f} 低={bfi_low:.2f} 差值={bfi_diff:+.2f}")

                if has_beh:
                    beh_high = row[beh_high_col]
                    beh_low = row[beh_low_col]
                    beh_diff = row[beh_diff_col]
                    print(f"    行为: 高={beh_high:.2f} 低={beh_low:.2f} 差值={beh_diff:+.2f}")

                print()

    # 计算平均效果
    print(f"{'='*80}")
    print(f"平均角色注入效果（所有模型）：")
    print(f"{'='*80}\n")

    for dim_cn in dims_cn:
        bfi_diff_col = f'BFI_{dim_cn}_差值'
        beh_diff_col = f'行为_{dim_cn}_差值'

        has_bfi = bfi_diff_col in df.columns and df[bfi_diff_col].notna().sum() > 0
        has_beh = beh_diff_col in df.columns and df[beh_diff_col].notna().sum() > 0

        if has_bfi or has_beh:
            print(f"【{dim_cn}】")

            if has_bfi:
                valid_diffs = df[bfi_diff_col].dropna()
                avg_diff = valid_diffs.mean()
                std_diff = valid_diffs.std(ddof=1) if len(valid_diffs) > 1 else 0
                print(f"  BFI平均差值:  {avg_diff:+.2f} ± {std_diff:.2f} (n={len(valid_diffs)})")

            if has_beh:
                valid_diffs = df[beh_diff_col].dropna()
                avg_diff = valid_diffs.mean()
                std_diff = valid_diffs.std(ddof=1) if len(valid_diffs) > 1 else 0
                print(f"  行为平均差值: {avg_diff:+.2f} ± {std_diff:.2f} (n={len(valid_diffs)})")

            print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='分析维度角色注入实验结果（V2: 11个显著子维度百分位校准）')
    parser.add_argument(
        '--result-dir',
        type=str,
        default='./dim_role_result',
        help='结果目录（默认: ./dim_role_result）'
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
        help='输出目录（默认与result_dir相同）'
    )

    args = parser.parse_args()

    analyze_dimension_role_results(
        result_dir=args.result_dir,
        human_data_file=args.human_data,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()

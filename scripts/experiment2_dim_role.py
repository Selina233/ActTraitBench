#!/usr/bin/env python3
"""
实验2：高低特质实验
分析不同模型在高低角色注入条件下的人格表现
"""

from pathlib import Path
from analyze_dim_role_results_v2 import analyze_dimension_role_results

def main():
    # 设置路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    result_dir = project_root / 'reference_results' / 'experiment2_dim_role' / 'dim_role_result'
    human_data_file = project_root / 'data' / 'data_with_gpt54_scores.json'
    output_dir = project_root / 'outputs' / 'experiment2_dim_role'

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("实验2：高低特质实验")
    print(f"输入: {result_dir}")
    print(f"输出: {output_dir}")
    print("=" * 80)

    # 运行分析
    analyze_dimension_role_results(
        result_dir=result_dir,
        human_data_file=human_data_file,
        output_dir=output_dir
    )

if __name__ == '__main__':
    main()

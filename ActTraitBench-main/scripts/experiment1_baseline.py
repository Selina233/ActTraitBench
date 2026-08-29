#!/usr/bin/env python3
"""
实验1：主实验（基础实验）
计算21个模型的BFI-2自评和行为测试分数
"""

from pathlib import Path
from batch_calculate_scores_v4_multi import calculate_all_scores

def main():
    # 设置路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    results_dir = project_root / 'reference_results' / 'experiment1_baseline' / 'results'
    human_data_file = project_root / 'data' / 'data_with_gpt54_scores.json'
    output_dir = project_root / 'outputs' / 'experiment1_baseline'

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("实验1：主实验（基础实验）")
    print(f"输入: {results_dir}")
    print(f"输出: {output_dir}")
    print("=" * 80)

    # 运行计算
    calculate_all_scores(
        results_dir=str(results_dir),
        human_data_file=str(human_data_file),
        output_dir=str(output_dir)
    )

if __name__ == '__main__':
    main()

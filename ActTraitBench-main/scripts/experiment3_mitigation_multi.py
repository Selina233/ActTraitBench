#!/usr/bin/env python3
"""
实验3：缓解实验（3次运行版本）
计算缓解实验的3次运行平均结果
"""

from pathlib import Path
from batch_calculate_scores_v4_multi import calculate_all_scores

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    results_dir = project_root / 'reference_results' / 'experiment3_mitigation' / 'reflection_results'
    human_data_file = project_root / 'data' / 'data_with_gpt54_scores.json'
    output_dir = project_root / 'outputs' / 'experiment3_mitigation'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("实验3：缓解实验（3次运行版本）")
    print(f"输入: {results_dir}")
    print(f"输出: {output_dir}")
    print("=" * 80)

    calculate_all_scores(
        results_dir=str(results_dir),
        human_data_file=str(human_data_file),
        output_dir=str(output_dir)
    )

if __name__ == '__main__':
    main()

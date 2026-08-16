from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from src.evaluation import EvaluationRunner

    parser = argparse.ArgumentParser(description="运行可复现的三方法教师 Agent 评估")
    parser.add_argument(
        "--mode",
        choices=("quick", "full"),
        default="full",
        help="quick=18×3 演示；full=18×3×3画像×5种子（默认）",
    )
    args = parser.parse_args()
    report = EvaluationRunner().run(mode=args.mode)
    print(f"\n教师 Agent 评估完成（{args.mode}）\n")
    for item in report.summary:
        print(
            f"{item['method']}: 决策质量={item['decision_quality']:.3f}, "
            f"学习增益={item['normalized_gain']:.3f} [{item['gain_ci_low']:.3f}, {item['gain_ci_high']:.3f}], "
            f"单位轮次增益={item['learning_efficiency']:.3f}, 迁移={item['transfer_accuracy']:.3f}"
        )
    print(f"\n报告已保存到: {ROOT / 'output' / 'evaluations'}")


if __name__ == "__main__":
    main()

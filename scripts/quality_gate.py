"""交付前的轻量一致性审计，不修改运行数据。

用法：python scripts/quality_gate.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    development_path = ROOT / "data" / "evaluation_cases.json"
    held_out_path = ROOT / "data" / "evaluation_cases_heldout.json"
    development = load_json(development_path)
    held_out = load_json(held_out_path)
    cases = development + held_out
    ids = [item["case_id"] for item in cases]
    errors: list[str] = []
    if len(development) != 6 or len(held_out) != 12 or len(ids) != 18 or len(ids) != len(set(ids)):
        errors.append("案例规模必须为 6 个开发集 + 12 个留出集，且 ID 不重复")
    courses = {item["goal"]["course"] for item in cases}
    if any(sum(item["goal"]["course"] == course for item in cases) != 6 for course in courses):
        errors.append("三门课程必须各有 6 个案例")

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"fail_under\s*=\s*(\d+)", pyproject)
    if not match or int(match.group(1)) < 92:
        errors.append("pyproject.toml 的覆盖率门槛必须不低于 92")

    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    collected_match = re.search(r"(\d+) tests collected", collected.stdout + collected.stderr)
    test_count = int(collected_match.group(1)) if collected_match else None
    matrix = (ROOT / "需求追踪矩阵.md").read_text(encoding="utf-8")
    if test_count is None or f"{test_count} 项" not in matrix:
        errors.append(f"需求追踪矩阵的测试数量未与当前收集结果同步（当前为 {test_count}）")

    evaluation_files = sorted((ROOT / "output" / "evaluations").glob("evaluation_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if evaluation_files:
        latest = load_json(evaluation_files[0])
        if latest.get("evaluation_protocol", {}).get("mode") == "full" and len(latest.get("case_results", [])) != 810:
            errors.append("最新 full 评估必须包含 810 个方法单元")

    if errors:
        print("质量审计未通过：")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"质量审计通过：18 个案例，{test_count} 项测试，覆盖率门槛 92%，评估结构已核对。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import (  # noqa: E402
    EvaluationReport,
    StudentProfile,
    StudentState,
    TeachingGoal,
    TeachingSession,
)
from src.storage import EvaluationStore, SessionStore  # noqa: E402


def main() -> None:
    with TemporaryDirectory() as temporary:
        store = SessionStore(Path(temporary))
        goal = TeachingGoal(course="程序设计", topic="二分查找", objective="理解边界", knowledge_points=["区间"])
        profile = StudentProfile(name="性能测试")
        started = perf_counter()
        for index in range(1000):
            store.save(
                TeachingSession(
                    display_title=f"性能会话 {index}",
                    goal=goal,
                    profile=profile,
                    state=StudentState(mastery={"区间": 0.3}),
                )
            )
        write_seconds = perf_counter() - started
        started = perf_counter()
        page, total = store.list_metadata(page=25, page_size=20, query="性能")
        query_ms = (perf_counter() - started) * 1000
        assert total == 1000 and len(page) == 20
        print(f"1000 writes: {write_seconds:.3f}s")
        print(f"indexed page query: {query_ms:.3f}ms")
        assert query_ms < 2000

        evaluation_store = EvaluationStore(Path(temporary) / "evaluations")
        report = EvaluationReport(
            seed=20260809,
            methods=[],
            case_results=[],
            summary=[],
            successful_case={},
            failure_case={},
        )
        for _ in range(500):
            evaluation_store.import_report(report)
        started = perf_counter()
        reports = evaluation_store.list_reports()
        report_query_ms = (perf_counter() - started) * 1000
        assert len(reports) == 500
        print(f"500-report archive listing: {report_query_ms:.3f}ms")
        assert report_query_ms < 2000


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from src.config import get_path
from src.models import EvaluationReport, TeachingSession, now_iso

SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,96}$")
_LOCK = RLock()


def _safe_id(value: str, label: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} 包含不安全字符")
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    # Windows antivirus/indexers can briefly hold the destination between close
    # and replace. Keep the write atomic, but tolerate a short transient lock.
    for attempt in range(6):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 5:
                temporary.unlink(missing_ok=True)
                raise
            time.sleep(0.01 * (attempt + 1))


class SessionStore:
    """Versioned JSON store with metadata index, archive and recoverable deletion."""

    def __init__(self, directory: Path | None = None):
        override = os.getenv("TEACHING_AGENT_SESSION_DIR", "").strip()
        self.directory = directory or (Path(override) if override else get_path("sessions"))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.trash = self.directory / ".trash"
        self.trash.mkdir(exist_ok=True)
        self.index_path = self.directory / "index.json"

    @staticmethod
    def _metadata(session: TeachingSession) -> dict:
        return {
            "session_id": session.session_id,
            "display_title": session.display_title or session.goal.topic,
            "topic": session.goal.topic,
            "student_name": session.profile.name,
            "course": session.goal.course,
            "status": str(session.status),
            "answered_rounds": session.answered_rounds(),
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "archived_at": session.archived_at,
        }

    def _read_index(self) -> list[dict]:
        if not self.index_path.exists():
            return self.rebuild_index()
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return self.rebuild_index()

    def _upsert_index(self, session: TeachingSession) -> None:
        rows = [row for row in self._read_index() if row.get("session_id") != session.session_id]
        rows.append(self._metadata(session))
        rows.sort(key=lambda row: row.get("updated_at", ""), reverse=True)
        _atomic_json(self.index_path, rows)

    def save(self, session: TeachingSession) -> Path:
        session_id = _safe_id(session.session_id, "会话 ID")
        session.schema_version = 6
        if not session.display_title:
            session.display_title = session.goal.topic
        path = self.directory / f"{session_id}.json"
        with _LOCK:
            _atomic_json(path, session.model_dump(mode="json"))
            self._upsert_index(session)
        return path

    def load(self, session_id: str) -> TeachingSession:
        session_id = _safe_id(session_id, "会话 ID")
        path = self.directory / f"{session_id}.json"
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload = self._migrate_payload(payload)
        session = TeachingSession.model_validate(payload)
        if session.schema_version < 6:
            session.schema_version = 6
        return session

    @staticmethod
    def _migrate_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Add v4/v5/v6 fields without rewriting the source file on read."""
        migrated = dict(payload)
        version = int(migrated.get("schema_version", 1) or 1)
        goal = migrated.get("goal") or {}
        migrated.setdefault("display_title", goal.get("topic", "未命名会话"))
        migrated.setdefault("available_skill_ids", [])
        migrated.setdefault("skill_snapshot", {})
        migrated.setdefault("imported_history", [])
        state = dict(migrated.get("state") or {})
        if version < 4:
            state.setdefault("phase", "diagnosis")
            if state.get("mastery_model_version") == "evidence-v1":
                state["mastery_model_version"] = "evidence-v1-legacy"
            migrated["state"] = state
            turns = []
            for raw_turn in migrated.get("turns", []) or []:
                turn = dict(raw_turn)
                strategy = turn.get("strategy_skill_id") or turn.get("selected_skill_id")
                content = turn.get("content_skill_id") or turn.get("support_skill_id")
                turn.setdefault("strategy_skill_id", strategy)
                turn.setdefault("content_skill_id", content)
                turn.setdefault(
                    "skill_plan",
                    {
                        "content_skill_id": content,
                        "strategy_skill_id": strategy,
                        "strategy_reason": turn.get("selection_reason", ""),
                        "candidate_content_skill_ids": turn.get("candidate_skill_ids", []),
                    },
                )
                micro = turn.get("micro_step") or {}
                turn.setdefault(
                    "question_contract",
                    {
                        "focus": micro.get("focus", ""),
                        "context": micro.get("context", ""),
                        "requested_target": micro.get("requested_target", ""),
                        "response_mode": micro.get("response_mode", "open"),
                        "option_ids": [item.get("option_id", "") for item in micro.get("options", [])],
                        "expected_signal": micro.get("expected_signal", ""),
                    },
                )
                turn.setdefault(
                    "generation_revisions",
                    [{"revision_index": 1, "teacher_message": turn.get("teacher_message", ""), "reason": "历史会话迁移"}],
                )
                turn.setdefault("llm_trace", [])
                turn.setdefault("phase", "diagnosis")
                turns.append(turn)
            migrated["turns"] = turns
        if version < 5:
            migrated.setdefault(
                "rounds_in_current_run",
                sum(bool((turn or {}).get("student_message", "").strip()) for turn in migrated.get("turns", []) or []),
            )
            migrated["schema_version"] = 5
        state = dict(migrated.get("state") or {})
        state.setdefault("current_difficulty", "unknown")
        state.setdefault("recommended_strategy", "subject")
        state.setdefault("misconception_confirmed", False)
        migrated["state"] = state
        if version < 6:
            migrated["turns"] = [
                {**dict(raw_turn), "difficulty_type": dict(raw_turn).get("difficulty_type", "unknown")}
                for raw_turn in migrated.get("turns", []) or []
            ]
            migrated["schema_version"] = 6
        return migrated

    def list_metadata(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        query: str = "",
        course: str = "",
        status: str = "",
        include_archived: bool = False,
    ) -> tuple[list[dict], int]:
        rows = self._read_index()
        if not include_archived:
            rows = [row for row in rows if not row.get("archived_at")]
        needle = query.strip().lower()
        if needle:
            rows = [row for row in rows if needle in " ".join(map(str, row.values())).lower()]
        if course.strip():
            rows = [row for row in rows if str(row.get("course", "")) == course.strip()]
        if status.strip():
            rows = [
                row for row in rows
                if str(row.get("status", "")).replace("SessionStatus.", "").lower() == status.strip().lower()
            ]
        total = len(rows)
        start = max(0, (page - 1) * page_size)
        return rows[start : start + page_size], total

    def list_courses(self, *, include_archived: bool = False) -> list[str]:
        rows = self._read_index()
        if not include_archived:
            rows = [row for row in rows if not row.get("archived_at")]
        return sorted({str(row.get("course", "")) for row in rows if row.get("course")})

    def list_sessions(self) -> list[Path]:
        return [self.directory / f"{row['session_id']}.json" for row in self._read_index()]

    def rebuild_index(self) -> list[dict]:
        rows: list[dict] = []
        corrupt = self.directory / ".corrupt"
        for path in self.directory.glob("*.json"):
            if path.name == "index.json":
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    session = TeachingSession.model_validate(json.load(handle))
                rows.append(self._metadata(session))
            except (OSError, json.JSONDecodeError, ValueError):
                corrupt.mkdir(exist_ok=True)
                shutil.move(str(path), str(corrupt / path.name))
        rows.sort(key=lambda row: row.get("updated_at", ""), reverse=True)
        _atomic_json(self.index_path, rows)
        return rows

    def update_metadata(self, session_id: str, *, topic: str | None = None, student_name: str | None = None,
                        display_title: str | None = None) -> TeachingSession:
        session = self.load(session_id)
        # Legacy arguments remain accepted, but historical semantic fields are immutable.
        proposed = display_title if display_title is not None else topic
        if proposed is not None:
            cleaned = proposed.strip()
            if not cleaned:
                raise ValueError("展示名称不能为空")
            session.display_title = cleaned
        session.updated_at = now_iso()
        self.save(session)
        return session

    def duplicate(self, session_id: str) -> TeachingSession:
        source = self.load(session_id)
        clone = source.model_copy(deep=True)
        clone.session_id = uuid4().hex[:12]
        clone.display_title = f"{source.display_title or source.goal.topic}（副本）"
        clone.archived_at = None
        clone.deleted_at = None
        clone.created_at = clone.updated_at = now_iso()
        self.save(clone)
        return clone

    def archive(self, session_id: str, archived: bool = True) -> TeachingSession:
        session = self.load(session_id)
        session.archived_at = now_iso() if archived else None
        session.updated_at = now_iso()
        self.save(session)
        return session

    def delete(self, session_id: str) -> bool:
        session_id = _safe_id(session_id, "会话 ID")
        path = self.directory / f"{session_id}.json"
        if not path.exists():
            return False
        with _LOCK:
            shutil.move(str(path), str(self.trash / path.name))
            rows = [row for row in self._read_index() if row.get("session_id") != session_id]
            _atomic_json(self.index_path, rows)
        return True

    def restore(self, session_id: str) -> TeachingSession:
        session_id = _safe_id(session_id, "会话 ID")
        source = self.trash / f"{session_id}.json"
        if not source.exists():
            raise FileNotFoundError(session_id)
        shutil.move(str(source), str(self.directory / source.name))
        session = self.load(session_id)
        self._upsert_index(session)
        return session

    def list_trash(self) -> list[str]:
        return [path.stem for path in sorted(self.trash.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)]

    def import_session(self, session: TeachingSession) -> TeachingSession:
        if (self.directory / f"{session.session_id}.json").exists():
            session = session.model_copy(deep=True)
            session.session_id = uuid4().hex[:12]
            session.display_title = f"{session.display_title or session.goal.topic}（导入副本）"
            session.created_at = session.updated_at = now_iso()
        self.save(session)
        return session

    def empty_trash(self) -> int:
        files = list(self.trash.glob("*.json"))
        for path in files:
            path.unlink()
        return len(files)


class EvaluationStore:
    """Immutable evaluation archives with recoverable bundle deletion."""

    def __init__(self, directory: Path | None = None):
        override = os.getenv("TEACHING_AGENT_EVALUATION_DIR", "").strip()
        self.directory = directory or (Path(override) if override else get_path("evaluations"))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.trash = self.directory / ".trash"
        self.trash.mkdir(exist_ok=True)
        self.archive_dir = self.directory / ".archive"
        self.archive_dir.mkdir(exist_ok=True)
        self.index_path = self.directory / "index.json"

    def _read_index(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return self.rebuild_index()
        try:
            value = json.loads(self.index_path.read_text(encoding="utf-8"))
            return value if isinstance(value, list) else self.rebuild_index()
        except (OSError, json.JSONDecodeError):
            return self.rebuild_index()

    @staticmethod
    def _metadata(file_name: str, report: EvaluationReport, archived_at: str | None = None) -> dict[str, Any]:
        return {
            "file_name": file_name,
            "generated_at": report.generated_at,
            "seed": report.seed,
            "mode": report.evaluation_protocol.get("mode", "unknown"),
            "case_count": len({item.case_id for item in report.case_results}),
            "method_count": len(report.methods),
            "human_evaluation_status": report.human_evaluation_status,
            "archived_at": archived_at,
        }

    def register_report(self, file_name: str, report: EvaluationReport, archived_at: str | None = None) -> None:
        rows = [row for row in self._read_index() if row.get("file_name") != file_name]
        rows.append(self._metadata(file_name, report, archived_at))
        rows.sort(key=lambda row: str(row.get("generated_at", "")), reverse=True)
        with _LOCK:
            _atomic_json(self.index_path, rows)

    def rebuild_index(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in self.directory.glob("evaluation_*.json"):
            try:
                report = EvaluationReport.model_validate_json(path.read_text(encoding="utf-8"))
                rows.append(self._metadata(path.name, report))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        rows.sort(key=lambda row: str(row.get("generated_at", "")), reverse=True)
        _atomic_json(self.index_path, rows)
        return rows

    def list_report_metadata(
        self, *, page: int = 1, page_size: int = 20, query: str = "", include_archived: bool = False
    ) -> tuple[list[dict[str, Any]], int]:
        rows = self._read_index()
        rows = [
            row for row in rows
            if include_archived or not row.get("archived_at")
        ]
        needle = query.strip().lower()
        if needle:
            rows = [row for row in rows if needle in " ".join(map(str, row.values())).lower()]
        total = len(rows)
        start = max(0, (page - 1) * page_size)
        return rows[start : start + page_size], total

    @staticmethod
    def _bundle_names(file_name: str) -> list[str]:
        safe_name = Path(file_name).name
        if safe_name != file_name or not safe_name.startswith("evaluation_") or not safe_name.endswith(".json"):
            raise ValueError("评估文件名不安全")
        token = safe_name[len("evaluation_") : -len(".json")]
        names = [f"evaluation_{token}{suffix}" for suffix in (".json", ".csv", ".md")]
        names += [f"annotation_blind_{token}.csv", f"annotation_key_{token}.csv"]
        return names

    def list_reports(self) -> list[Path]:
        rows = [row for row in self._read_index() if not row.get("archived_at")]
        return [self.directory / str(row["file_name"]) for row in rows if (self.directory / str(row["file_name"])).exists()]

    def load(self, file_name: str) -> EvaluationReport:
        safe_name = Path(file_name).name
        if safe_name != file_name or not safe_name.startswith("evaluation_") or not safe_name.endswith(".json"):
            raise ValueError("评估文件名不安全")
        with (self.directory / safe_name).open("r", encoding="utf-8") as handle:
            return EvaluationReport.model_validate(json.load(handle))

    def import_report(self, report: EvaluationReport) -> Path:
        stamp = now_iso().replace(":", "").replace("+", "_").replace("-", "")
        path = self.directory / f"evaluation_imported_{stamp}_{uuid4().hex[:6]}.json"
        _atomic_json(path, report.model_dump(mode="json"))
        self.register_report(path.name, report)
        return path

    def save_human_review(
        self,
        report_file_name: str,
        rows: list[dict[str, str]],
        summary: dict[str, Any],
    ) -> Path:
        """Persist blind ratings as an independent, non-mutating attachment."""
        safe_name = Path(report_file_name).name
        if safe_name != report_file_name or not safe_name.startswith("evaluation_") or not safe_name.endswith(".json"):
            raise ValueError("评估文件名不安全")
        token = safe_name[len("evaluation_") : -len(".json")]
        path = self.directory / f"human_review_{token}_{uuid4().hex[:6]}.json"
        _atomic_json(
            path,
            {
                "schema_version": 1,
                "report_file": safe_name,
                "created_at": now_iso(),
                "summary": summary,
                "rows": rows,
            },
        )
        return path

    def archive(self, file_name: str) -> list[Path]:
        moved = []
        for name in self._bundle_names(file_name):
            source = self.directory / name
            if source.exists():
                target = self.archive_dir / name
                shutil.move(str(source), str(target))
                moved.append(target)
        if moved:
            rows = self._read_index()
            for row in rows:
                if row.get("file_name") == file_name:
                    row["archived_at"] = now_iso()
            _atomic_json(self.index_path, rows)
        return moved

    def list_archived(self) -> list[str]:
        return [
            path.name
            for path in sorted(
                self.archive_dir.glob("evaluation_*.json"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
        ]

    def restore_archive(self, file_name: str) -> list[Path]:
        restored = []
        for name in self._bundle_names(file_name):
            source = self.archive_dir / name
            if source.exists():
                target = self.directory / name
                shutil.move(str(source), str(target))
                restored.append(target)
        if restored:
            rows = self._read_index()
            for row in rows:
                if row.get("file_name") == file_name:
                    row["archived_at"] = None
            _atomic_json(self.index_path, rows)
        return restored

    def delete_bundle(self, file_name: str) -> list[Path]:
        moved = []
        for name in self._bundle_names(file_name):
            source = self.directory / name
            if source.exists():
                target = self.trash / name
                shutil.move(str(source), str(target))
                moved.append(target)
        if moved:
            rows = [row for row in self._read_index() if row.get("file_name") != file_name]
            _atomic_json(self.index_path, rows)
        return moved

    def restore_bundle(self, file_name: str) -> list[Path]:
        token = Path(file_name).name.removeprefix("evaluation_").removesuffix(".json")
        restored = []
        for source in self.trash.glob(f"*{token}*"):
            target = self.directory / source.name
            shutil.move(str(source), str(target))
            restored.append(target)
        if restored:
            try:
                self.register_report(file_name, self.load(file_name))
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        return restored

    def list_trash(self) -> list[str]:
        return [path.name for path in sorted(self.trash.glob("evaluation_*.json"), key=lambda item: item.stat().st_mtime, reverse=True)]

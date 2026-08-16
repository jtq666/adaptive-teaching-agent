from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    """Never let automated tests write into the demonstrator's production archives."""
    base = ROOT / ".pytest-tmp" / "isolated-runtime"
    base.mkdir(parents=True, exist_ok=True)
    sessions = base / "sessions"
    evaluations = base / "evaluations"
    skill_runtime = Path(tempfile.mkdtemp(prefix="teaching-agent-skills-")) / "skills"
    sessions.mkdir(parents=True, exist_ok=True)
    evaluations.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "data" / "skills", skill_runtime, dirs_exist_ok=True)
    os.environ["TEACHING_AGENT_SESSION_DIR"] = str(sessions)
    os.environ["TEACHING_AGENT_EVALUATION_DIR"] = str(evaluations)
    os.environ["TEACHING_AGENT_SKILL_DIR"] = str(skill_runtime)

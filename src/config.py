from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
# Explicit process variables take precedence. This lets tests and deployments
# force offline mode with ``LLM_API_KEY=""`` without a local .env overriding it.
load_dotenv(ROOT_DIR / ".env", override=False)


@lru_cache(maxsize=1)
def load_settings() -> dict[str, Any]:
    path = ROOT_DIR / "config" / "settings.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_agent_settings() -> dict[str, Any]:
    return dict(load_settings().get("agent", {}))


def get_llm_settings() -> dict[str, Any]:
    config = dict(load_settings().get("llm", {}))
    force_offline = os.getenv("TEACHING_AGENT_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}
    config.update(
        {
            "api_key": "" if force_offline else os.getenv("LLM_API_KEY", "").strip(),
            "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip(),
            "model": os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
        }
    )
    return config


def get_path(name: str) -> Path:
    override_names = {
        "skills": "TEACHING_AGENT_SKILL_DIR",
        "skill_custom": "TEACHING_AGENT_CUSTOM_SKILL_DIR",
        "cases": "TEACHING_AGENT_CASES_PATH",
        "sessions": "TEACHING_AGENT_SESSION_DIR",
        "evaluations": "TEACHING_AGENT_EVALUATION_DIR",
    }
    override = os.getenv(override_names.get(name, ""), "").strip() if name in override_names else ""
    if override:
        path = Path(override)
        if path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)
        return path
    configured = load_settings().get("paths", {}).get(name)
    if not configured:
        raise KeyError(f"未知路径配置: {name}")
    path = ROOT_DIR / configured
    if path.suffix == "":
        path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_dirs() -> None:
    get_path("sessions")
    get_path("evaluations")

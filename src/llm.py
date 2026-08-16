from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI

from src.config import get_llm_settings


class LLMUnavailableError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(self, settings: dict[str, Any] | None = None):
        self.settings = settings or get_llm_settings()
        self.api_key = self.settings.get("api_key", "")
        self.base_url = self.settings.get("base_url") or None
        self.model = self.settings.get("model", "gpt-4o-mini")
        self.timeout = int(self.settings.get("timeout_seconds", 90))
        self.retries = int(self.settings.get("retries", 2))
        self.total_calls = 0
        self.last_call: dict[str, Any] = {}
        self.client = (
            OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
            if self.api_key
            else None
        )

    @property
    def available(self) -> bool:
        return self.client is not None

    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        if not self.client:
            raise LLMUnavailableError("未配置 LLM_API_KEY，当前使用离线规则模式")
        last_error: Exception | None = None
        started = time.perf_counter()
        for attempt in range(self.retries + 1):
            self.total_calls += 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=temperature,
                    max_tokens=int(self.settings.get("max_tokens", 1800)),
                )
                content = response.choices[0].message.content or ""
                self.last_call = {
                    "operation": "chat",
                    "model": self.model,
                    "attempts": attempt + 1,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "success": True,
                }
                return content
            except Exception as exc:  # SDK and providers expose different exception types
                last_error = exc
                if attempt < self.retries and self._is_retryable(exc):
                    time.sleep(0.5 * (attempt + 1))
        self.last_call = {
            "operation": "chat",
            "model": self.model,
            "attempts": self.retries + 1,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "success": False,
            "error_class": type(last_error).__name__ if last_error else "UnknownError",
        }
        raise LLMUnavailableError(f"LLM 调用失败: {last_error}")

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        """Retry only transient provider failures, never auth/schema errors."""
        name = type(error).__name__.lower()
        status = getattr(error, "status_code", None)
        if status in {408, 409, 425, 429} or isinstance(status, int) and status >= 500:
            return True
        return any(token in name for token in ("timeout", "ratelimit", "connection", "serviceunavailable"))

    def structured(
        self,
        system: str,
        user: str,
        schema_hint: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        prompt = f"{user}\n\n只输出单个 JSON 对象，不要 Markdown。字段约束：\n{schema_hint}"
        text = self.chat(system, prompt, temperature=temperature)
        self.last_call["operation"] = "structured"
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("LLM 未返回 JSON 对象")
        raw = match.group(0)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            # Models sometimes place LaTeX or another literal backslash inside
            # a JSON string (for example ``\(``). Escape only sequences that
            # are not valid JSON escapes; preserve valid \n, \t and \uXXXX.
            repaired = self._escape_invalid_json_backslashes(raw)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                raise exc from None

    @staticmethod
    def _escape_invalid_json_backslashes(raw: str) -> str:
        valid = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
        output: list[str] = []
        index = 0
        while index < len(raw):
            char = raw[index]
            if char == "\\" and index + 1 < len(raw):
                next_char = raw[index + 1]
                if next_char not in valid:
                    output.append("\\\\")
                else:
                    output.append(char)
            else:
                output.append(char)
            index += 1
        return "".join(output)

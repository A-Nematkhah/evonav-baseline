"""
LLM client for reward-code generation.

Thin provider wrappers (Groq / local vLLM OpenAI-compatible / scripted).
Interface mirrors mobile_robot_env.rewards.generators: no crowd-nav logic
inside the client — only text in, raw completion strings out.

Primary API for EvoNav Stage I:
    generate(prompt: str, n: int) -> list[str]
"""

from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence


class LLMClient(ABC):
    """Minimal text-completion interface (sister-project compatible)."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        raise NotImplementedError

    def generate(self, prompt: str, n: int = 1) -> List[str]:
        """Return ``n`` raw completion strings for the same prompt."""
        if n <= 0:
            raise ValueError("n must be positive")
        return [self.complete(prompt) for _ in range(n)]


class ScriptedLLMClient(LLMClient):
    """Deterministic stand-in: returns pre-canned completions in order."""

    def __init__(self, completions: Sequence[str]) -> None:
        if not completions:
            raise ValueError("ScriptedLLMClient requires at least one completion.")
        self._completions = tuple(completions)
        self._index = 0

    def complete(self, prompt: str) -> str:
        if self._index >= len(self._completions):
            raise IndexError(
                f"ScriptedLLMClient has no remaining completions "
                f"({len(self._completions)} provided)."
            )
        completion = self._completions[self._index]
        self._index += 1
        return completion

    @property
    def remaining(self) -> int:
        return max(0, len(self._completions) - self._index)


class GroqLLMClient(LLMClient):
    """
    Live Groq chat completion (same design as mobile_robot_env GroqLLMClient).

    Auth (first match wins):
      1. ``api_key=`` constructor argument (single key)
      2. ``GROQ_API_KEY`` environment variable (single key)
      3. ``evonav_env/groq_keys.json`` key pool via :mod:`key_manager`
         (rate-limit rotation; copy from ``groq_keys.json.example``)

    Requires optional dependency: ``pip install groq``.
    """

    DEFAULT_MODEL = "openai/gpt-oss-120b"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.4,
        max_tokens: int = 1500,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_attempts: int = 3,
        retry_delay_seconds: float = 3.0,
        use_key_pool: bool = True,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or (
            "You are a reward-function designer. Reply with one Python code block only."
        )
        if int(max_attempts) < 1:
            raise ValueError("max_attempts must be >= 1")
        if float(retry_delay_seconds) < 0.0:
            raise ValueError("retry_delay_seconds must be >= 0")
        self.max_attempts = int(max_attempts)
        self.retry_delay_seconds = float(retry_delay_seconds)

        self._key_manager = None
        explicit = (api_key or "").strip() or None
        if explicit:
            self.api_key = explicit
        else:
            from crowd_nav.reward_search.key_manager import (
                groq_key_pool_available,
                resolve_single_groq_api_key,
            )

            env_key = resolve_single_groq_api_key()
            if env_key:
                self.api_key = env_key
            elif use_key_pool and groq_key_pool_available():
                from crowd_nav.reward_search.key_manager import get_key_manager

                self._key_manager = get_key_manager()
                self.api_key = None
            else:
                self.api_key = None

    @staticmethod
    def _is_non_retryable(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        text = str(exc).lower()
        if status == 404 or "model_not_found" in text or "does not exist" in text:
            return True
        if status == 400 and "model" in text:
            return True
        return False

    def complete(self, prompt: str) -> str:
        if not self.api_key and self._key_manager is None:
            raise RuntimeError(
                "GroqLLMClient requires GROQ_API_KEY, api_key=..., or "
                "evonav_env/groq_keys.json (see groq_keys.json.example)"
            )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        create_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self._key_manager is not None:
            try:
                response = self._key_manager.chat_completion(**create_kwargs)
            except ImportError as exc:
                raise RuntimeError(
                    "GroqLLMClient requires the 'groq' package. "
                    "Install with: pip install groq"
                ) from exc
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("Groq returned an empty completion.")
            return str(content)

        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError(
                "GroqLLMClient requires the 'groq' package. "
                "Install with: pip install groq"
            ) from exc

        client = Groq(api_key=self.api_key)
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = client.chat.completions.create(**create_kwargs)
                content = response.choices[0].message.content
                if not content:
                    raise RuntimeError("Groq returned an empty completion.")
                return str(content)
            except Exception as exc:  # noqa: BLE001 — retry transient errors
                last_error = exc
                if self._is_non_retryable(exc):
                    raise RuntimeError(
                        f"Groq LLM non-retryable error for model={self.model!r}: {exc}"
                    ) from exc
                if attempt >= self.max_attempts:
                    break
                time.sleep(self.retry_delay_seconds)
        raise RuntimeError(
            f"Groq LLM failed after {self.max_attempts} attempts. Last error: {last_error}"
        ) from last_error


class VLLMLLMClient(LLMClient):
    """
    Local gpt-oss-120B (or any chat model) served by vLLM's OpenAI-compatible API.

    Default base URL: ``http://127.0.0.1:8000/v1`` (override with ``base_url``
    or ``VLLM_BASE_URL``). Uses the ``openai`` Python SDK when available,
    otherwise urllib JSON.
    """

    DEFAULT_MODEL = "gpt-oss-120b"
    DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        temperature: float = 0.4,
        max_tokens: int = 1500,
        system_prompt: Optional[str] = None,
        max_attempts: int = 3,
        retry_delay_seconds: float = 3.0,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL") or self.DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or (
            "You are a reward-function designer. Reply with one Python code block only."
        )
        self.max_attempts = int(max_attempts)
        self.retry_delay_seconds = float(retry_delay_seconds)

    def complete(self, prompt: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._complete_once(prompt)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                time.sleep(self.retry_delay_seconds)
        raise RuntimeError(
            f"vLLM LLM failed after {self.max_attempts} attempts. Last error: {last_error}"
        ) from last_error

    def _complete_once(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            from openai import OpenAI

            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("vLLM returned an empty completion.")
            return str(content)
        except ImportError:
            return self._complete_urllib(messages)

    def _complete_urllib(self, messages: list) -> str:
        import json
        import urllib.error
        import urllib.request

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"vLLM request failed: {exc}") from exc
        content = body["choices"][0]["message"]["content"]
        if not content:
            raise RuntimeError("vLLM returned an empty completion.")
        return str(content)


class SeedVariantLLMClient(LLMClient):
    """
    Always emits a valid ``compute_reward`` variant of the D.5 seed.

    Used for dry-runs / ``--llm seed`` so Algorithm 1 can execute without
    an API key or a huge scripted completion list.
    """

    def __init__(self, base_code: Optional[str] = None) -> None:
        from crowd_nav.reward_search.prompts import D5_SEED_FUNCTION

        self._base = (base_code or D5_SEED_FUNCTION).strip()
        self._n = 0

    def complete(self, prompt: str) -> str:
        self._n += 1
        pot = 2.0 + 0.05 * ((self._n % 20) - 10)
        # Tiny localized tweak so Stage I/II/III see distinct codes.
        code = (
            "```python\n"
            "def compute_reward(state):\n"
            "    success_reward = 10.0\n"
            "    collision_penalty = -20.0\n"
            f"    pot_factor = {pot:.4f}\n"
            "    if state.reaching_goal:\n"
            "        return float(success_reward)\n"
            "    if state.collision:\n"
            "        return float(collision_penalty)\n"
            "    dist = ((state.robot.px - state.robot.gx) ** 2 + "
            "(state.robot.py - state.robot.gy) ** 2) ** 0.5\n"
            "    return float(pot_factor * (-dist))\n"
            "```\n"
        )
        return code


def make_llm_client(
    provider: str = "scripted",
    *,
    completions: Optional[Sequence[str]] = None,
    model: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """
    Factory: ``provider`` in {``scripted``, ``seed``, ``groq``, ``vllm``}.

    For ``scripted``, pass ``completions=...``.
    For ``seed``, returns ``SeedVariantLLMClient`` (no API key).
    """
    name = str(provider).strip().lower()
    if name in ("seed", "seed-variant", "d5"):
        return SeedVariantLLMClient()
    if name == "scripted":
        if not completions:
            raise ValueError("make_llm_client(scripted) requires completions=...")
        return ScriptedLLMClient(completions)
    if name == "groq":
        return GroqLLMClient(model=model or GroqLLMClient.DEFAULT_MODEL, **kwargs)
    if name in ("vllm", "local", "gpt-oss", "gpt-oss-120b"):
        return VLLMLLMClient(model=model or VLLMLLMClient.DEFAULT_MODEL, **kwargs)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


def extract_python_code(completion: str) -> str:
    """Pull the first fenced Python (or generic) code block from an LLM reply."""
    text = completion.strip()
    fence = "```python"
    if fence in text:
        start = text.index(fence) + len(fence)
        rest = text[start:]
        end = rest.find("```")
        if end >= 0:
            return rest[:end].strip()
        return rest.strip()
    if "```" in text:
        start = text.index("```") + 3
        rest = text[start:]
        if rest.startswith("\n"):
            rest = rest[1:]
        end = rest.find("```")
        if end >= 0:
            return rest[:end].strip()
        return rest.strip()
    return text


_DEF_RE = re.compile(
    r"^def\s+(cal_reward|seed_reward_func|compute_reward_v\d+|[\w]+_v2)\s*\(",
    re.MULTILINE,
)


def normalize_to_compute_reward(code: str, target_name: str = "compute_reward") -> str:
    """
    Ensure the top-level function is named ``compute_reward`` for the sandbox.

    Renames common paper aliases (cal_reward, *_v2, seed_reward_func) only.
    """
    text = code.strip()
    match = re.search(r"^def\s+(\w+)\s*\(", text, re.MULTILINE)
    if not match:
        return text
    name = match.group(1)
    if name == target_name:
        return text
    # Rename first def only.
    return re.sub(
        rf"^def\s+{re.escape(name)}\s*\(",
        f"def {target_name}(",
        text,
        count=1,
        flags=re.MULTILINE,
    )

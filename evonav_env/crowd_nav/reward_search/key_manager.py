"""
Groq API key pool (rate-limit rotation).

Loads keys from ``evonav_env/groq_keys.json`` (or ``GROQ_KEYS_PATH``).
Falls back to a single ``GROQ_API_KEY`` env var when the JSON file is absent.

Copy ``groq_keys.json.example`` → ``groq_keys.json`` and fill in your keys.
Never commit ``groq_keys.json``.

Usage::

    from crowd_nav.reward_search.key_manager import get_key_manager

    manager = get_key_manager()
    response = manager.chat_completion(
        model="openai/gpt-oss-120b",
        messages=[...],
        max_tokens=1500,
        temperature=0.4,
    )
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

COOLDOWN_SECONDS = 60
_PLACEHOLDER_PREFIXES = ("gsk_REPLACE", "gsk_your_", "YOUR_KEY")

_EVONAV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_GROQ_KEYS_PATH = os.path.join(_EVONAV_ROOT, "groq_keys.json")


def _is_rate_limit_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    text = str(exc).lower()
    return "rate limit" in text or "429" in text


def _is_placeholder_key(key: str) -> bool:
    k = (key or "").strip()
    if not k:
        return True
    return any(k.startswith(p) for p in _PLACEHOLDER_PREFIXES)


def resolve_groq_keys_path() -> str:
    return os.environ.get("GROQ_KEYS_PATH", DEFAULT_GROQ_KEYS_PATH)


def load_groq_keys(keys_path: Optional[str] = None) -> List[str]:
    """
    Load Groq keys from JSON ``{"keys": ["gsk_...", ...]}``.

    Returns an empty list if the file is missing (caller may fall back to env).
    """
    path = keys_path or resolve_groq_keys_path()
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("keys", [])
    if not isinstance(raw, list):
        raise ValueError(f"{path}: 'keys' must be a JSON array")
    return [str(k).strip() for k in raw if k and not _is_placeholder_key(str(k))]


def resolve_single_groq_api_key() -> Optional[str]:
    """``GROQ_API_KEY`` env var, if set and non-empty."""
    key = (os.environ.get("GROQ_API_KEY") or "").strip()
    return key or None


class GroqKeyManager:
    """Rotate through a pool of Groq keys on HTTP 429 / rate-limit errors."""

    def __init__(self, keys_path: Optional[str] = None, *, keys: Optional[List[str]] = None):
        self.keys_path = keys_path or resolve_groq_keys_path()
        if keys is not None:
            self.keys = [k for k in keys if k and not _is_placeholder_key(k)]
        else:
            self.keys = load_groq_keys(self.keys_path)
        if not self.keys:
            raise RuntimeError(
                f"No Groq API keys available. Set GROQ_API_KEY or create "
                f"{self.keys_path} from groq_keys.json.example"
            )
        self.index = 0
        self.cooldown_until = {key: 0.0 for key in self.keys}

    def _current_key(self) -> str:
        return self.keys[self.index]

    def _advance(self) -> None:
        self.index = (self.index + 1) % len(self.keys)

    def _mark_rate_limited(self, key: str) -> None:
        self.cooldown_until[key] = time.time() + COOLDOWN_SECONDS
        logger.warning(
            "Groq key ...%s rate-limited; cooling down %ds",
            key[-6:],
            COOLDOWN_SECONDS,
        )

    def _select_available_key(self) -> str:
        now = time.time()
        for _ in range(len(self.keys)):
            key = self._current_key()
            if self.cooldown_until[key] <= now:
                return key
            self._advance()
        return self._current_key()

    def get_client(self) -> Tuple[Any, str]:
        from groq import Groq

        key = self._select_available_key()
        return Groq(api_key=key), key

    def chat_completion(self, **kwargs: Any) -> Any:
        """
        Same signature as ``client.chat.completions.create(**kwargs)``.

        Tries each key in the pool on rate-limit errors; other errors propagate.
        """
        last_exception: Optional[Exception] = None
        for _ in range(len(self.keys)):
            client, key = self.get_client()
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                if _is_rate_limit_error(exc):
                    self._mark_rate_limited(key)
                    self._advance()
                    continue
                raise
        raise RuntimeError(
            f"All {len(self.keys)} Groq API keys are rate-limited or failing. "
            f"Last error: {last_exception}"
        )


_manager_singleton: Optional[GroqKeyManager] = None


def get_key_manager(*, keys_path: Optional[str] = None, force_reload: bool = False) -> GroqKeyManager:
    """Lazily build one ``GroqKeyManager`` per process."""
    global _manager_singleton
    if force_reload or _manager_singleton is None:
        _manager_singleton = GroqKeyManager(keys_path=keys_path)
    return _manager_singleton


def groq_key_pool_available(keys_path: Optional[str] = None) -> bool:
    """True if ``groq_keys.json`` exists and has at least one real key."""
    try:
        return len(load_groq_keys(keys_path)) > 0
    except (OSError, ValueError, json.JSONDecodeError):
        return False

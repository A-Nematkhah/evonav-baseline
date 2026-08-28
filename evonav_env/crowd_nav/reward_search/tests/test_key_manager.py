"""Tests for Groq key pool (no live API calls)."""

from __future__ import annotations

import json

import pytest

from crowd_nav.reward_search.key_manager import (
    GroqKeyManager,
    _is_rate_limit_error,
    load_groq_keys,
)


def test_is_rate_limit_error():
    class E429(Exception):
        status_code = 429

    assert _is_rate_limit_error(E429("x"))
    assert _is_rate_limit_error(RuntimeError("HTTP 429 too many"))


def test_load_keys_filters_placeholders(tmp_path):
    path = tmp_path / "groq_keys.json"
    path.write_text(
        json.dumps(
            {
                "keys": [
                    "gsk_REPLACE_ME",
                    "gsk_real_key_abcdefghijklmnop",
                    "",
                ]
            }
        ),
        encoding="utf-8",
    )
    keys = load_groq_keys(str(path))
    assert keys == ["gsk_real_key_abcdefghijklmnop"]


def test_manager_requires_real_keys(tmp_path):
    path = tmp_path / "groq_keys.json"
    path.write_text(json.dumps({"keys": ["gsk_REPLACE"]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="No Groq API keys"):
        GroqKeyManager(keys_path=str(path))


def test_manager_accepts_inline_keys():
    m = GroqKeyManager(keys=["gsk_test_key_aaaaaaaaaaaaaaaa"])
    assert m.keys == ["gsk_test_key_aaaaaaaaaaaaaaaa"]

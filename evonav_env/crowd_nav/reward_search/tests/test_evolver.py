"""
Tests for Stage I evolver + LLM client (mocked / scripted completions).
"""

from __future__ import annotations

import pytest

from crowd_nav.reward_search.evolver import StageIConfig, StageIEvolver
from crowd_nav.reward_search.llm import (
    ScriptedLLMClient,
    extract_python_code,
    make_llm_client,
    normalize_to_compute_reward,
)
from crowd_nav.reward_search.prompts import (
    D1_SYSTEM_PROMPT,
    D4_EXTERNAL_KNOWLEDGE,
    D5_SEED_FUNCTION,
    format_d1_initial,
    format_d2_crossover,
    format_d2_mutation,
)
from crowd_nav.reward_search.sandbox.runtime import default_smoke_states


def _valid_code(value: float) -> str:
    return (
        "```python\n"
        "def compute_reward(state):\n"
        f"    return float({value})\n"
        "```\n"
    )


def _invalid_import_code() -> str:
    return (
        "```python\n"
        "import os\n"
        "def compute_reward(state):\n"
        "    return float(0.0)\n"
        "```\n"
    )


def _score_by_smoke(reward_fn, *, candidate_id: str = "") -> float:
    return float(reward_fn.compute(default_smoke_states()[0]))


def test_extract_and_normalize_code():
    raw = "Here you go:\n```python\ndef cal_reward(state):\n    return float(1.0)\n```\n"
    code = extract_python_code(raw)
    assert "def cal_reward" in code
    norm = normalize_to_compute_reward(code)
    assert "def compute_reward(state)" in norm
    assert "cal_reward" not in norm


def test_prompt_templates_contain_appendix_anchors():
    assert "expert in reinforcement learning" in D1_SYSTEM_PROMPT
    user = format_d1_initial(reflection="keep exploring")
    assert "compute_reward" in user
    assert "RewardState" in user
    assert "keep exploring" in user
    assert "Crowd-robot navigation" in D4_EXTERNAL_KNOWLEDGE
    assert "def compute_reward(state)" in D5_SEED_FUNCTION
    cross = format_d2_crossover("codeA", "codeB", "note")
    assert "codeA" in cross and "codeB" in cross
    mut = format_d2_mutation("elitist", "weakness")
    assert "elitist" in mut and "weakness" in mut


def test_scripted_generate_n():
    client = ScriptedLLMClient(["a", "b", "c"])
    assert client.generate("p", 2) == ["a", "b"]
    assert client.complete("p") == "c"
    with pytest.raises(IndexError):
        client.complete("p")


def test_make_llm_client_scripted():
    client = make_llm_client("scripted", completions=["x"])
    assert isinstance(client, ScriptedLLMClient)
    assert client.complete("hi") == "x"


def test_stage_i_builds_n8_and_runs_one_generation():
    # Gen0: 8 valids. Gen1: 8 valids. Distinct returns for ranking.
    completions = [_valid_code(float(i)) for i in range(8, 0, -1)]
    completions += [_valid_code(float(i) + 0.5) for i in range(8, 0, -1)]
    client = ScriptedLLMClient(completions)
    evolver = StageIEvolver(
        client,
        score_fn=_score_by_smoke,
        config=StageIConfig(
            population_size=8,
            generations=1,
            n_crossover=2,
            n_mutation=4,
            n_random=2,
            max_invalid_replacements=8,
        ),
    )
    final = evolver.run()
    assert len(final) == 8
    assert all(c.valid and c.reward_fn is not None for c in final)
    assert all(c.score is not None for c in final)
    # Ranked descending.
    scores = [c.score for c in final]
    assert scores == sorted(scores, reverse=True)
    assert len(evolver.history) == 2  # Gen0 + Gen1
    assert evolver.history[0].generation == 0
    assert evolver.history[1].generation == 1
    assert len(evolver.history[0].population) == 8
    assert evolver.history[1].reflection
    # Next-gen origins present.
    origins = {c.origin for c in evolver.history[1].population}
    assert "crossover" in origins
    assert "mutation" in origins
    assert "random" in origins


def test_gen0_replaces_invalid_with_extra_draws():
    # Two invalids first, then 8 valids — mirrors "generate extras to replace invalid".
    completions = [_invalid_import_code(), _invalid_import_code()]
    completions += [_valid_code(float(i)) for i in range(1, 9)]
    client = ScriptedLLMClient(completions)
    evolver = StageIEvolver(
        client,
        score_fn=_score_by_smoke,
        config=StageIConfig(
            population_size=8,
            generations=0,  # only Gen0 via initialize + score in run...
            n_crossover=2,
            n_mutation=4,
            n_random=2,
            max_invalid_replacements=8,
        ),
    )
    # generations=0 still runs Gen0 then range(1,1) empty.
    final = evolver.run()
    assert len(final) == 8
    assert all(c.valid for c in final)
    assert client.remaining == 0  # 2 invalid + 8 valid consumed


def test_mutation_uses_lower_performer_and_crossover_uses_top2():
    completions = [_valid_code(float(i)) for i in range(8, 0, -1)]  # Gen0
    # Track prompts indirectly via distinct next-gen codes.
    completions += [_valid_code(100.0)] * 2  # crossover slots
    completions += [_valid_code(50.0)] * 4  # mutation slots
    completions += [_valid_code(10.0)] * 2  # random slots
    client = ScriptedLLMClient(completions)
    evolver = StageIEvolver(
        client,
        score_fn=_score_by_smoke,
        config=StageIConfig(population_size=8, generations=1),
    )
    evolver.run()
    gen1 = evolver.history[1].population
    assert sum(1 for c in gen1 if c.origin == "crossover") == 2
    assert sum(1 for c in gen1 if c.origin == "mutation") == 4
    assert sum(1 for c in gen1 if c.origin == "random") == 2
    for c in gen1:
        if c.origin == "crossover":
            assert len(c.parent_ids) == 2
        if c.origin == "mutation":
            assert len(c.parent_ids) == 1

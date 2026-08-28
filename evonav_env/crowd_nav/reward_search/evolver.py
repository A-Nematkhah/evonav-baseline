"""
EvoNav Stage I evolutionary loop (analytical Score1).

Population N=8, G1=10 generations. Each generation:
  1. Score candidates via ``score1_for_dataset`` (injected / Phase 2).
  2. Rank by score.
  3. Build next population: mutation (lower performers + weakness feedback),
     crossover (top-2), occasional random restart (D.1) — **no** novelty
     filter / archive (those are AMFRS-only).
  4. Accumulate a short reflective note for the next generation's prompts.

Invalid sandbox candidates are replaced by regenerating (Gen0: up to N
extra draws, same spirit as AMFRS "generate 8 extra to replace invalid").
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, List, Optional, Sequence

from crowd_nav.reward_search.llm import (
    LLMClient,
    extract_python_code,
    normalize_to_compute_reward,
)
from crowd_nav.reward_search.prompts import (
    D1_SYSTEM_PROMPT,
    D5_SEED_FUNCTION,
    format_d1_initial,
    format_d2_crossover,
    format_d2_mutation,
)
from crowd_nav.reward_search.sandbox import RewardSandboxError, RewardValidator
from crowd_nav.reward_search.sandbox.runtime import SandboxedReward
from crowd_nav.reward_search.state import RewardFunction


@dataclass
class RewardCandidate:
    """One evolvable reward individual."""

    candidate_id: str
    code: str
    reward_fn: Optional[RewardFunction] = None
    score: Optional[float] = None
    valid: bool = False
    origin: str = "unknown"  # initial | mutation | crossover | random
    parent_ids: tuple = ()
    validation_error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_executable(self) -> bool:
        return self.valid and self.reward_fn is not None

    def as_reward_function(self) -> RewardFunction:
        """Return the validated reward callable (Stage I Score1 entry point)."""
        if self.reward_fn is None:
            raise ValueError(f"Candidate {self.candidate_id} has no reward_fn")
        return self.reward_fn


@dataclass
class StageIConfig:
    population_size: int = 8
    generations: int = 10
    n_crossover: int = 2
    n_mutation: int = 4
    n_random: int = 2
    # Extra LLM draws allowed when replacing invalids (Gen0 + each gen).
    max_invalid_replacements: int = 8
    func_name: str = "compute_reward"
    include_external_knowledge: bool = True


@dataclass
class GenerationRecord:
    generation: int
    population: List[RewardCandidate]
    ranking: List[str]
    reflection: str
    best_score: Optional[float]
    best_id: Optional[str]


class StageIEvolver:
    """
    Stage I: evolve N reward candidates for G1 generations under Score1.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        score_fn: Optional[Callable[..., float]] = None,
        validator: Optional[RewardValidator] = None,
        config: Optional[StageIConfig] = None,
        system_prompt: str = D1_SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm
        if score_fn is None:
            raise ValueError(
                "StageIEvolver requires score_fn=. Pass make_score1_fn(load_stage1_dataset(...)) "
                "for paper Score1, or make_smoke_score_fn() only for --fast / --score1 smoke tests."
            )
        self.score_fn = score_fn
        self.validator = validator or RewardValidator()
        self.config = config or StageIConfig()
        self.system_prompt = system_prompt
        self._counter = 0
        self.reflection: str = ""
        self.history: List[GenerationRecord] = []

        n = self.config.population_size
        if self.config.n_crossover + self.config.n_mutation + self.config.n_random != n:
            raise ValueError(
                "n_crossover + n_mutation + n_random must equal population_size "
                f"({self.config.n_crossover}+{self.config.n_mutation}+"
                f"{self.config.n_random} != {n})"
            )

    # ------------------------------------------------------------------ helpers

    def _next_id(self, prefix: str) -> str:
        cid = f"{prefix}_{self._counter:04d}"
        self._counter += 1
        return cid

    def _completion_to_code(self, raw: str) -> str:
        code = extract_python_code(raw)
        return normalize_to_compute_reward(code, self.config.func_name)

    def _validate(self, code: str) -> tuple[Optional[SandboxedReward], Optional[str]]:
        return self.validator.try_validate(code)

    def _make_candidate(
        self,
        code: str,
        *,
        origin: str,
        parent_ids: tuple = (),
        metadata: Optional[dict] = None,
    ) -> RewardCandidate:
        cid = self._next_id(origin[:3])
        reward_fn, err = self._validate(code)
        return RewardCandidate(
            candidate_id=cid,
            code=code,
            reward_fn=reward_fn,
            valid=reward_fn is not None,
            origin=origin,
            parent_ids=parent_ids,
            validation_error=err,
            metadata=dict(metadata or {}),
        )

    def _llm_code(self, user_prompt: str, *, system: Optional[str] = None) -> str:
        # Providers that support system messages do so internally; we prepend
        # the Appendix D.1 system instruction for scripted / plain complete().
        sys_txt = system if system is not None else self.system_prompt
        full = f"{sys_txt}\n\n{user_prompt}"
        raw = self.llm.complete(full)
        return self._completion_to_code(raw)

    def _fill_valid(
        self,
        needed: int,
        factory: Callable[[], RewardCandidate],
        *,
        max_extra: int,
    ) -> List[RewardCandidate]:
        """Collect ``needed`` valid candidates; regenerate up to max_extra on failure."""
        out: List[RewardCandidate] = []
        attempts = 0
        budget = needed + max_extra
        while len(out) < needed and attempts < budget:
            cand = factory()
            attempts += 1
            if cand.valid:
                out.append(cand)
        if len(out) < needed:
            raise RuntimeError(
                f"Could not obtain {needed} valid candidates after {attempts} attempts "
                f"({len(out)} valid)."
            )
        return out

    # ------------------------------------------------------------------ Gen0

    def initialize_population(self) -> List[RewardCandidate]:
        """
        Gen0: LLM generates N diverse candidates from the D.1 + D.5 seed.

        Invalid programs are replaced by regenerating (up to
        ``max_invalid_replacements`` extra draws).
        """
        cfg = self.config

        def _one() -> RewardCandidate:
            prompt = format_d1_initial(
                func_name=cfg.func_name,
                include_seed=True,
                include_external_knowledge=cfg.include_external_knowledge,
                reflection=self.reflection,
            )
            code = self._llm_code(prompt)
            return self._make_candidate(code, origin="initial")

        return self._fill_valid(
            cfg.population_size,
            _one,
            max_extra=cfg.max_invalid_replacements,
        )

    # ------------------------------------------------------------------ scoring

    def score_population(self, population: Sequence[RewardCandidate]) -> List[RewardCandidate]:
        scored: List[RewardCandidate] = []
        for cand in population:
            if not cand.is_executable:
                scored.append(
                    replace(cand, score=float("-inf"), metadata={**cand.metadata, "unscored": True})
                )
                continue
            value = float(
                self.score_fn(
                    cand.as_reward_function(),
                    candidate_id=cand.candidate_id,
                )
            )
            scored.append(replace(cand, score=value))
        scored.sort(
            key=lambda c: float("-inf") if c.score is None else float(c.score),
            reverse=True,
        )
        return scored

    # ------------------------------------------------------------------ evolution ops

    def _mutate(self, parent: RewardCandidate, reflection: str) -> RewardCandidate:
        weakness = (
            f"Parent {parent.candidate_id} score={parent.score}. "
            f"Weakness focus: improve analytical Score1 relative to elites. "
            f"Global reflection: {reflection}"
        )
        prompt = format_d2_mutation(
            parent.code,
            weakness,
            func_name=self.config.func_name,
        )
        code = self._llm_code(prompt)
        return self._make_candidate(
            code,
            origin="mutation",
            parent_ids=(parent.candidate_id,),
            metadata={"weakness": weakness},
        )

    def _crossover(
        self, parent_a: RewardCandidate, parent_b: RewardCandidate, reflection: str
    ) -> RewardCandidate:
        prompt = format_d2_crossover(
            parent_a.code,
            parent_b.code,
            reflection,
            func_name=self.config.func_name,
        )
        code = self._llm_code(prompt)
        return self._make_candidate(
            code,
            origin="crossover",
            parent_ids=(parent_a.candidate_id, parent_b.candidate_id),
        )

    def _random_restart(self) -> RewardCandidate:
        prompt = format_d1_initial(
            func_name=self.config.func_name,
            include_seed=True,
            include_external_knowledge=self.config.include_external_knowledge,
            reflection=self.reflection,
        )
        code = self._llm_code(prompt)
        return self._make_candidate(code, origin="random")

    def _next_generation(self, ranked: Sequence[RewardCandidate]) -> List[RewardCandidate]:
        cfg = self.config
        if len(ranked) < 2:
            raise ValueError("Need at least 2 ranked candidates to evolve.")

        top_a, top_b = ranked[0], ranked[1]
        # Lower performers: bottom half (paper: mutate using their weakness).
        lower = list(ranked[len(ranked) // 2 :])
        if not lower:
            lower = list(ranked[-1:])

        next_pop: List[RewardCandidate] = []

        # Crossover from top-2.
        for i in range(cfg.n_crossover):
            def _cross(i=i) -> RewardCandidate:
                return self._crossover(top_a, top_b, self.reflection)

            next_pop.extend(
                self._fill_valid(1, _cross, max_extra=cfg.max_invalid_replacements)
            )

        # Mutation on lower performers (cycle if fewer parents than slots).
        for i in range(cfg.n_mutation):
            parent = lower[i % len(lower)]

            def _mut(p=parent) -> RewardCandidate:
                return self._mutate(p, self.reflection)

            next_pop.extend(
                self._fill_valid(1, _mut, max_extra=cfg.max_invalid_replacements)
            )

        # Occasional random restarts (D.1-style).
        for _ in range(cfg.n_random):
            next_pop.extend(
                self._fill_valid(
                    1, self._random_restart, max_extra=cfg.max_invalid_replacements
                )
            )

        assert len(next_pop) == cfg.population_size
        return next_pop

    def _build_reflection(self, ranked: Sequence[RewardCandidate]) -> str:
        """Short reflective note (Section 4.2) for the next generation's prompts."""
        lines = []
        best = ranked[0]
        worst = ranked[-1]
        lines.append(
            f"Best={best.candidate_id} score={best.score}; "
            f"Worst={worst.candidate_id} score={worst.score}."
        )
        # Summarize lower half weaknesses for mutation guidance.
        lower = ranked[len(ranked) // 2 :]
        if lower:
            ids = ", ".join(c.candidate_id for c in lower[:3])
            lines.append(
                f"Lower performers ({ids}) should strengthen goal progress and "
                f"collision/discomfort penalties while keeping dense shaping."
            )
        lines.append(
            "Prefer combining elite safety terms with efficient progress shaping; "
            "avoid near-constant rewards."
        )
        return " ".join(lines)

    # ------------------------------------------------------------------ run

    def run(self) -> List[RewardCandidate]:
        """
        Run Gen0 + G1 evolutionary generations.

        Returns the final ranked population (best first).
        """
        population = self.initialize_population()
        ranked = self.score_population(population)
        self.reflection = self._build_reflection(ranked)
        self.history.append(
            GenerationRecord(
                generation=0,
                population=list(ranked),
                ranking=[c.candidate_id for c in ranked],
                reflection=self.reflection,
                best_score=ranked[0].score,
                best_id=ranked[0].candidate_id,
            )
        )

        for g in range(1, self.config.generations + 1):
            population = self._next_generation(ranked)
            ranked = self.score_population(population)
            self.reflection = self._build_reflection(ranked)
            self.history.append(
                GenerationRecord(
                    generation=g,
                    population=list(ranked),
                    ranking=[c.candidate_id for c in ranked],
                    reflection=self.reflection,
                    best_score=ranked[0].score,
                    best_id=ranked[0].candidate_id,
                )
            )

        return ranked


def seed_reward_code() -> str:
    """Appendix D.5 seed, already adapted to ``compute_reward(state)``."""
    return D5_SEED_FUNCTION

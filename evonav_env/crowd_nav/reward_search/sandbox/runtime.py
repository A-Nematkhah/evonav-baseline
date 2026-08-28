"""
Restricted execution of validated reward source.

Ported from mobile_robot_env.rewards.sandbox.runtime; adapted to RewardState.
"""

from __future__ import annotations

import math
import threading
from typing import Callable, Sequence

from crowd_nav.reward_search.sandbox.config import SandboxConfig
from crowd_nav.reward_search.sandbox.errors import RewardSandboxError
from crowd_nav.reward_search.state import (
    HumanObservable,
    RewardFunction,
    RewardState,
    RobotRewardState,
)

ComputeFn = Callable[[RewardState], float]

_SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "float": float,
    "int": int,
    "bool": bool,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "sum": sum,
    "round": round,
    "pow": pow,
    "sorted": sorted,
    "reversed": reversed,
    "all": all,
    "any": any,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    # Safe read-only type check (no getattr/eval reflection surface).
    "isinstance": isinstance,
    "True": True,
    "False": False,
    "None": None,
}


def require_finite_float(value: object) -> float:
    """Reject bool, non-numeric, NaN, and Inf return values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RewardSandboxError(
            f"compute_reward must return a finite int or float, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise RewardSandboxError(f"compute_reward returned a non-finite value: {number}")
    return number


def compile_compute_reward(code: str, config: SandboxConfig) -> ComputeFn:
    """
    exec() the already-AST-checked source and return compute_reward.

    ``code`` must have passed parse/check_structure/check_interface first.
    """
    namespace = {"__builtins__": dict(_SAFE_BUILTINS)}
    if "math" in config.allowed_modules:
        namespace["math"] = math
    try:
        exec(compile(code, "<reward_candidate>", "exec"), namespace, namespace)
    except RewardSandboxError:
        raise
    except Exception as exc:
        raise RewardSandboxError(f"exec failed: {type(exc).__name__}: {exc}") from exc

    fn = namespace.get(config.required_function_name)
    if not callable(fn):
        raise RewardSandboxError(
            f"{config.required_function_name} was not defined as a callable"
        )
    return fn


def run_with_timeout(fn: Callable[[], object], timeout_seconds: float) -> object:
    """
    Run ``fn()`` on a daemon thread and raise RewardSandboxError on timeout.

    Research timeout only — the worker is not forcibly killed.
    """
    if timeout_seconds <= 0.0:
        raise RewardSandboxError("timeout_seconds must be positive")

    box: dict = {"value": None, "error": None}

    def _target() -> None:
        try:
            box["value"] = fn()
        except Exception as exc:  # noqa: BLE001 - must surface any smoke failure
            box["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    if thread.is_alive():
        raise RewardSandboxError(f"execution exceeded timeout of {timeout_seconds}s")
    if box["error"] is not None:
        raise box["error"]
    return box["value"]


def smoke_test_compute(
    compute_fn: ComputeFn,
    states: Sequence[RewardState],
    config: SandboxConfig,
) -> None:
    """Call compute_fn on each smoke state; require finite floats within timeout."""
    if not states:
        raise RewardSandboxError("smoke test requires at least one RewardState")

    def _run() -> None:
        for state in states:
            try:
                value = compute_fn(state)
            except RewardSandboxError:
                raise
            except Exception as exc:
                raise RewardSandboxError(
                    f"runtime error during smoke test: {type(exc).__name__}: {exc}"
                ) from exc
            require_finite_float(value)

    run_with_timeout(_run, config.timeout_seconds)


def default_smoke_states() -> tuple[RewardState, ...]:
    """Small, deterministic RewardState snapshots for sandbox smoke tests."""

    def make_state(
        *,
        dmin: float,
        collision: bool = False,
        reaching_goal: bool = False,
        timeout: bool = False,
        px: float = 0.0,
        py: float = 0.0,
        gx: float = 4.0,
        gy: float = 0.0,
        human_d: float = 2.0,
    ) -> RewardState:
        action = (0.5, 0.0)
        humans = (
            HumanObservable(px + human_d, py, 0.0, 0.0, 0.3),
        )
        return RewardState(
            robot=RobotRewardState(
                px=px,
                py=py,
                vx=0.5,
                vy=0.0,
                radius=0.3,
                gx=gx,
                gy=gy,
                v_pref=1.0,
            ),
            humans=humans,
            dmin=float(dmin),
            discomfort_dist=0.25,
            collision=collision,
            reaching_goal=reaching_goal,
            timeout=timeout,
            action=action,
            time_step=0.25,
            global_time=1.0,
            time_limit=50.0,
        )

    return (
        make_state(dmin=1.0),
        make_state(dmin=0.1),
        make_state(dmin=0.5, reaching_goal=True, px=3.9, gx=4.0),
        make_state(dmin=-0.05, collision=True, human_d=0.4),
        make_state(dmin=2.0, timeout=True),
        make_state(dmin=0.5, px=1.0, gx=5.0),
    )


class SandboxedReward(RewardFunction):
    """
    RewardFunction wrapper around a sandbox-compiled compute_reward.

    Every call re-checks type and finiteness. Timeout is enforced at
    validation/smoke time, not on every PPO step.
    """

    def __init__(self, compute_fn: ComputeFn, config: SandboxConfig) -> None:
        self._compute_fn = compute_fn
        self._config = config

    def reset(self) -> None:
        return None

    def compute(self, state: RewardState) -> float:
        try:
            value = self._compute_fn(state)
        except RewardSandboxError:
            raise
        except Exception as exc:
            raise RewardSandboxError(
                f"runtime error in sandboxed compute(): {type(exc).__name__}: {exc}"
            ) from exc
        return require_finite_float(value)

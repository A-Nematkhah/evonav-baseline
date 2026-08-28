#!/usr/bin/env python
"""
Human-triggered Stage III smoke test against the real PPO trainer.

Intentionally slow. For CI / pytest use StubPolicyTrainer in
``crowd_nav/reward_search/tests/test_stage3.py``.

Paper K3 = 1e7; this smoke uses a tiny budget. Scale up on a GPU cluster::

    python scripts/run_stage3_smoke.py --train-env-steps 10000000 --eval-episodes 500

Example (from ``evonav_env/`` with the project venv active)::

    python scripts/run_stage3_smoke.py --no-refine
    python scripts/run_stage3_smoke.py --train-env-steps 200 --eval-episodes 1 --h-sweep 5,10
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage III real-trainer smoke")
    parser.add_argument(
        "--train-env-steps",
        type=int,
        default=200,
        help="K3 (smoke default << STAGE3_STEPS / paper 1e7)",
    )
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--horizon-steps", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--population-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=425)
    parser.add_argument("--output-root", type=str, default="trained_models/stage3_smoke")
    parser.add_argument(
        "--h-sweep",
        type=str,
        default="5,10,20",
        help="Comma-separated human counts (Table 6: 5,10,15,20)",
    )
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--no-h-sweep", action="store_true")
    args = parser.parse_args()
    sys.argv = [sys.argv[0], "--no-cuda", "--num-processes", "1"]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import crowd_sim  # noqa: F401
    from crowd_nav.reward_search.evolver import RewardCandidate
    from crowd_nav.reward_search.llm import ScriptedLLMClient
    from crowd_nav.reward_search.prompts import D5_SEED_FUNCTION
    from crowd_nav.reward_search.sandbox import RewardValidator
    from crowd_nav.reward_search.stage3 import (
        STAGE3_PAPER_STEPS,
        RealPolicyTrainer,
        Stage3Config,
        Stage3Runner,
    )

    human_counts = tuple(int(x) for x in args.h_sweep.split(",") if x.strip())

    validator = RewardValidator()
    code = D5_SEED_FUNCTION.strip() + "\n"
    reward_fn, err = validator.try_validate(code)
    if reward_fn is None:
        logging.error("Seed reward failed sandbox: %s", err)
        return 1

    pop = [
        RewardCandidate(
            candidate_id=f"smoke_{i}",
            code=code,
            reward_fn=reward_fn,
            valid=True,
            origin="seed",
        )
        for i in range(args.population_size)
    ]

    cfg = Stage3Config(
        population_size=args.population_size,
        rounds=args.rounds,
        train_env_steps=args.train_env_steps,
        eval_episodes=args.eval_episodes,
        horizon_steps=args.horizon_steps,
        algo="ppo",
        seed=args.seed,
        output_root=args.output_root,
        device="cpu",
        env_name="CrowdSimVarNum-v0",
        human_counts=human_counts,
        train_human_num=20,
    )
    logging.info(
        "Smoke K3=%d (paper=%d); H=%s",
        cfg.train_env_steps,
        STAGE3_PAPER_STEPS,
        human_counts,
    )

    trainer = RealPolicyTrainer()

    if args.no_refine:
        bundle = trainer.train_and_eval(pop[0], round_index=0, config=cfg)
        logging.info("Smoke metrics: %s", bundle.metrics.feedback_text())
        if not args.no_h_sweep:
            report = trainer.evaluate_at_human_counts(
                pop[0], bundle, config=cfg, human_counts=human_counts
            )
            logging.info("H-sweep:\n%s", report.summary_table())
        return 0

    refined = (
        "```python\n"
        "def compute_reward(state):\n"
        "    if state.reaching_goal:\n"
        "        return float(10.0)\n"
        "    if state.collision:\n"
        "        return float(-20.0)\n"
        "    dist = ((state.robot.px - state.robot.gx) ** 2 + "
        "(state.robot.py - state.robot.gy) ** 2) ** 0.5\n"
        "    return float(-0.1 * dist)\n"
        "```\n"
    )
    completions = [refined] * (args.population_size * args.rounds)
    runner = Stage3Runner(
        ScriptedLLMClient(completions),
        trainer,
        validator=validator,
        config=cfg,
    )
    out = runner.run(pop, run_h_sweep=not args.no_h_sweep)
    for rec in runner.history:
        logging.info(
            "round=%d id=%s refined=%s kept_prev=%s metrics=%s",
            rec.round_index,
            rec.candidate_id,
            rec.refined,
            rec.kept_previous,
            rec.metrics.feedback_text(),
        )
    for report in runner.sweep_reports:
        logging.info("H-sweep:\n%s", report.summary_table())
    logging.info("Final population: %s", [c.candidate_id for c in out])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

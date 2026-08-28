# EvoNav Algorithm 1 (this fork)

Extension of CrowdNav++ for reproducing **EvoNav** (arXiv:2605.11859) without AMFRS mechanisms.

Upstream simulator docs: `README.md` in this directory. Architecture audit: `AUDIT.md`.

## Install

```bash
cd evonav_env
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements_pinned.txt
pip install groq                # only if using --llm groq
# PyTorch 1.12.1 + Python-RVO2: see main README.md
```

Use **only** this `.venv` — do not merge with `mobile_robot_env`.

## API keys (Groq)

1. Copy `groq_keys.json.example` → `groq_keys.json`
2. Add keys: `{"keys": ["gsk_...", "..."]}`
3. Or set `GROQ_API_KEY` for a single key (pool disabled)

## Run matrix

| Goal | Command | Hardware | Time |
|------|---------|----------|------|
| Wiring smoke | `python scripts/run_evonav.py --fast` | CPU | seconds |
| Stage I dataset (M=100) | `python scripts/collect_stage1_dataset.py --regime without_random` | CPU | ~tens of min |
| Local validation | `python scripts/run_evonav.py --llm groq --device cuda --regime without_random --stage1-dataset data/stage1_dataset --stage3-train-steps 500000` | GPU + Groq | hours |
| Paper scale | `python scripts/run_evonav_paper_scale.py --device cuda --llm groq` | GPU + Groq | days (K3=1e7 × seeds) |

Defaults (AUDIT.md §8): `without_random`, Stage II/III `predict_method=inferred`, GST `...-seed_1000/sj`.

## Tests

```bash
pytest crowd_nav/reward_search/tests -m "not slow"   # CI default, 74 tests
pytest crowd_nav/reward_search/tests -m slow         # 1 real-env collect test
```

## Baseline checkpoints

Pretrained ORCA/SF/GST under `trained_models/` (see `scripts/report.py`). GST weights under `gst_updated/results/`.

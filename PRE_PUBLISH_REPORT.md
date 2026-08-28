# Pre-publish audit report

**Date:** 2026-08-28  
**Auditor:** automated pass (Cursor agent)  
**Scope requested:** `mobile_robot_env` (AMFRS) + `evonav_env` (CrowdNav++ fork + `reward_search`)  
**Workspace audited:** `D:\Thesis\Implementation\Evonav`

> **Do not make this repository public until Section 1 decisions are resolved and Section 2 untracking is executed if you choose that path.**

---

## Repository layout (important)

| Path | Role | Git |
|------|------|-----|
| `Evonav/` (this tree) | Parent monorepo | `git` root at `D:/Thesis/Implementation/Evonav` |
| `evonav_env/` | CrowdNav++ fork + EvoNav `reward_search` | **Nested** `git` repo (`main`, remote `origin`) |
| `mobile_robot_env/` | AMFRS research code | **Not present** on disk under `D:\Thesis` or `D:\Thesis\Implementation` |

**Implication:** Sections that reference `mobile_robot_env` are **N/A in this workspace**. Publish strategy must decide whether AMFRS lives in a **separate public repo** or is added here before release.

---

## 1. Secrets and sensitive data

### 1.1 Git history scan (`evonav_env` nested repo)

Commands run:
- `git log -p --all -S "gsk_"` → **no matches**
- `git log -p --all` piped through `groq|api_key|token|secret` (case-insensitive) → **no credential-like strings in committed history** (single upstream commit `3907731`)

**Finding:** No evidence that `groq_keys.json` or live Groq keys were ever committed to `evonav_env` history.

### 1.2 Working tree

| File | Tracked? | Ignored? | Notes |
|------|----------|----------|-------|
| `evonav_env/groq_keys.json` | **No** | **Yes** (`.gitignore:15`) | Local file exists with 6 keys — **never commit** |
| `evonav_env/groq_keys.json.example` | Untracked (new) | N/A | Safe template with placeholders |
| `evonav_env/.env` | Not present | Not in `.gitignore` yet | **Fixed:** added `.env*` to `.gitignore` |
| `key_manager.py` | Untracked | N/A | No hardcoded keys; loads from JSON/env only |
| `llm.py` | Untracked | N/A | Reads `GROQ_API_KEY` / key pool only |

### 1.3 History rewrite required?

**No** — based on current `evonav_env` history (1 upstream commit, no key patterns found).

**Caveat:** If you copied keys into commit messages, other branches, or the **parent** `Evonav` repo history outside `evonav_env`, re-scan that repo separately before publishing the parent.

### 1.4 `.gitignore` coverage (before → after this pass)

| Pattern | Before | After fix |
|---------|--------|-----------|
| `groq_keys.json` | Yes | Yes |
| `.env*` | **No** | **Yes** |
| `.venv/` | **No** (only `venv/`) | **Yes** |
| `__pycache__/`, `*.pyc` | Partial | Yes |
| `.pytest_cache/` | **No** | **Yes** |
| `results/`, `data/stage1_dataset/` | **No** | **Yes** (local artifacts) |
| `trained_models/` | Listed but **violated** | Still listed; see §2 |

### 1.5 Tracked files that violate `.gitignore`

`trained_models/` is in `.gitignore` but **27 files remain tracked** (added before ignore rule). Same pattern for `gst_updated/results/` (**28 tracked files**, including GST `epoch_100.pt`).

**Requires your decision (§2):** `git rm --cached` for these paths, or document intentional inclusion.

---

## 2. Repository hygiene

### 2.1 Cache / junk files

- `__pycache__/`, `.pytest_cache/`: not present in working tree (or under `.venv` only).
- `.DS_Store`: none found.
- `.pip_cache/`, `.tmp/`: **untracked** local dirs — now in `.gitignore`.

### 2.2 Large binaries tracked in `evonav_env` git

| Category | Count | Approx size | Recommendation |
|----------|-------|-------------|----------------|
| `trained_models/**/checkpoints/*.pt` | 5 | ~10–11 MB each (~50 MB total) | **Keep for baseline reproduction** OR move to Git LFS / release assets + download script |
| `gst_updated/results/**/epoch_100.pt` | 2 | ~1 MB each | **Keep** (required for `predict_method=inferred` without external download) |
| TensorBoard `events.out.tfevents.*` | many | small–medium | **Untrack** (`git rm --cached`) — not needed for inference |
| `data/stage1_dataset/*.npz` | 0 tracked | ~15 MB local | Correctly **untracked**; document generation in README |
| `results/**` | 0 tracked | local only | Correctly **untracked** |

**`results/ppo_run/*.zip`:** **Not present** in this workspace — N/A.

### 2.3 Uncommitted EvoNav additions (not in `evonav_env` git yet)

Large untracked tree: `crowd_nav/reward_search/`, `scripts/`, `configs/`, `AUDIT.md`, `data/`, `results/`, etc.  
**Before publish:** commit `reward_search` + docs; **do not** commit `groq_keys.json`, `.venv`, `results/`, or `data/stage1_dataset/` unless you intend to ship the 15 MB dataset as a release asset.

### 2.4 Fixes applied automatically

- Expanded `evonav_env/.gitignore` (`.env*`, `.venv/`, caches, `results/`, local data).
- Added `groq_keys.json.example`.
- Added `.github/workflows/evonav-ci.yml` (fast tests only).

### 2.5 Requires explicit sign-off (destructive)

```bash
# From evonav_env/ — removes from git index only, keeps files on disk
git rm -r --cached trained_models/   # if you move baselines to LFS/assets
git rm -r --cached gst_updated/results/**/aoe gst_updated/results/**/foe gst_updated/results/**/loss gst_updated/results/**/events.out.*
```

---

## 3. Dependency hygiene

### 3.1 `evonav_env`

| File | Purpose |
|------|---------|
| `requirements.txt` | Upstream CrowdNav++ (loose pins) |
| `requirements_pinned.txt` | **Verified** Windows/py3.10 stack (`gym==0.15.7`, `numpy==1.23.5`, `scipy==1.10.1` explicit) |
| `.venv/` | Isolated env — **do not merge** with AMFRS |

**Verified in `.venv`:** `gym 0.15.7` (matches pinned file; avoids PassiveEnvChecker drift).

**Gaps found:**
| Package | Used by | In `requirements_pinned.txt`? |
|---------|---------|-------------------------------|
| `groq` | `--llm groq` | **Was missing** → **added** in this pass |
| `pytest` | test suite | **Was missing** → **added** |
| `scipy` | Stage I scoring | **Yes** (`1.10.1`) — no longer transitive-only |

**Not pinned:** PyTorch (install per upstream README). Documented in `requirements_pinned.txt` header.

### 3.2 `mobile_robot_env`

**N/A** — project not on disk. Cannot cross-check `pyproject.toml` or scipy transitive use.

### 3.3 Isolation between projects

Documented in `evonav_env/AUDIT.md` and `requirements_pinned.txt` header: dedicated `.venv` only.

---

## 4. Dead code, duplication, stubs

### 4.1 `mobile_robot_env` pyflakes / `_json_safe` / `n_llm_mutations`

**N/A** — source tree absent.

### 4.2 `evonav_env/crowd_nav/reward_search`

**pyflakes:** not installed in `.venv`; not run. Manual review:

| Item | Status |
|------|--------|
| `_json_safe` duplication | **N/A** — not present in `reward_search` |
| `n_llm_mutations` counter bug | **N/A** — AMFRS-only; not in evonav |
| `make_smoke_score_fn` | **Isolated** — only via `--fast` / `score1_mode=smoke` in `pipeline.py` |
| `NotImplementedError` in production path | **None** — only ABC stubs (`LLMClient`, `PolicyTrainer`) and opt-in smoke paths |
| `score1_for_dataset` | **Implemented** (not stub) |

### 4.3 Stubs reachable from non-test paths

| Stub | Gate |
|------|------|
| `Stage2StubTrainer` / `Stage3StubTrainer` | `stage2_use_stub` / `stage3_use_stub` or `--fast` |
| `make_smoke_score_fn` | `score1_mode=smoke` or `--fast` |
| `SeedVariantLLMClient` | `--llm seed` |
| `paper_scale` dry-run | `--dry-run-stubs` + no `CI` env |

---

## 5. Documentation accuracy

### 5.1 Numeric claims (current, verified 2026-08-28)

| Claim | Verified value |
|-------|----------------|
| `reward_search` tests | **75** collected in **12** files under `crowd_nav/reward_search/tests/` |
| Fast CI subset (`-m "not slow"`) | **74 passed**, 1 deselected |
| Slow tests | **1** (`test_collect_stage1_dataset_smoke`) |
| `gym` version | **0.15.7** in active `.venv` |
| Default regime | `without_random` (`regime.py`, `AUDIT.md` §8.1) |
| GST path (without_random) | `gst_updated/results/...-seed_1000/sj` |
| Stage II/III `predict_method` | `inferred` (choice **a**, `AUDIT.md` §8.2) |

### 5.2 Stale / missing docs

| Document | Issue |
|----------|-------|
| `evonav_env/README.md` | Still **upstream CrowdNav++ only** — no EvoNav Algorithm 1 / `reward_search` |
| `docs/architecture.md` (AMFRS) | **N/A** — missing with `mobile_robot_env` |
| `AUDIT.md` §8 Decisions | **Present** and consistent with code (regime, obs-space a, GST assert) |
| Top-level `Evonav/README.md` | **Missing** — **added** `README.md` in this pass |

### 5.3 Fixes applied

- `README.md` (repo root): relationship AMFRS vs evonav_env, install, test, run commands.
- `evonav_env/README_EVONAV.md`: EvoNav-specific quick start.

---

## 6. Licensing and attribution

| Item | Status |
|------|--------|
| `evonav_env/LICENSE` | **MIT**, Shuijing Liu 2023 — **unmodified** |
| `evonav_env/NOTICE.md` | **Correct** ICRA 2023 attribution |
| Top-level AMFRS LICENSE | **N/A** — no `mobile_robot_env` / parent LICENSE in tree |
| License conflict | **Low risk** if AMFRS ships separately; if monorepo, add top-level LICENSE + per-subdir notices |
| Citations | **Added** BibTeX block to root `README.md` (EvoNav arXiv:2605.11859 + Liu ICRA 2023) |

---

## 7. Test suite health

### 7.1 `evonav_env` — full run

```
pytest crowd_nav/reward_search/tests --collect-only  → 75 tests
pytest -m "not slow"                                 → 74 passed
pytest -m slow                                       → 1 passed
```

### 7.2 Slow / excluded from default CI

| Test | Reason |
|------|--------|
| `test_collect_stage1_dataset_smoke` | Real simulator rollouts (`@pytest.mark.slow`) |

### 7.3 Hidden environment assumptions

| Risk | Finding |
|------|---------|
| GPU in unit tests | **None** in fast suite |
| Network in unit tests | **None** in fast suite (Groq tests use mocks / inline keys) |
| Windows paths | Slow collect test uses repo-relative paths; passed on Windows |
| gym version | Pinned `0.15.7` in `requirements_pinned.txt` |

### 7.4 CI added

`.github/workflows/evonav-ci.yml` — runs `pytest -m "not slow"` on `push`/`pull_request` to `evonav_env/`.

**`mobile_robot_env` CI:** **N/A** — not in workspace.

---

## 8. Structural check

### 8.1 Cross-imports `mobile_robot_env` ↔ `evonav_env`

**No Python imports** of `mobile_robot_env` in `evonav_env` (only comments in `llm.py`). **Pass.**

### 8.2 Sandbox parity with AMFRS

Cannot diff against `mobile_robot_env` (absent). **evonav_env** sandbox includes:
- `isinstance` in `_SAFE_BUILTINS` (`runtime.py`)
- AST allowlist (`sandbox/validator.py`)
- Timeout smoke tests (`test_sandbox_rejects_infinite_loop_timeout`)

**Regression test:** `test_sandbox_isinstance_is_available`, `test_reward_adapter_governs_crowdsimpredrealgst`.

**AMFRS parity:** **Unverified** — requires `mobile_robot_env` checkout.

---

## 9. Publish checklist (your decisions)

- [ ] **Monorepo vs split:** Publish `evonav_env` alone (nested repo) or flatten into one public repo?
- [ ] **Untrack** `trained_models/` / TensorBoard junk via `git rm --cached`?
- [ ] **Ship baselines:** keep ~50 MB `.pt` in git, Git LFS, or external download?
- [ ] **Stage I dataset:** document `collect_stage1_dataset.py` (do not commit 15 MB `npz` unless intended)
- [ ] **Commit** untracked `reward_search/` tree before tagging release
- [ ] **Add** `mobile_robot_env` to public repo or link separate AMFRS repository
- [ ] **Rotate Groq keys** if any were ever pasted in chat/logs (local `groq_keys.json` never committed)

---

## 10. Summary

| Section | Result |
|---------|--------|
| 1 Secrets | **Pass** (no keys in git history; local `groq_keys.json` untracked) |
| 2 Hygiene | **Needs decision** on tracked `trained_models/` + tfevents |
| 3 Dependencies | **Fixed** groq/pytest pins; gym 0.15.7 verified |
| 4 Dead code | **Pass** for evonav; AMFRS **N/A** |
| 5 Docs | **Partially fixed**; upstream README still CrowdNav-only |
| 6 License | **Pass** for fork; AMFRS **N/A** |
| 7 Tests | **75/75 pass**; CI workflow added |
| 8 Structure | **Pass** (no cross-import); sandbox AMFRS diff **N/A** |

**Automatic fixes in this pass:** `.gitignore`, `requirements_pinned.txt`, CI workflow, root `README.md`, `README_EVONAV.md`, this report.

**Not done (by design):** `git push`, history rewrite, `git rm --cached` on large artifacts, making repo public.

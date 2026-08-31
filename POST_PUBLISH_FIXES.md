# Post-publish fixes

**Repo:** [evonav-baseline](https://github.com/A-Nematkhah/evonav-baseline) (public)  
**Report date:** 2026-08-28 (updated 2026-08-31)

---

## Issue 1 — `baselines_openai` broken gitlink (CRITICAL)

**Found:** After public release, `baselines_openai/` was committed as a git submodule
gitlink (`mode 160000`, commit `ea25b9e8…`) with **no** `.gitmodules` entry. Fresh
clones received an empty directory, causing:

```
ModuleNotFoundError: No module named 'baselines'
```

from `evonav_env/rl/networks/{dummy_vec_env,envs,shmem_vec_env}.py` and
`evonav_env/rl/vec_env/{vec_normalize,logger}.py` (via `baselines.common.*`).

**Fix (commit `634ae35`):**
- Removed nested `.git` from `baselines_openai/`
- Replaced gitlink with full vendored OpenAI Baselines source (upstream commit
  `ea25b9e8`, MIT — see `baselines_openai/LICENSE` and `baselines_openai/NOTICE.md`)
- Documented install: `pip install tensorflow` then `pip install -e ../baselines_openai`
  (TensorFlow required because upstream `setup.py` asserts it at install time)
- Added `tensorflow==2.11.0` to `evonav_env/requirements_pinned.txt` (follow-up commit)

---

## Issue 2 — Vendored `Python-RVO2/` removed

**Found:** `Python-RVO2/` was vendored at repo root with Windows-only binaries
(`rvo2.cp310-win_amd64.pyd`, `RVO.lib`), build artifacts, and an unrelated
`rvo2-navmesh-test.blend`. The simulator imports the **`rvo2` pip package**
(`crowd_sim/envs/crowd_sim.py`, `crowd_nav/policy/orca.py`), not this vendored tree.

**Fix (commit `634ae35`):**
- `git rm -r Python-RVO2/`
- Install instructions unchanged in `evonav_env/README.md` step 4:
  [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2)

---

## Issue 3 — README wording

**Fix:** Root `README.md` no longer says “before making this repository public”;
now points to `PRE_PUBLISH_REPORT.md` (audit history) and this file.

---

## Verification (fresh clone, 2026-08-31)

Clean temp clone from local repo (`git clone` → new directory, not reused checkout):

| Step | Result |
|------|--------|
| `baselines_openai/setup.py` present after clone | **PASS** (was empty gitlink before fix) |
| `Python-RVO2/` absent after clone | **PASS** |
| `pip install -r requirements_pinned.txt` + `pip install -e ../baselines_openai` + `pip install torch==1.12.1` | **PASS** (with `tensorflow` in pinned requirements) |
| Baselines import smoke: `from rl.networks.envs import make_vec_envs` | **PASS** |
| Fast pipeline: `python scripts/run_evonav.py --fast` | **PASS** (completed Stage I–III, artifacts written) |

**Note:** `train.py` additionally requires PyTorch, Python-RVO2, and GST checkpoints
per `evonav_env/README.md`; the baselines import path above is the specific regression
fixed by flattening `baselines_openai`.

---

## Commits

| Hash | Description |
|------|-------------|
| `634ae35` | Flatten `baselines_openai`; remove `Python-RVO2/` |
| `2c2ba74` | Pin TensorFlow; doc install order; update this report |

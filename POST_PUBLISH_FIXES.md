# Post-publish fixes

**Repo:** EvoNav faithful-replication baseline (public)  
**Report date:** 2026-08-28

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

**Fix (commit pending verification):**
- Removed nested `.git` from `baselines_openai/`
- Replaced gitlink with full vendored OpenAI Baselines source (upstream commit
  `ea25b9e8`, MIT — see `baselines_openai/LICENSE` and `baselines_openai/NOTICE.md`)
- Documented install step: `pip install -e ../baselines_openai` from `evonav_env/`

**Verification:** See § Verification below.

---

## Issue 2 — Vendored `Python-RVO2/` removed

**Found:** `Python-RVO2/` was vendored at repo root with Windows-only binaries
(`rvo2.cp310-win_amd64.pyd`, `RVO.lib`), build artifacts, and an unrelated
`rvo2-navmesh-test.blend`. The simulator imports the **`rvo2` pip package**
(`crowd_sim/envs/crowd_sim.py`, `crowd_nav/policy/orca.py`), not this vendored tree.

**Fix:**
- `git rm -r Python-RVO2/`
- Install instructions unchanged in `evonav_env/README.md` step 4:
  [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2)

---

## Issue 3 — README wording

**Fix:** Root `README.md` no longer says “before making this repository public”;
now points to `PRE_PUBLISH_REPORT.md` (audit history) and this file.

---

## Verification (fresh clone)

_Placeholder — updated after clean-clone test completes._

| Step | Command | Result |
|------|---------|--------|
| Clean clone | `git clone <repo> /tmp/evonav_verify` | |
| Baselines imports | `pip install -e ../baselines_openai` + import smoke | |
| Fast pipeline | `python scripts/run_evonav.py --fast` | |

---

## Commit

_Fill in commit hash after push._

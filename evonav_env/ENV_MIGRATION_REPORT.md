# Environment Migration Report

## Detection

- Python on PATH: `3.10.11`
- NVIDIA driver: `610.62`
- CUDA UMD reported by `nvidia-smi`: `13.3`
- Verified torch: `2.11.0+cu128`
- `torch.cuda.is_available()`: `True`
- GPU: NVIDIA GeForce RTX 3050, 4GB
- Recorded wheel index: `https://download.pytorch.org/whl/cu128`

## Work completed

- Removed the legacy split-Python launcher.
- Added the GPU torch pin and CUDA-index note to `requirements_pinned.txt`.
- Added Stage I global-best tracking, regression warnings, and one-slot elitism.
- Ensured the global-best candidate is present in the Stage II input population.
- Added `test_global_best_survives_a_regressive_generation`.
- Installed and verified `rvo2` from a local MSVC-built wheel; the upstream
  Windows build needed Release configuration and its `src/Release` library path.
- Updated trusted project checkpoint loads for Torch 2.6+ with
  `weights_only=False`.

## Verification

- Ollama client tests: `5 passed`.
- Stage I evolver tests, including the new regression test: `8 passed`.
- Full `crowd_nav/reward_search/tests -m "not slow"` suite: `84 passed`.
- Changed files compile successfully with Python 3.10.
- Real GPU smoke completed successfully with `--device cuda`: Stage I, Stage II,
  Stage III, and H-sweep all completed. GST checkpoints reported `device:
  cuda:0`; final candidate was `cro_0002_v3`.
- The smoke used `stage1_population=2`, one round per stage, 8 train steps,
  and 2 evaluation episodes. It was intentionally not a paper-scale run.
- No paper-scale or M=100 run was attempted.
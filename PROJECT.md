# PROJECT.md

Project-specific context that supplements `AGENTS.md`.

`AGENTS.md` is fixed across projects.
`PROJECT.md` accumulates what is specific to this one.

Update this file when a new project-specific constraint, environment
detail, or convention is discovered during work.

Do not duplicate:
- verified architecture → `docs/`;
- current work → `plan.md`;
- unresolved problems → `issues.md`.

## Template

- Initialized from (source, version, or commit): explicit-agent 0.1.0 (7ff7fe0b562c)
- Last synced to (source, version, or commit): explicit-agent 0.1.0 (7ff7fe0b562c)
- Last applied migration id (see the template's `migrations.json`): 0

## Purpose

A fork of FlashRT for extending inference support to ImageWAM. Two
people split the work by target hardware: Orin is handled elsewhere;
this fork covers Jetson AGX Thor (sm_110).

## Environment

- Platform / hardware: local development machine, 8GB VRAM (`nvidia-smi`
  reported) / 23GB RAM, WSL2 (`Linux 5.15.167.4-microsoft-standard-WSL2+`);
  target deployment hardware is Jetson AGX Thor (sm_110), accessed
  separately, not on this machine.
- Shared with (other users or projects on the same account or machine):
  none known.
- Constraints imposed by the environment: ImageWAM's FLUX.2-4B variant
  needs roughly 18GB of weights alone (FLUX.2-4B backbone + Qwen3-4B
  text encoder + VAE, bf16) / ~8.9GB for just the transformer weights
  this project's own pipeline actually loads (fp16, no VAE/text
  encoder) — either way, more than the local machine's 8GB VRAM.
  **Refined 2026-09-15 (OPT-001)**: this is a SOFT limit, not a hard
  one, on this specific WSL2 environment — `torch.cuda.max_memory_allocated()`
  reached ~9.86GB during a real end-to-end construct/capture/infer run
  without raising `OutOfMemoryError`, because this environment's CUDA
  driver pages beyond the reported dedicated VRAM into host RAM rather
  than failing. It technically completes, but at a catastrophic cost
  (~13.6s per `infer()` call, vs. the same math's own per-layer-bench
  timing which sums to well under 1s) — for any correctness check that
  doesn't care about latency, the full real-weight pipeline CAN now run
  here; for anything timing-sensitive, still use the per-layer-isolated
  benchmarks (`imagewam_thor_bench.py`, one layer type resident at a
  time) or hand off to Thor.
- Local FlashRT build: usable for kernel-correctness testing on this
  machine's Ada (sm_89) GPU, not for Thor-specific performance
  (correctness only — new kernels here are not FP8-specific, so this
  is not the same class of Ada/Thor portability problem the sibling
  pi0.5_ggml project's ISSUE-018 found for native FP8 PTX, but Thor
  timing still needs the real target either way).
  Build steps: `git clone --depth 1 --branch v4.4.2
  https://github.com/NVIDIA/cutlass.git third_party/cutlass`, `uv pip
  install -e ".[torch]"` into a venv with a working torch+CUDA (the
  `third_party/openpi/.venv` venv from the sibling pi0.5 project
  worked), then configure with the slim recipe below +
  `cmake --build build -j4 --target flash_rt_kernels`. The system
  pybind11 (2.9.1, `/usr/include`) is too old for this venv's Python
  3.11.13 (`PyFrameObject` is an incomplete type in newer CPython
  headers; pybind11 2.9.x's error-formatting code needs it complete) —
  install pybind11>=2.13 into the same venv and pass
  `-Dpybind11_DIR=$(python3 -c "import pybind11;
  print(pybind11.get_cmake_dir())")` to cmake to force it to use the
  venv's newer copy instead of falling back to the system one.
- Slim build (recommended default for this project): this fork only
  ever needs `flash_rt_kernels` — never `flash_rt_fa2`'s separate
  vendored-kernel build, and never the Motus/Qwen3.6/NVFP4-specific
  kernel groups meant for other models on other hardware. `CMakeLists.txt`
  forces `ENABLE_FA2` on unconditionally for `GPU_ARCH=89`
  (`flash_rt_kernels` hard-links `$<TARGET_OBJECTS:fa2_vendor_obj>`, so
  FA2 itself cannot be skipped on this arch), but its instantiation
  matrix can be shrunk, and `FLASHRT_SLIM_BUILD` drops the unrelated
  model-specific kernel groups entirely. Configure with:
  ```
  cmake -B build -S . -DGPU_ARCH=89 \
    -DFA2_ARCH_NATIVE_ONLY=ON -DFA2_HDIMS="64" -DFA2_DTYPES="fp16" \
    -DFLASHRT_SLIM_BUILD=ON \
    -DPython3_EXECUTABLE=<venv python> -Dpybind11_DIR=<venv pybind11 cmake dir>
  ```
  Confirmed this still builds `flash_rt_kernels` correctly and produces
  a `.so` that passes `tests/test_imagewam_mot_joint_kernel.py`
  unchanged (`cosine=1.000000, rel_l2=0.000441`) — the slim flags only
  drop unrelated kernel groups and shrink FA2's instantiation count,
  they never touch kernel math.
- INT4/INT8 exploration build: add `-DENABLE_SM80_INT8_CUTLASS=ON
  -DFLASHRT_ENABLE_CHAMELEON=ON` to the slim-build command above to
  additionally build the SM80-family CUTLASS INT8/INT4 rowwise GEMM
  kernels (`cutlass_int4_rowwise_fp16out`, `cutlass_int8_rowwise_fp16out`
  etc.) — these are gated only by these two flags, not by `GPU_ARCH`,
  and confirmed to build and run correctly on this machine's Ada
  (sm_89) despite being written for Jetson Orin SM87 (see
  `opportunities.md` OPT-007 and `plan.md`'s GEMM-only comparison).
  Not needed for this project's own pipeline (`pipeline_thor.py` uses
  none of these); kept as a documented option for future INT4
  exploration, not part of the default recommended build above.
  The current local `build/` directory has these two flags ON (from
  the INT4 exploration) — additive only, so it is still a valid
  superset of the default recommended build (`tests/test_imagewam_mot_joint_kernel.py`
  still passes unchanged); rebuild without them only if the extra
  compile time/kernels are unwanted.
- Memory is a real constraint on this machine (23GB total RAM,
  already had 635MB in swap before any build started) — a full
  `cmake --build build -j$(nproc)` (nproc=20) OOM-killed partway
  through the unrelated `fa2_vendor_obj`/`flash_rt_fa2` target (large
  FA2 attention kernel instantiations) even though the actually-needed
  `flash_rt_kernels` target (everything this plan's own kernel work
  lives in) finished and linked successfully first. Do not re-run a
  full `-j$(nproc)` build without a real reason — a much lower `-j`
  (e.g. 4) is safer, and this plan's own pipeline/frontend files
  (Phases 3-5) are pure Python, needing no C++/CUDA rebuild at all
  once `flash_rt_kernels` itself is built once. Clean up scratch build
  logs and any redundant build directories promptly; disk itself has
  headroom (729GB free) but is not a reason to be careless about it.
- **Precision-testing division of labor (user's explicit standing
  instruction, given this machine's limited memory/VRAM)**: this local
  machine (Ada sm_89) does INT4/INT8 (SM80 CUTLASS) inference speed
  work only — that's the precision tier this hardware can actually
  execute. FP4 (NVFP4) testing belongs on Thor, not here: FP4 needs
  Blackwell hardware this machine does not have at all (`nvfp4_sim`
  emulates its numerics for accuracy work). FP8 cuBLASLt runs on sm_89
  and sm_90 through the TN layout (`issues.md` ISSUE-001, resolved);
  FP8 speed still belongs on Thor. Do not ask Thor to test INT4/INT8
  (SM80) either — that path is a confirmed, measured dead end there
  (~8.6x slower than FP16, see `opportunities.md` OPT-007) precisely
  because it doesn't use Thor's own native tensor cores.

### Shared H100 server (current dev machine since 2026-09-17)

- Hardware: 8x NVIDIA H100 NVL (sm_90, 94GB each), 755GB RAM, driver
  580.159.03 (CUDA 13.0). Another user's long-running training job keeps
  about 40GB of each GPU in use at about 100% utilization. Latency
  measured on this machine is contaminated by that job and is not a
  performance number. Use this machine for correctness checks and Thor
  for speed.
- Environment script: `/home/user1/workspace/jingwu/imagewam_env.sh`,
  outside the repository. It sets the CUDA 12.6 toolkit (`nvcc`), the
  `.venv` (Python 3.11.13, torch 2.14.0+cu126, transformers 4.56.1,
  diffusers 0.39.0, pybind11 3.1.0), `IMAGEWAM_SRC`, `FLUX2_SRC`,
  `CKPT_PATH`, `FLUX2_MODEL_PATH`, `FLUX2_AE_MODEL_PATH`/`AE_MODEL_PATH`,
  `QWEN3_MODEL_SPEC`, and `DATA_ROOT`.
- Assets:
  - `/home/user1/workspace/jingwu/models/imagewam_flux2_4b_libero/`: HF
    `yuyangalin/ImageWAM-FLUX.2-4B-LIBERO`, with `model.pt`,
    `config.yaml`, and `dataset_stats.json`.
  - `/home/user1/workspace/jingwu/models/flux2_klein_4b/`:
    `flux-2-klein-base-4b.safetensors` and `ae.safetensors` (from the
    gated `FLUX.2-dev`).
  - Qwen3-4B from the HF cache.
  - ImageWAM upstream at `/home/user1/workspace/jingwu/ImageWAM`.
  - LIBERO (`yuanty/LIBERO-fastwam`, all four suites, extracted) at
    `/home/user1/workspace/jingwu/data/libero_mujoco3.3.2`.
- Build: `cmake -B build -S . -G Ninja -DGPU_ARCH=90
  -DFLASHRT_SLIM_BUILD=ON -DPython3_EXECUTABLE=<.venv python>
  -Dpybind11_DIR=<.venv pybind11 cmake dir>`, then `cmake --build build
  -j32 --target flash_rt_kernels`. This takes about 45 seconds. On
  sm_90, FA2, SM100 CUTLASS, and NVFP4 are all disabled. ImageWAM's
  attention uses the cuBLAS attention kernels, so the FP16 pipeline
  still runs.
- Thor (sm_110) compile check: a CUDA 13.0.88 toolkit is installed at
  `/home/user1/workspace/jingwu/cuda13` (conda, `nvidia` and
  `conda-forge` channels). `/home/user1/workspace/jingwu/sm110_check.sh
  <src_dir> <name>` mirrors a source tree to
  `/home/user1/workspace/jingwu/sm110_mirror/<name>` and builds
  `flash_rt_kernels` and `flash_rt_fp4` there with `GPU_ARCH=110`. The
  mirror keeps the source tree's own sm_90 `.so` from being overwritten.
  The first build takes about 10-15 minutes at `-j24`; later builds are
  incremental. A passing build shows only that the code compiles and
  links. Correctness and speed still have to be checked on Thor.
- Baseline on this machine: `pytest tests/test_imagewam_*.py` gave 68
  passed and 6 skipped before ISSUE-001 was resolved. FP8 cuBLASLt now
  runs here through the TN layout; the remaining skips are FA4, NVFP4,
  SM100 CUTLASS FP8, and the FP8 NN-vs-TN comparison (needs a GPU that
  supports both layouts).
- NVFP4 numerics can be checked here with `precision="nvfp4_sim"`
  (`flash_rt/models/imagewam/nvfp4_sim.py`, bit-exact to the real
  quantizer; `tests/test_imagewam_nvfp4_sim.py` JIT-compiles
  `csrc/quantize/quantize_fp4_dynamic.cu` for sm_90 to prove it).
- Real activation-calibration files (`docs/imagewam_calibration.md`)
  live outside the repository, under
  `/home/user1/workspace/jingwu/artifacts/calibration/`.
- End-to-end check against the official model:
  `benchmarks/imagewam_e2e_official_compare.py`. On fp16 over 20
  LIBERO frames, the median action cosine is 0.9984 when both sides use
  the same noise. The numbers are recorded in `issues.md` ISSUE-002.
- The real-checkpoint tests read `CKPT_PATH`, `AE_MODEL_PATH`, and
  `IMAGEWAM_SRC`. When these are unset, they fall back to the original
  `/home/ljw/...` paths.

## Credentials

Services that require a scoped identity in this project (git remotes,
package registries, model hubs, cloud APIs).

| Service | Scope | Source | Notes |
|---|---|---|---|
| git remote `origin` | `luo-jingw/FlashRT` (personal fork) | global git identity | not `flashrt-project/FlashRT` upstream; no upstream PR is planned for this work |

Do not assume a global or default identity applies.
See `AGENTS.md` → Credentials and Environment Isolation.

Do not record token values or other secrets here.

## Project-Specific Constraints

- This fork does not send pull requests to `flashrt-project/FlashRT`
  upstream. `CONTRIBUTING.md`'s upstream PR workflow does not apply to
  work done here.
- The `AGENTS.md` this project ships with is the generic
  `explicit-agent` template. FlashRT's own prior `AGENTS.md` (deleted
  from this repository when the structures layer moved to the sibling
  `FlashRT-Structures` repository) is not adopted here: it is scoped to
  that now-external structures/kernel-hub layer, and several of its
  rules (real calibration data only, kernels sourced only from an
  external kernel hub, additive-only changes to existing code) are
  incompatible with this project's current stage, which requires
  writing new kernels from scratch and testing with randomly
  initialized weights before any real checkpoint or calibration data
  is involved.
- **Superseded 2026-09-15**: the note that used to stand here ("current
  work uses randomly initialized weights... explicitly excludes FP8
  quantization, calibration") is no longer accurate. Real checkpoint
  loading is DONE (`opportunities.md` OPT-001, `plan.md`'s own
  "OPT-001" plan, Phases 1-3 completed, verified end-to-end on this
  machine). FP8/NVFP4 quantized GEMM (dynamic, static-scale, and
  static-scale+CUTLASS) is DONE with real Thor numbers (`opportunities.md`
  OPT-004 steps 5-6). What is genuinely still deferred: the FULL house
  calibration mechanism (`docs/calibration.md`'s multi-sample/
  percentile calibration against real OBSERVATION data, not just a
  disposable random tensor) — `_calibrate_fp8`'s own current
  implementation freezes a scale from random noise, not a real
  activation distribution; upgrading this needs real per-model
  observation data whose tensor shapes actually match this project's
  own `img_raw`/`context`/`action_latent` conventions, still an open
  question as of this note (see `opportunities.md` OPT-004's own
  calibration entry for the exact gap and what was tried).
- **Confirmed end goal (2026-09-14, explicit user direction): this
  targets real Thor deployment, not an indefinitely-scoped structural
  dry run.** `pipeline_thor.py`/`_imagewam_thor_spec.py`/
  `flash_rt/frontends/torch/imagewam_thor.py` should converge toward
  the SAME standard implementation framework FlashRT's other real
  Thor deployments already use (Pi0.5's own `pipeline_thor.py` +
  frontend is the reference pattern to align with: pointer-owned
  steady-state buffers, one CUDA Graph capture/replay, the
  `AttentionBackendBase` protocol, real checkpoint loading) — not kept
  as a permanent parallel/throwaway "correctness only" path. The
  `real_*.py`/`pipeline_real.py` modules under
  `flash_rt/models/imagewam/` (opportunities.md OPT-002) are the
  verified real-math building blocks for that convergence, not a
  separate destination in themselves.
- **Real checkpoint testing happens ONLY on Thor, never on this
  machine — this has been stated multiple times, stop re-asking.**
  **Correction (2026-09-15): the claim "this dev machine has no
  checkpoint... and never will" is FALSE, discovered while planning
  OPT-001.** The actual checkpoint FILES are present locally
  (`/home/ljw/projects/pi0.5/models/flux2_klein_4b/` — base FLUX.2
  safetensors + AE, 7.7GB; `/home/ljw/projects/pi0.5/models/imagewam_flux2_4b_libero/model.pt`
  — the real ImageWAM-LIBERO fine-tune, 9GB, plus `config.yaml`), and
  the `imagewam` Python package IS importable here
  (`PYTHONPATH=/home/ljw/projects/pi0.5/tmp/ImageWAM/src`). What is
  STILL true and still blocks a real forward pass on this machine: (a)
  `flux2` (the actual FLUX.2 model-definition code from
  `black-forest-labs/flux2`) is NOT cloned anywhere on this machine —
  `imagewam`'s own `ImageWAM.from_flux2_klein_pretrained` hard-depends
  on it; (b) the full model needs ~18-23GB resident, this machine has
  8GB VRAM (`nvidia-smi` confirmed). So: checkpoint files can be
  inspected directly (state_dict keys/shapes via `torch.load(...,
  map_location='cpu', mmap=True)` — confirmed working, no `flux2`/`imagewam`
  needed, ~9GB fits in this machine's 20GB free RAM without moving
  anything to GPU) but a real forward pass still needs Thor. Any
  real-checkpoint-dependent verification is written here (dev machine)
  as a script/test with clear instructions, then handed to the user to
  run on Thor and report back — see `benchmarks/imagewam_real_checkpoint_validation.py`
  for the established pattern. Confirmed real environment details from
  that first real run (2026-09-14, cosine=0.999927 backbone /
  0.999963 ActionDiT): checkpoint file is `model.pt` (not
  `checkpoint.pt`), sibling config is `config.yaml` (not
  `train_config.yaml`), `action_dim=7` (LIBERO 7-DoF), `imagewam`
  installed via `PYTHONPATH=<repo>/src` (NOT `pip install -e .` --
  that downgrades Thor's `torch==2.9.1+cu130` to `2.7.1` via the
  package's own pinned deps), `FLUX2_SRC` points at `flux2` cloned at
  pinned commit `50fe5162777813d869182b139e83b10743caef15`,
  `model.load_checkpoint` reported `missing_keys=0 unexpected_keys=0`
  (the LoRA-merge branch flagged as a possible complication was a
  non-issue for this release).
- **Correction (2026-09-15): `black-forest-labs/flux2` IS clonable
  from THIS dev machine too** — `git clone https://github.com/black-forest-labs/flux2.git`
  succeeds directly and pins to the exact commit above without ever
  having had a local checkout before. Cloned at
  `FlashRT/third_party/flux2` (gitignored, matching `third_party/cutlass`'s
  own convention). This unlocked a real VAE encoder locally — see
  `opportunities.md`'s own "Real VAE encoder + text-context wiring"
  entry (OPT-008) for what that found (the real
  `flux2.autoencoder.AutoEncoder` class is NOT the same as
  `diffusers.AutoencoderKLFlux2`, despite `model_index.json` naming the
  latter) and `plan.md`'s matching plan for what got built.
- **New isolated venv, `FlashRT/.venv`** (gitignored, separate from the
  shared `third_party/openpi/.venv` every prior session used): needed
  because that shared venv's `lerobot==0.4.4` pins `diffusers<0.36.0`,
  incompatible with what the real VAE work needed. Built via `uv venv
  --python 3.11.13 .venv` + `uv pip install -e ".[torch]" diffusers
  transformers pybind11 einops av opencv-python-headless
  huggingface_hub` (torch resolved to `2.14.0+cu130`, pybind11 to
  `3.1.0` — both confirmed working here and ABI-compatible with the
  existing `flash_rt_kernels.so`, same Python 3.11.13). Rebuild
  `flash_rt_kernels` against this venv with the same slim-build cmake
  recipe below, pointing `-DPython3_EXECUTABLE`/`-Dpybind11_DIR` at
  `.venv`'s own copies — verified the rebuilt `.so` still works from
  BOTH venvs afterward (same output path, same CUDA arch/build flags).
  Use THIS venv (not the shared one) for any work touching
  `diffusers`/`transformers`/the real VAE.

## Onboarding

- ImageWAM upstream source, read-only reference:
  `https://github.com/yuyangalin/ImageWAM`, cloned at
  `/home/ljw/projects/pi0.5/tmp/ImageWAM`.
- FlashRT's own new-model workflow:
  `docs/adding_new_model.md`, `flash_rt/frontends/torch/_template/`.
- Closest existing structural precedent in this codebase:
  `flash_rt/models/cosmos3_edge/` (a diffusion-transformer denoise
  engine already on Thor: encode-once, then a denoise loop captured as
  one CUDA Graph and replayed — the same shape ImageWAM's own
  backbone-prefill + action-expert-denoise split needs).
- See `plan.md` for the current implementation plan.

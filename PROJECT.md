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

- Platform / hardware: local development machine, 8GB VRAM / 32GB RAM
  (too small to hold ImageWAM's own weights — see below); target
  deployment hardware is Jetson AGX Thor (sm_110), accessed separately,
  not on this machine.
- Shared with (other users or projects on the same account or machine):
  none known.
- Constraints imposed by the environment: ImageWAM's FLUX.2-4B variant
  needs roughly 18GB of weights alone (FLUX.2-4B backbone + Qwen3-4B
  text encoder + VAE, bf16) before activation memory — this does not
  fit the local machine's 8GB VRAM. Local work is limited to
  component-level testing that does not require the full model
  resident on one GPU (e.g. the Qwen3-4B text-context precompute path
  on CPU, or the VAE/patch-tokenization path alone); end-to-end
  verification of the full pipeline happens on separate, larger
  hardware.
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
- Current work (see `plan.md`) uses randomly initialized weights, not
  a downloaded checkpoint, and explicitly excludes FP8 quantization,
  calibration, and accuracy validation. These are deferred, not
  abandoned — they require real weights and calibration data this
  stage does not use.

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

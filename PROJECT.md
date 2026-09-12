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

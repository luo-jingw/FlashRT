# ImageWAM backbone: last single-stream layer, K/V-only computation

## Scope

Whether the real FLUX.2-4B backbone's LAST layer (the 25th, a
single-stream block) does work that nothing downstream ever reads, and
if so, a design for computing only the part that is actually read. This
is a design note plus a standalone proof-of-concept; nothing in
`flash_rt/models/imagewam/pipeline_thor.py` is changed.

Evidence is drawn directly from `flash_rt/models/imagewam/pipeline_thor.py`
(`_single_stream_layer`, `imagewam_prefill`, `imagewam_denoise_step`,
`imagewam_denoise_loop`) and `flash_rt/hardware/thor/attn_backend.py`
(`ImageWAMAttnBackend`), read in full.

## Does the ActionDiT read every backbone layer's K/V, or only some?

Every layer's K/V is read, but through a **layer-index-matched
bijection**, not an all-pairs "every ActionDiT layer reads every
backbone layer" relationship.

`ImageWAMAttnBackend` gives both attention sites ("backbone" and "mot")
ONE shared, physical per-layer K/V cache (`attn_backend.py`'s own
`ImageWAMAttnBackend` docstring: "Both sites share ONE physical
per-layer K/V buffer pair... K/V here are a single shared set per
position"). `get_slot_ptrs(site, layer_idx)` indexes into that SAME
`self._per_layer_kv` list regardless of which site is asked
(`attn_backend.py:868-878`).

`imagewam_prefill` writes layer `L`'s K/V at `site_layer_idx=L` for
`L` in `0..24` (`pipeline_thor.py:614-624`: double-stream layers use
`layer_idx` directly as `0..4`; single-stream layers use
`num_double + i` for `i` in `0..19`, i.e. `5..24`).

`imagewam_denoise_step` reads that SAME per-layer cache through the
"mot" site, at the SAME `site_layer_idx` numbering, once per denoise
step (`pipeline_thor.py:849-855`):

```python
for layer_idx in range(num_double):                      # 0..4
    _action_double_layer(..., layer_idx, layer_idx, ...)  # site_layer_idx = layer_idx
for i in range(dims["action_num_layers_single"]):         # 0..19
    _action_single_layer(..., i, num_double + i, ...)     # site_layer_idx = 5..24
```

So ActionDiT layer `L` (double `0..4`, then single `5..24` by the same
numbering) reads **backbone layer `L`'s own K/V slot only** — each
`attn.run("mot", site_layer_idx, ...)` call passes exactly one
`site_layer_idx` and `get_slot_ptrs` returns exactly one `(K, V)`
pointer pair for it (`attn_backend.py:909`, `934-1066`). There is no
mask or loop inside `run()` that reads more than one layer's K/V per
call. `kv_seq=dims["total"]` is the SEQUENCE length (rows) attended
over within that one layer's cache, not a count of layers.

**Consequence for backbone layer 24 (the last one, `site_layer_idx=24`,
which is `weight_layer_idx=19`, the last of the 20 single-stream
layers):** its K/V slot IS read — by ActionDiT's own last single-stream
layer (`i=19` → `site_layer_idx=24`), once per denoise step
(`dims["num_denoise_steps"]` times, `pipeline_thor.py:893-898`). It is
not dead. **This means the original "if every layer's K/V is read for
every ActionDiT layer, stop" condition from this task's own
instructions does not trigger** — the relationship is a bijection, not
an all-pairs read, and no backbone layer's K/V goes completely unread.

## What about backbone layer 24's OTHER computation, beyond K/V?

This is a separate question from K/V liveness, and the answer differs.

`bufs["backbone_hidden"]` (the persistent residual, `combined` inside
`_single_stream_layer`/`_double_stream_layer`) is:
- read by `imagewam_prefill`'s own loop, layer by layer, each layer
  reading the PREVIOUS layer's write (`pipeline_thor.py:492`,
  `608-624`);
- **never read anywhere in `imagewam_denoise_step` or
  `imagewam_denoise_loop`.** Neither function references
  `bufs["backbone_hidden"]` at all (`pipeline_thor.py:782-899` — grep
  confirms no occurrence). The denoise loop reads the backbone's output
  exclusively through the per-layer K/V cache slots
  (`attn.get_slot_ptrs("mot", site_layer_idx)`), which
  `ImageWAMAttnBackend.__init__` allocates as separate, distinct
  buffers from `bufs["backbone_hidden"]` — the pipeline passes
  `backbone_slots`/`mot_slots` with their own `K`/`V` pointers at
  attention-backend construction time (see the frontend, e.g.
  `imagewam_thor.py`), entirely independent memory from
  `bufs["backbone_hidden"]`. They are not aliased views of it.

So: the LAST single-stream layer's own write into `bufs["backbone_hidden"]`
(`_single_stream_layer`'s final `fvk.gate_res_bf16res(from_attn, gate_t.data_ptr(), combined, ...)`,
`pipeline_thor.py:534`) has no reader. Nothing after `imagewam_prefill`
returns ever looks at `bufs["backbone_hidden"]` again — it is dead
after the last layer runs. (For every OTHER backbone layer, `L<24`,
this same residual write IS read — by layer `L+1`'s own
`ada_layer_norm` at the top of its block. Only the very last layer's
write is unread.)

Tracing what depends on that dead write, inside `_single_stream_layer`
(`pipeline_thor.py:471-534`, `merge_qkv_mlp=True` path — the real
deployment default, see "Real dims" below):

```
combined = bufs["backbone_hidden"]
ada_layer_norm_bf16in_fp16out(combined, ...) -> modded          # <- feeds linear1; NEEDED (Q/K/V/MLP all read modded)
linear1.weight(modded, linear1_out, ...)                        # ONE GEMM, N = 3*hidden + 2*mlp_hidden
  columns [0,hidden)         -> Q_O    (copy + rms_norm + rope)  # DEAD: only feeds this layer's own self-attn below
  columns [hidden,2*hidden)  -> K_cache (copy + rms_norm + rope) # LIVE: read by mot's layer-matched ActionDiT layer
  columns [2*hidden,3*hidden)-> V_cache (copy)                   # LIVE: read by mot's layer-matched ActionDiT layer
  columns [3*hidden, end)    -> silu_glu_merged_fp16 -> mlp_gated# DEAD: only feeds mlp_down below
attn.run("backbone", 24, q_seq=a0)     # self-attn using Q_O/K_cache/V_cache -> overwrites Q_O
  -> attn_out_proj.weight(Q_O, from_attn, ...)                   # DEAD: only feeds the residual write below
mlp_down.weight(mlp_gated, from_mlp, ...)                        # DEAD: only feeds the residual write below
_add_inplace(from_attn, from_mlp, ...)                           # DEAD
gate_res_bf16res(from_attn, gate_t, combined, ...)               # DEAD: writes backbone_hidden, never read again
```

Everything marked DEAD produces a value that only feeds another DEAD
value, transitively, ending at the unread residual write. Everything
marked LIVE (or feeding a LIVE value) must still run.

## Design: K/V-only last-block computation

Since only K and V (not Q, not the MLP) are needed, and V needs no
RMSNorm/RoPE (confirmed: `_single_stream_layer` calls `rms_norm_fp16`/
`rope_apply_fp16_perhead` on `Q_O` and `K_cache` only, never on
`V_cache` — `pipeline_thor.py:522-525`), the last single-stream layer's
K/V-producing work reduces to:

1. `ada_layer_norm_bf16in_fp16out(combined, scale, shift, modded, a0, hidden, eps, stream)`
   — unchanged, the GEMM input needs it regardless of which output
   columns are read.
2. A GEMM against a **column-sliced `linear1.weight`**: only columns
   `[hidden, 3*hidden)` (K and V; Q is `[0,hidden)`, MLP gate/up is
   `[3*hidden, end)` — order confirmed directly from
   `_single_stream_layer`'s own `_copy_slice` calls). The slice is
   sliced ONCE, at weight-load time, as a plain contiguous column-slice
   copy (`w[:, hidden:3*hidden].contiguous()`) — the SAME pattern
   `quant_linear.CutlassFp16SwiGluMlp`/`Nvfp4SwiGluMlp` already use in
   this codebase to split a merged gate/up weight into two separate
   contiguous tensors at construction. **No new CUDA kernel is
   needed** — the existing `GemmRunner.fp16_nn` (or an `Fp8`/`Nvfp4`
   equivalent) runs unchanged, just at `N = 2*hidden` instead of
   `N = 3*hidden + 2*mlp_hidden`.
3. `_copy_slice` the GEMM output's two halves into `K_cache`/`V_cache`
   (same as today).
4. `rms_norm_fp16` + `rope_apply_fp16_perhead` on `K_cache` only.

Dropped entirely: the Q column of `linear1` and its copy/RMSNorm/RoPE,
the backbone's own self-attention call for this layer
(`attn.run("backbone", 24, ...)`), `attn_out_proj.weight`, the MLP gate/up
columns of `linear1` and `silu_glu_merged_fp16`, `mlp_down.weight`,
`_add_inplace`, and `gate_res_bf16res`.

This is implemented, as a standalone (not wired into `pipeline_thor.py`)
Python-level prototype, in `tests/test_imagewam_last_block_kv_only.py`:
`make_kv_only_linear` (the one-time weight slice) and
`last_single_stream_layer_kv_only` (the reduced per-call sequence).

## Correctness

`tests/test_imagewam_last_block_kv_only.py` runs the real, unmodified
`_single_stream_layer` (imported read-only from `pipeline_thor.py`, not
reimplemented) as ground truth, and the sliced prototype above, against
the SAME weights and the SAME input residual bytes, then compares the
two paths' `K_cache`/`V_cache` outputs. `merge_qkv_mlp=True` is used
throughout (the real deployment default for every precision except
`fp16_cutlass`, `flash_rt/frontends/torch/imagewam_thor.py:163`).

Tested at `hidden=3072, HD=128, NH=24, mlp_hidden=9216` throughout
(real FLUX.2-4B/ImageWAM width — `benchmarks/imagewam_thor_graph_bench.py`'s
`REAL_DIMS`, see "Real dims" below), at three row counts:
`a0 ∈ {9, 130, 905}` (905 is the real production value).

**Result: bit-exact at the real production shape.** At `a0=905`,
`K_cache`/`V_cache` from the sliced computation are literally
`torch.equal` to the full block's own output (`max_abs_diff=0`). At the
smaller synthetic row counts (`a0=9`, `a0=130`), the two paths are
numerically identical to full FP16 precision (cosine similarity
`1.0` to 6+ decimal places) but not literally bit-identical: `gemm.fp16_nn`
(`csrc/gemm/gemm_runner.cu`) caches a cuBLASLt algorithm keyed on the
GEMM's own `(M,N,K)` shape (`get_or_create_cached(FP16_NN, M, N, K)`);
the sliced GEMM's `N=2*hidden` differs from the full block's own
`linear1` `N=3*hidden+2*mlp_hidden`, so cuBLASLt is free to pick a
different, equally valid reduction order for the small-`M` shapes —
observed as a median 1-ULP / 90th-percentile 4-ULP difference on ~40%
of elements, with `max_abs_diff=0.0078` (2 ULP at the affected values'
own magnitude). This is ordinary GPU GEMM floating-point
non-associativity between two different real kernel launches, not a
slicing bug — and it does not occur at the real production shape, which
is the one that matters for deployment.

## Real dims and computed savings

`flash_rt/models/imagewam/libero_dims.py`, named in this task's
originating instructions, **does not exist anywhere in this
repository** (checked: no file of that name, and a repository-wide
search for the specific row counts named there, `{25, 417, 905}`, finds
nothing beyond `a0=905` itself). The only place ImageWAM's real,
checkpoint-confirmed production dims are recorded is
`benchmarks/imagewam_thor_graph_bench.py`'s `REAL_DIMS` (its own
docstring: "Real dims (opportunities.md, confirmed 2026-09-15/16/17)"),
used throughout this design and its test instead:

```
hidden=3072, HD=128, NH=24, mlp_hidden=9216, a0=905
```

(`num_layers_double=5, num_layers_single=20` — 25 layers total, matching
"25" from the task's own row-count list; `25` is a layer count here,
not a row count, which may be the source of that number appearing in
the original instructions.)

### Removed weight bytes (last layer only, NVFP4, 0.5625 bytes/param)

| GEMM | shape (K,N) | full bytes | removed |
|---|---|---:|---:|
| `linear1` (Q + MLP gate/up columns) | (3072, 21504) | 37,158,912 (35.44 MiB) | all |
| `linear1` (K,V columns, kept) | (3072, 6144) | 10,616,832 (10.12 MiB) | — |
| `attn_out_proj.weight` | (3072, 3072) | 5,308,416 (5.06 MiB) | all |
| `mlp_down.weight` | (9216, 3072) | 15,925,248 (15.19 MiB) | all |
| **total removed** | | | **58,392,576 bytes (55.69 MiB)** |

(`linear1`'s full weight is 47,775,744 bytes / 45.56 MiB;
kept + removed columns reproduce that exactly, 10,616,832 + 37,158,912
= 47,775,744.)

### Removed FLOPs (last layer only, one `imagewam_prefill` call, `a0=905`)

GEMM FLOPs counted as `2*M*N*K`; self-attention counted as
`4*NH*a0^2*HD` (QK^T + softmax·V, both `O(NH*a0^2*HD)` multiply-adds).

| Removed op | FLOPs |
|---|---:|
| `linear1` Q+MLP columns (`M=905, N=21504, K=3072`) | 119.57 GFLOP |
| `attn_out_proj.weight` (`M=905, N=3072, K=3072`) | 17.08 GFLOP |
| `mlp_down.weight` (`M=905, N=3072, K=9216`) | 51.24 GFLOP |
| backbone's own self-attention at layer 24 | ~10.06 GFLOP |
| **total removed** | **~197.96 GFLOP** |

For scale: the full `linear1` GEMM alone (all columns, this one layer)
is 153.73 GFLOP; the removed Q+MLP columns are 77.8% of it
(21504/27648 columns) — the K,V slice that remains is the other 22.2%
(42.16 GFLOP).

These savings are realized ONCE per `imagewam_prefill` call (prefill
runs once per new observation, not once per denoise step) — they do
not multiply by `num_denoise_steps`.

### Kernel launches removed (from `_single_stream_layer`'s own
`merge_qkv_mlp=True` code path, `pipeline_thor.py:471-534`)

| Call site | Disposition |
|---|---|
| `fvk.ada_layer_norm_bf16in_fp16out` | kept (needed for the GEMM input) |
| `key("linear1.weight")(...)` (the GEMM) | kept, but **N shrinks from 27648 to 6144** |
| `_copy_slice(Q_O, ...)` | **removed** |
| `_copy_slice(K_cache, ...)` | kept |
| `_copy_slice(V_cache, ...)` | kept |
| `fvk.silu_glu_merged_fp16(...)` | **removed** |
| `fvk.rms_norm_fp16(Q_O, ...)` | **removed** |
| `fvk.rms_norm_fp16(K_cache, ...)` | kept |
| `fvk.rope_apply_fp16_perhead(Q_O, ...)` | **removed** |
| `fvk.rope_apply_fp16_perhead(K_cache, ...)` | kept |
| `attn.run("backbone", 24, ...)` (self-attention) | **removed** |
| `key("attn_out_proj.weight")(...)` (GEMM) | **removed** |
| `key("mlp_down.weight")(...)` (GEMM) | **removed** |
| `_add_inplace(...)` | **removed** |
| `fvk.gate_res_bf16res(...)` | **removed** |

9 of 15 call sites disappear entirely (`_copy_slice(Q)`, `rms_norm(Q)`,
`rope(Q)`, `silu_glu_merged_fp16`, `attn.run` self-attention,
`attn_out_proj` GEMM, `mlp_down` GEMM, `_add_inplace`,
`gate_res_bf16res`); the remaining `linear1` GEMM shrinks in N (and
therefore FLOPs/bytes) but does not disappear; `ada_layer_norm`,
`_copy_slice(K)`/`_copy_slice(V)`, `rms_norm(K)`, `rope(K)` are
unchanged.

## Applicability

This applies ONLY to the single, very last backbone layer
(`site_layer_idx=24`, the 20th and final single-stream layer). For
every other backbone layer (`0..23`), `bufs["backbone_hidden"]` IS read
again — by the next layer's own `ada_layer_norm` — so the full
computation (Q, self-attention, MLP, residual write) remains necessary
there. This is a one-layer, one-time (per `imagewam_prefill` call)
optimization, not a pattern that recurs across the backbone's depth.

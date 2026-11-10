# VoLoAgent Full-loop master workload replayed with the ImageWAM executor (RTX 5090, FlashRT)

Question: with the master calls and call frequencies of VoLoAgent's Full loop, what does the FlashRT planner
(Qwen3.8-27B master, NVFP4) cost per episode, can it keep up, and what does co-location do to the ImageWAM executor
when the executor is queried at the harness's rate (one action chunk of 8 steps at 15 Hz, every 533 ms)?

## Workload

| item | value |
|---|---|
| harness configuration | `--mode subgoal --failure-monitor vlm --recovery-mode replan_tools --check-interval 80 --use-front-camera`; system prompts read from VoLoAgent `ad3e3c4c` (`extract_prompts.py`, `prompts.json`) |
| episode | LIBERO-10 episodes 0, 43, 188; 800 env steps at 15 Hz (53.3 s sim time); frames from the dataset videos, letterboxed to the harness's camera sizes, JPEG q90 (`volo_calls.py`) |
| calls per episode | decompose at step 0 (3 images); monitor every 80 steps (10 calls; 6 images: before / now x front 864x480, exterior 1280x720, wrist 224x224 or 1280x720); replan after a `replan` action or a `next` on the last subgoal (2 front images; one call in the six recordings) |
| call contents | recorded once with the planner (`record_calls.py`, `calls/ep<E>_wrist<W>.jsonl`); greedy, thinking off, `max_tokens` 512 (replan 1024) |
| planner | the LoRA-merged Qwen3.8-27B master, FlashRT NVFP4 (`Qwen36VlTorchFrontendRtx`), FP8 KV cache, long-context route, MTP head of the official Qwen3.8-27B-FP8 release, `max_seq` 8 448, 4 608 image-token slots |
| executor | ImageWAM LIBERO fp8, slimmed after `set_prompt` (6.30 GB), cuBLAS 13.6.1.10; one query = `stage_inputs` + graph replay, 24.5 ms alone (`executor_imagewam.py`) |

Recorded calls (wrist 224 / 1280x720):

| call | prompt tokens | image tokens | output tokens | call at EOS stop (alone) | call at `max_tokens` |
|---|---|---|---|---|---|
| decompose | 2 126 / 2 942 | 1 349 / 2 165 | 21-40 | 0.65 / 0.71 s | 16-17 s |
| monitor | 4 079 / 5 711 | 2 698 / 4 330 | 38-87 (median 55-59) | 1.02 / 1.27 s | 11.2-11.4 s |
| replan | 1 427 | 810 | 12 | 0.43 s | 32 s |

Sources of the workload parameters:

- Monitor every 80 env steps: VoLoAgent's `--check-interval` default (its help text: 80 steps are about 10 pi0.5
  chunks), kept by the agent harness's `scripts/volo_tool/eval_agent.sh` (`CHECK_INTERVAL` default 80). With
  `--env libero` VoLoAgent replaces the default by 40 steps (`vlm_orchestrator/cli.py`, LIBERO replans every 5 sim
  steps). The VoLoAgent paper gives about 0.2 Hz (5 s).
- 15 Hz and 8 steps per executor query: the harness's robolab evaluation client (`scripts/volo_tool/robolab_run.py`:
  15 Hz joint position control, `CHUNK = 8`, VoLoAgent's pi0.5 chunk length). The 5.33 s and 533 ms are derived from
  them. The executor here, ImageWAM, is evaluated on LIBERO at 20 Hz with 16 predicted and 12 executed steps by its own
  evaluation script, which this cadence does not follow.
- Call contents, `max_tokens`, the synchronous protocol and the event calls: VoLoAgent's source at `ad3e3c4c` for the
  harness's configuration. Image sizes: the harness sends full-resolution images; the wrist size is not determinable
  from its code.
- Set here: LIBERO-10 frames in place of robolab scenes, 800 env steps per episode, replies recorded by the planner
  itself, the planner models, the ImageWAM executor and its configuration.

`Qwen36VlTorchFrontendRtx.generate` has no EOS stop: it decodes `max_new_tokens` and cuts the reply afterwards. A
replayed call therefore decodes the recorded reply's length through EOS (`output_tokens + 1`); the last column is
the cost of the same call with `max_tokens` as the decode budget.

## Replay modes (`replay_bench.py`)

| mode | schedule |
|---|---|
| `sync` | the harness's protocol: at a step with a master call, the call runs to completion, then the executor query; one thread, no wall-clock schedule |
| `realtime` | wall-clock schedule: executor released every 533 ms (101 queries per episode), each master call due at step / 15 s in a planner thread; sharing `prio` (executor stream priority -1), `green50` (planner on 88 SMs, executor on the other 82), `coop` (`colocation_qwen27b_5090/coop_gate.py`: planner segments start only in the executor's idle window; needs the eager prefill with 256-token chunks) |

Executor latency = finish - release. Decision latency = reply time - the call's due time. A call is cold when the
planner captures a CUDA graph during it (a prompt length or decode position not seen before in the process).

## Results (`cells/`, three episodes per cell; `replay_table.py`)

| mode | share | wrist | wall s / ep | master s / ep | monitor p50 (warm) ms | monitor max ms | decision latency max (steps) | exec p50 / p99 / max ms | > 66.7 ms | > 533 ms | same reply |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sync | - | 224 | 17.1 | 14.5 | 1 157 (1 020) | 3 338 | - | 24.6 / 24.8 / 25.4 | 0% | 0% | 33/33 |
| sync | - | 720 | 19.0 | 16.4 | 1 345 (1 168) | 3 666 | - | 24.5 / 24.8 / 24.9 | 0% | 0% | 33/33 |
| realtime | prio | 224 | 54.4 | 14.9 | 1 190 (1 035) | 3 434 | 51.7 | 25.4 / 35.2 / 47.6 | 0% | 0% | 33/33 |
| realtime | green50 | 224 | 54.7 | 18.3 | 1 479 (1 321) | 4 037 | 60.8 | 33.4 / 74.7 / 101.5 | 1.3% | 0% | 33/33 |
| realtime | coop | 224 | 54.6 | 17.0 | 1 409 (1 226) | 3 554 | 53.5 | 25.5 / 26.7 / 29.3 | 0% | 0% | 4/33 |
| realtime | prio | 720 | 54.3 | 16.8 | 1 361 (1 202) | 3 788 | 57.1 | 25.5 / 49.5 / 98.2 | 0.7% | 0% | 33/33 |
| realtime | green50 | 720 | 54.6 | 21.4 | 1 765 (1 586) | 4 589 | 69.1 | 33.4 / 175.0 / 303.6 | 1.7% | 0% | 33/33 |
| realtime | coop | 720 | 54.4 | 19.7 | 1 734 (1 461) | 3 537 | 53.4 | 25.3 / 26.7 / 30.8 | 0% | 0% | 4/33 |

- Model time per episode in `sync`: 17.1 / 19.0 s against 53.3 s of sim time, of which the master is 14.5 / 16.4 s
  (the robot waits 27-31% of the episode) and the executor 2.5 s.
- The planner keeps up in `realtime`: every call returns before the next one is due (80 steps = 5.33 s apart); the
  0.1-1.9 s by which an episode's wall time exceeds the sim time is the step-800 monitor finishing after the last
  executor query.
- No executor query exceeded 533 ms in any cell. Over one env step (66.7 ms): `prio` 0-0.7%, `green50` 1.3-1.7%
  (half the SMs: p50 33.4 ms), `coop` 0%.
- Master cost by sharing (monitor p50): `prio` +1-3% over `sync`, `coop` +22-29%, `green50` +28-31%.
- Cold calls: 23-26 of 29-30 monitor calls per cell capture graphs, median 5-6 graphs, monitor p50 1.22 / 1.40 s
  against 1.02 / 1.17 s warm (`sync`); the first monitor of a process captures 56-59 graphs and takes 3.3-3.8 s.
  Decision latency maxima (52-69 steps) are these first calls.
- `coop` replies differ from the recorded ones because of its 256-token prefill chunk, not co-location: the planner
  alone with the eager prefill and 256-token chunks changes 8 of 11 replies of episode 0 (one monitor decision goes
  from `in_progress / continue` to `complete / next`), the eager prefill with the default chunk changes none
  (`prefill_mode_check.py`, `smoke/prefill_pf0_c{256,def}.json`; ISSUE-110). The replay decodes the recorded length,
  so a changed reply costs the same decode; a longer changed reply is cut.

## Executor instruction switch (`cells/instruction_switch.json`)

Executor alone with the Qwen3-4B text encoder resident (bf16, 7.6 GB checkpoint), `set_prompt` for the subgoals of
the recorded decompositions:

| | ms |
|---|---|
| first call in the process | 751 |
| new instruction, context length not seen before (encode + capture of the graph for that length) | 240-244 |
| new instruction, context length seen before / same instruction again | 30-31 |

Allocated memory: 22.55 GB with the text encoder and the FP16 source weights, 6.30 GB after `slim_executor`. With
the planner resident (2.6-3.0 GB free in every replay cell), the 7.6 GB encoder does not fit next to the slimmed
executor; a new subgoal reaches the executor only through another encoding path.

## MTP speculative decode on these calls (`mtp_tail_model.py`, `mtp_tail_model_table.py`, `cells/mtp_tail_{master,base}.json`)

The merged master checkpoint has no MTP tensors; the drafter is the official Qwen3.8-27B-FP8 MTP head. The RTX long
route fills the drafter's K/V with the prompt's last rows after the prefill (the MTP prompt tail; the frontend's table
gives 512 rows for the monitor calls and 2 048 for decompose); without it the drafter sees only generated tokens.

Grid per main model (the merged master, and the official Qwen3.8-27B converted with the same script and drafted by
the same MTP file): MTP prompt tail (off / the frontend's table / the whole prompt) x tail fc kernel (the frontend's
small-M kernel / cuBLASLt) x K (1, 3, 5, 7, the frontend's table), on the 22 recorded calls of ep0 and ep43 (wrist 224)
and four synthetic prompts. Each model first decodes each call once with `max_tokens` and then decodes to its own EOS
in every cell (official model: monitor median 72 tokens, decompose 39; master 52 / 24). One warm-up and one timed call
per cell.

Recorded calls, K = 3, tail fc on cuBLASLt; tokens per main-model forward / decode ms per call / speedup over the
single-token estimate:

| main model | MTP prompt tail | monitor x 20 (4 076 tokens) | decompose x 2 (2 127 tokens) |
|---|---|---|---|
| master | off | 2.13 / 623 / 1.74x | 1.52 / 374 / 1.22x |
| master | table (512 / 2 048 rows) | 2.60 / 511 / 2.12x | 2.61 / 218 / 2.09x |
| master | whole prompt | 2.85 / 468 / 2.31x | 2.47 / 230 / 1.98x |
| official | off | 2.46 / 726 / 2.00x | 2.30 / 398 / 1.85x |
| official | table | 2.72 / 656 / 2.21x | 2.92 / 314 / 2.34x |
| official | whole prompt | 3.05 / 587 / 2.47x | 3.04 / 302 / 2.43x |

The full grid (all K, both fc kernels, synthetic prompts) is `mtp_tail_table.md`.

- The single-token estimate is (decoded tokens - 1) x the Kv = 1 main-model forward (graph replay, `cells/mtp_speedup.json`);
  FlashRT has no graph-captured single-token decode on this route, and the estimate leaves out per-step host work,
  so the speedups are lower bounds.
- For the master, the prompt tail is the larger factor: monitor 2.13 -> 2.60 / 2.85 tokens per forward, decompose
  1.52 -> 2.61. At the same tail setting the official model is 5-15 % higher on monitor and 12-51 % on decompose (two
  calls); its replies differ from the master's, so this does not isolate the head.
- With the tail, K = 3 is the fastest K for both call kinds of the master (monitor 511 / 468 ms, decompose 218 ms);
  K = 5 / 7 accept more but cost more per cycle; the table's K = 6 for decompose gives 241 ms.
- Tail fc: the shared small-M kernel costs 21 / 84 ms at 512 / 2 048 rows (whole tail prefill with cuBLASLt 0.4 /
  1.2 ms); tokens per forward differ by at most 2 % between the two kernels (monitor 2.66 / 2.60), and tail + decode
  is shorter with cuBLASLt (decompose 302 -> 219 ms). The RTX frontend's tail fc is cuBLASLt with the heuristic's
  first algorithm (`max_algos=1`; the fc alone 0.24 / 0.97 ms at 512 / 2 048 rows, `smoke/mtp_tail_fc_default.log`);
  on the official model it reproduces the grid's cuBLASLt cells (tokens per forward 2.72 monitor, 2.92 / 3.45
  decompose at K = 3 / 6; prefill 311 / 172 ms against 338 / 258 ms on the small-M kernel;
  `thor/planner_calls_vl_rtx.json`).
- The tail's NVFP4 k/v projections use the activation scratch of `self.max_seq` rows (2 048 in long-context mode), in
  chunks of that many rows; an unchunked tail above 2 048 rows wrote past the scratch and lowered every later call's
  acceptance in the process (`smoke/mtp_state_bisect.log` against `smoke/mtp_state_bisect_fixed.log`).
- Outputs do not change with the tail or K: 516 of 520 master cells and 520 of 520 official-model cells equal their
  reference; the four others are the 16 315-token synthetic prompt at K = 1 (ISSUE-109).
- The synthetic prompts' replies reach EOS after 12-35 tokens (master), too few for acceptance; they serve the
  forward cost by prompt length:

| prompt tokens | main-model forward, Kv = 1 / 2 / 4 / 6 / 8 (ms) | MTP draft chain, K = 1 / 3 / 5 / 7 (ms) |
|---|---|---|
| 602 | 19.35 / 19.13 / 20.63 / 22.20 / 23.13 | 0.96 / 2.90 / 4.83 / 6.77 |
| 4 023 | 19.89 / 19.70 / 21.18 / 22.73 / 23.68 | 0.97 / 2.91 / 4.85 / 6.78 |
| 8 110 | 19.70 / 19.46 / 20.98 / 22.71 / 23.65 | 0.97 / 2.91 / 4.86 / 6.79 |
| 16 315 | 22.39 / 22.21 / 23.68 / 25.23 / 26.19 | 0.97 / 2.92 / 4.87 / 6.82 |

A cycle costs the Kv = K + 1 forward plus the K-step chain (K = 3 at 602 tokens: 20.6 + 2.9 = 23.5 ms, measured
23.6 ms per cycle).

## Planner call shapes on the Thor frontend (`thor/planner_call_shapes.py`)

LM-only time of the official Qwen3.8-27B NVFP4 with its MTP head at the harness call shapes (decompose 2 126 prompt
tokens / 25 new, monitor 4 072 / 53, monitor 5 711 / 53), prompt = the first N tokens of the repository's
documentation; the same code runs on the Thor (`Qwen36TorchFrontendThor`, Thor row H3a) and here
(`thor/planner_call_shapes_rtx.{json,log}`). ViT time and image content are not part of it. The RTX run predates the
cuBLASLt tail fc (its prefill includes the small-M kernel's 84 / 21 ms at 2 048 / 512 tail rows).

| shape (prompt / new tokens) | RTX 5090 prefill / decode ms, ms per cycle, tokens per forward | Jetson Thor (round `1104_h3`) | prefill + decode, Thor |
|---|---|---|---|
| decompose (2 126 / 25) | 256 / 353, 27.2, 1.85 | 3 708 / 2 194, 157, 1.71 | 5.90 s |
| monitor, wrist 224 (4 072 / 53) | 334 / 487, 24.4, 2.60 | 4 408 / 2 998, 136, 2.36 | 7.41 s |
| monitor, wrist 1280x720 (5 711 / 53) | 475 / 520, 24.7, 2.48 | 6 106 / 2 456, 136, 2.89 | 8.56 s |

On the Thor the language model alone exceeds the 5.33 s monitor interval at every shape (M-tile tail fc); prefill is
12.9-14.5x the 5090's. Decode ms per cycle on the Thor depends on the part of the session: the same script gave
156.9 / 99.8 / 93.3 ms at the start and 99.0 / 88.4 / 87.7 ms at the end of round `1104_h3b` (round `1104_h3`: 156.7 /
136.3 / 136.4 ms), so the Thor per-cycle values here are not a denominator (issues.md ISSUE-112). 88 ms per cycle at
K = 3 (about 17.5 GB of weights per cycle) is about 199 GB/s, 84 % of the Thor's measured copy bandwidth (236 GB/s);
136 ms is 55 % (derived). `thor/decode_cycle_drift.py` generates the monitor shape back to back for 300 s in one
process with the clock readings the machine exposes; on the 5090 a cycle takes 88 ms in the first generation (graph
captures) and 24.2-24.5 ms in the other 371 (`thor/decode_cycle_drift_rtx_table.txt`). On the Thor (round
`1104_h3c`, the first GPU process of the round) it takes 136.4-137.0 ms from 28 s to 153 s, 95.7-99.7 ms from 172 s
to 218 s and 87.4-88.0 ms from 232 s on, with isolated 120-249 ms generations; prefill moves 1.4 % between the slow
and the steady part (4 033 / 3 977 ms). CPU frequencies and the readable devfreq clocks (GPC, NVD, VIC) are the same
in all 44 readings; the EMC clock is not readable. Round `1105_h6h4` (another GPU process ran until just before)
had no slow part at the start; after 300 s of idle in the same process the first two generations took 137.9 / 138.3
ms per cycle and the next 8 recovered to 87.6 ms over about 50 s, while a 1 GiB copy probe dropped only 11 % (228.6 /
256.5 GB/s) in the first two and was back while decode still ran 94-118 ms: the slow state follows GPU idle. Round
`1110_h7h4bh8` (H7) inserted idle gaps of 5 / 15 / 30 / 60 / 120 s: none reproduced the plateau (first two
generations 87.0-87.3 ms after 5-60 s apart from one 164.9 ms, 96.7 / 95.4 ms after 120 s), so gaps shorter than the
monitor interval do not trigger it.

## The official planner with images on the Thor image frontend (`thor/planner_calls_vl.py`)

The official Qwen3.8-27B NVFP4 with its MTP head on the 22 recorded calls of ep0 and ep43 (wrist 224), each decoded
through the model's own EOS (lengths and reference replies from `cells/mtp_tail_base.json`), at `spec_k` 6 (the
prompt-length table) and 3, then one untimed 160-token call per call at K = 3 whose reply is parsed into the call's
decision (`thor/planner_calls_vl_table.py`: (status, action) for monitor, the subgoal list for decompose). Medians per
call; the same reply / same decision columns count calls equal to the reference:

| run | call | spec_k (effective K) | tokens per forward | ViT ms | prefill ms | decode ms | wall ms | first call (warm-up) wall ms | same reply | same decision |
|---|---|---|---|---|---|---|---|---|---|---|
| RTX 5090, `rtx` | decompose x 2 | 6 (6) | 3.48 | 41 | 174 | 321 | 546 | 1 778 | 2/2 | 2/2 |
| | decompose x 2 | 3 (3) | 2.94 | 41 | 174 | 313 | 538 | 1 266 | 2/2 | |
| | monitor x 20 | 6 (3) | 2.72 | 81 | 315 | 634 | 1 045 | 1 163 | 20/20 | 20/20 |
| | monitor x 20 | 3 (3) | 2.72 | 81 | 315 | 634 | 1 043 | 1 042 | 20/20 | |
| RTX 5090, `rtx_thor_vit` | decompose x 2 | 6 (6) | 4.49 | 65 | 175 | 249 | 498 | 1 535 | 0/2 | 0/2 |
| | decompose x 2 | 3 (3) | 3.45 | 65 | 175 | 265 | 515 | 1 197 | 0/2 | |
| | monitor x 20 | 6 (3) | 2.68 | 129 | 317 | 659 | 1 117 | 1 250 | 1/20 | 18/20 |
| | monitor x 20 | 3 (3) | 2.68 | 129 | 317 | 659 | 1 115 | 1 115 | 1/20 | |
| Jetson Thor, `thor`, M-tile tail fc (round `1104_h3`; decisions `1104_h3b`) | decompose x 2 | 6 | 3.84 | 140 | 3 652 | 1 365 | 5 173 | | 0/2 | 0/2 |
| | decompose x 2 | 3 | 2.94 | 140 | 3 653 | 1 435 | 5 242 | | 0/2 | |
| | monitor x 20 | 6 | 2.76 | 269 | 4 357 | 2 352 | 7 020 | | 0/20 | 18/20 |
| | monitor x 20 | 3 | 2.76 | 269 | 4 356 | 2 306 | 6 968 | | 0/20 | |
| Jetson Thor, `thor`, cuBLASLt tail fc (round `1104_h3c`) | decompose x 2 | 6 | 3.69 | 136 | 2 103 | 1 122 | 3 377 | 8 245 | 0/2 | 0/2 |
| | decompose x 2 | 3 | 2.94 | 136 | 2 104 | 1 136 | 3 390 | 5 920 | 0/2 | |
| | monitor x 20 | 6 | 2.81 | 270 | 3 986 | 2 228 | 6 506 | 6 972 | 0/20 | 18/20 |
| | monitor x 20 | 3 | 2.82 | 269 | 3 986 | 2 311 | 6 590 | 6 511 | 0/20 | |

- `rtx` = `Qwen36VlTorchFrontendRtx` (FP8 / FA2 ViT); `rtx_thor_vit` = the RTX language model with the Thor
  frontend's ViT configuration (BF16 linears, SDPA); `thor` = `Qwen36VlTorchFrontendThor`, which ran its M-tile tail
  fc until round `1104_h3b` and now inherits the RTX frontend's cuBLASLt tail fc (the `1104_h3b` row ran it as the
  variant `thor_cublaslt_tail` at commit `791b1168`, prefill 2 102 / 3 983 ms, monitor 2.82 tokens per forward). The
  `1104_h3c` row is the default path at commit `9609be92`, run after the 300 s drift process of the same round;
  monitor decode per cycle 85.8 ms (median), 2 of 44 timed calls slower (140.2, 101.8 ms). Logs and tables:
  `thor/planner_calls_vl_{rtx,rtx_thor_vit}.{json,log}`, `thor/planner_calls_vl_{rtx,rtx_thor_vit}_table.txt`.
- The first-call column of `spec_k` 6 includes the graph captures of each new prompt length (it runs first); peak
  allocated 22.1 / 22.4 GiB (`rtx` / `rtx_thor_vit`).
- The ViT configuration alone changes 42 of 44 replies and 4 of 22 decisions on the 5090: both decompose calls (ep0:
  3 subgoals become 2; ep43: the same 3 subgoals in longer wording), ep43 monitor@160 (`continue` -> `grasp_tool`)
  and ep43 monitor@800 (`replan` -> `continue`). Merged-feature cosine against the FP8 / FA2 default: mean
  0.9986-0.9993 (`smoke/vit_thor_config_check.log`). The BF16 / SDPA ViT takes 65 / 129 ms against 41 / 81 ms.
- On the Thor, every monitor call exceeds the 5.33 s monitor interval (40 of 40 with either tail fc). With the
  cuBLASLt tail fc (round `1104_h3c`) a monitor call takes 6.51 / 6.59 s (median, `spec_k` 6 / 3; 5.9-7.6 s per
  call), 6.2-6.3x the 5090: ViT 0.27 s (4 %), prefill 3.99 s (61 %, 12.7x the 5090's), decode 2.23-2.31 s (34-35 %,
  3.5-3.6x). With the M-tile tail fc (round `1104_h3`) it took 6.97-7.02 s, prefill 62 % of the call.
- Thor decisions (round `1104_h3b`, both tail fc kernels): 18 of 22 equal to the references, the same count as
  `rtx_thor_vit`; the two changes in meaning (ep0 decompose 3 -> 2 subgoals, ep43 monitor@800 `replan` ->
  `continue`) are the same as `rtx_thor_vit`'s; ep43 decompose differs in wording and ep0 monitor@720 does not parse
  (issues.md ISSUE-112).
- Thor prefill per prompt token: decompose 1.71 ms (2 048 tail rows), monitor 1.07 ms (512 tail rows). Fitting
  prefill = a x prompt tokens + b x tail rows to the two call kinds gives a = 0.97 ms, b = 0.77 ms (derived); the same
  coefficients fit H3a's three shapes within 3 %. `thor/mtp_tail_fc_kernels.py` measures the tail fc directly: on the
  Thor (round `1104_h3b`) the M-tile kernel takes 380 / 1 519 / 3 019 ms at 512 / 2 048 / 4 071 rows, the warp
  kernel 728 / 2 913 / 5 791 ms, cuBLASLt 0.89 / 3.46 / 14.7 ms; on the 5090 (`thor/mtp_tail_fc_kernels_rtx.log`)
  cuBLASLt 0.24 / 0.97 / 1.98 ms, the M-tile kernel unavailable. With cuBLASLt the Thor prefill is 0.98-0.99 ms per
  prompt token on both call kinds.
- Thor prefill by kernel binding (round `1110_h7h4bh8`, H4b, `thor/prefill_binding_times.py thor`): 3 981.9 ms,
  binding sum 4 047.3 ms; linear attention 57.1 % (28x the 5090's), GEMM 18.5 % (4.4x), xqa attention 8.3 %; one
  binding reported as `gdn_chunk` takes 2 235 ms, 55 % of the prefill (`opportunities.md` OPT-040 candidate 5).
- With the monitor prefix restored from a capsule and decode stopped when the `action` value closes, a Thor monitor
  call takes 2.61 s instead of 6.40 s (round `1110_h7h4bh8`, H8;
  `../../../long-horizon-planner-executor/experiments/monitor_call/README.md`).

## Files

`extract_prompts.py`, `prompts.json`, `volo_calls.py`, `record_calls.py` (recording), `planner_replay.py`,
`executor_imagewam.py`, `replay_bench.py`, `run_replay.sh` (replay grid and instruction switch), `replay_table.py`,
`first_call_check.py`, `prefill_mode_check.py`, `mtp_speedup.py`, `mtp_speedup_table.py`, `mtp_tail_model.py`,
`mtp_tail_model_table.py`, `thor/planner_call_shapes.py`, `thor/planner_calls_vl.py`, `thor/planner_calls_vl_table.py`,
`thor/mtp_tail_fc_kernels.py`, `thor/decode_cycle_drift.py`, `smoke/mtp_tail_fc_default.py`,
`smoke/vit_thor_config_check.py`; recorded calls in `calls/`,
results in `cells/` and `smoke/`, logs `record.log`, `replay.log`, `replay_realtime.log`, `mtp_speedup.log`,
`mtp_tail_{master,base}.log`.

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
- Tail fc: the frontend's small-M kernel costs 21 / 84 ms at 512 / 2 048 rows (cuBLASLt 0.4 / 1.2 ms); tokens per
  forward differ by at most 2 % between the two kernels (monitor 2.66 / 2.60), and tail + decode is shorter with
  cuBLASLt (decompose 302 -> 219 ms). The frontend default is the small-M kernel (OPT-039).
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
(`thor/planner_call_shapes_rtx.{json,log}`). RTX 5090: prefill 256 / 334 / 475 ms, decode 353 / 487 / 520 ms,
24.4-27.2 ms per cycle. ViT time and image content are not part of it.

## Files

`extract_prompts.py`, `prompts.json`, `volo_calls.py`, `record_calls.py` (recording), `planner_replay.py`,
`executor_imagewam.py`, `replay_bench.py`, `run_replay.sh` (replay grid and instruction switch), `replay_table.py`,
`first_call_check.py`, `prefill_mode_check.py`, `mtp_speedup.py`, `mtp_speedup_table.py`, `mtp_tail_model.py`,
`mtp_tail_model_table.py`, `thor/planner_call_shapes.py`; recorded calls in `calls/`,
results in `cells/` and `smoke/`, logs `record.log`, `replay.log`, `replay_realtime.log`, `mtp_speedup.log`,
`mtp_tail_{master,base}.log`.

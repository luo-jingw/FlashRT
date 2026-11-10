"""Monitor-call latency of the planner by prefix reuse and decode budget (FlashRT; RTX 5090 or Jetson AGX Thor).

Workload: the recorded monitor calls of the VoLoAgent Full-loop replay (`harness_replay_5090/calls/*.jsonl`). Every
monitor call of an episode has the same system prompt and the same three BEFORE images (the episode's step-0 frames);
only the three NOW images and the trailing text change. The reply is `{"status":...,"action":...,"reason":...}`; the
harness acts on `action`.

Grid per call (each cell: one warm-up call, one timed call):
  prefix  = "none"    : ViT over the six images, prefill of the whole prompt;
            "aligned" : prefix state restored from a capsule taken at capsule_aligned_len(B) (a multiple of the long
                        prefill chunk; token-identical to a cold full prefill by the frontend's contract), ViT over
                        the three NOW images, prefill of the rest;
            "exact"   : as "aligned" with the capsule at B itself (B = first token of the first NOW image);
  budget  = "eos"      : decode the reference reply's length through EOS (reply tokens + 1), as the replay does;
            "stop"     : the "eos" budget, decode stream read chunk by chunk, stopped after the chunk that closes the
                         action value (what a caller that acts on the action alone would do);
            "decision" : decode exactly the reference reply's tokens through the action's closing quote (a lower
                         bound of "stop": the budget is known in advance).
  plus one reference cell per call: the image frontend's `generate` with the "eos" budget (the replay's path).

Reference reply (REPLIES):
  recorded : the recorded reply (the replies were recorded by the LoRA-merged master; use with that checkpoint);
  own      : the model's own reply, from one untimed `generate` per call with OWN_BUDGET new tokens, cut at EOS.

Capsules are built once per episode from the episode's first monitor call (one warm-up build, one timed build).
Stage times are synchronized wall times (preprocess = chat template and image processing, vit, inject = embedding rows
and RoPE rows, restore, prefill, decode).

usage (cwd anywhere; PLANNER_CKPT = NVFP4 checkpoint with model.visual, PLANNER_MTP_DIR = MTP head directory):
  PLANNER_CKPT=DIR PLANNER_MTP_DIR=DIR python monitor_call_bench.py {rtx|thor} {recorded|own} OUT_JSON CALLS_JSONL...
"""
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
REPLAY_DIR = os.path.join(HERE, "..", "..", "..", "wam-inference-acceleration", "experiments", "harness_replay_5090")
CKPT = os.environ["PLANNER_CKPT"]
os.environ.update({
    "FLASHRT_QWEN36_MTP_CKPT_DIR": os.environ["PLANNER_MTP_DIR"],
    "FLASHRT_QWEN36_LONG_CTX_THRESHOLD": "4096",
    "FLASHRT_QWEN36_LONG_KV_CACHE": "fp8",
    "FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ": "1",
    "FLASHRT_QWEN36_TQ_PREFILL_GRAPH": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
})

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from flash_rt.frontends.torch import _qwen3_vl_geometry as geo  # noqa: E402
from flash_rt.frontends.torch.qwen36_vl_rtx import Qwen36VlTorchFrontendRtx  # noqa: E402
from flash_rt.frontends.torch.qwen36_vl_thor import Qwen36VlTorchFrontendThor  # noqa: E402

FRONTENDS = {"rtx": Qwen36VlTorchFrontendRtx, "thor": Qwen36VlTorchFrontendThor}
MAX_SEQ = 8448            # harness_replay_5090/record_calls.py
MAX_IMAGE_TOKENS = 4608   # harness_replay_5090/record_calls.py
CHAT_KWARGS = {"enable_thinking": False}
OWN_BUDGET = 160          # more than any recorded monitor reply (38-87 tokens)
ACTION_RE = re.compile(r'"action"\s*:\s*"([^"]*)"')
SPEC_K = 6  # Qwen36VlTorchFrontendRtx.generate default: the frontend's prompt-length table
LOOKAHEAD_ROWS = 32  # qwen36_vl_rtx._SPEC_LOOKAHEAD_ROWS


@dataclass
class PreparedCall:
    """One recorded monitor call after the chat template: token ids, image inputs and the prefix boundary."""
    row: dict                    # the recorded row (relative image paths)
    row_abs: dict                # the same row with absolute image paths
    input_ids: torch.Tensor      # (S,) long, on the device
    pixel_values: torch.Tensor   # (patches, dim) bf16
    grid: torch.Tensor           # (images, 3) long
    image_paths: list[str]
    boundary: int                # first token of the first NOW image
    first_now_image: int         # index of that image among the call's images
    preprocess_ms: float         # this first chat template (cold); every cell re-times its own


@dataclass
class CellRecord:
    episode: str
    step: int
    prefix: str          # none | aligned | exact | generate
    budget: str          # eos | stop | decision
    prefix_tokens: int   # tokens restored from the capsule
    prompt_tokens: int
    new_tokens: int
    preprocess_ms: float
    vit_ms: float
    inject_ms: float
    restore_ms: float
    prefill_ms: float
    decode_ms: float
    call_ms: float
    spec_attempts: int
    spec_accepts: int
    text: str
    same_reply: bool | None     # eos budget: reply equals the reference reply
    decision: str | None        # action parsed from the emitted text
    same_decision: bool         # action equals the reference reply's action
    recorded_decision: str | None  # action of the recorded (master) reply


@dataclass
class CapsuleRecord:
    episode: str
    prefix: str
    prefix_tokens: int
    build_ms: float      # ViT of the BEFORE images + inject + prefix prefill + snapshot
    nbytes: int


def to_messages(row: dict) -> list[dict]:
    """The call as a chat (system message, then one user message whose parts keep the harness's order); same as
    harness_replay_5090/volo_calls.to_messages."""
    content = [{"type": "text", "text": p["text"]} if "text" in p
               else {"type": "image", "image": Image.open(p["image"]).convert("RGB")} for p in row["user_parts"]]
    return [{"role": "system", "content": row["system"]}, {"role": "user", "content": content}]


def parse_action(text: str) -> str | None:
    """The action value once its closing quote is emitted (`"action":"x"` or `"action": "x"`), else None."""
    m = ACTION_RE.search(text)
    return None if m is None else m.group(1)


def sync_ms(t0: float) -> float:
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


class MonitorBench:
    def __init__(self, frontend: str) -> None:
        self.fe = FRONTENDS[frontend](CKPT, max_seq=MAX_SEQ, max_image_tokens=MAX_IMAGE_TOKENS)
        self.llm = self.fe.llm
        self.tok = self.fe.processor.tokenizer
        torch.cuda.synchronize()

    # ---- inputs ----
    def prepare(self, row: dict, base_dir: str, before_paths: list[str]) -> PreparedCall:
        r = dict(row)
        r["user_parts"] = [{"image": os.path.join(base_dir, p["image"])} if "image" in p else p for p in row["user_parts"]]
        msgs = to_messages(r)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        inputs = self.fe.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            processor_kwargs={"device": self.fe.device}, **CHAT_KWARGS).to(self.fe.device)
        pre_ms = sync_ms(t0)
        ids = inputs["input_ids"][0]
        paths = [p["image"] for p in row["user_parts"] if "image" in p]
        first_now = next(i for i, p in enumerate(paths) if p not in before_paths)
        is_img = (ids == self.fe._image_token_id).cpu()
        starts = [i for i in range(len(is_img)) if is_img[i] and (i == 0 or not is_img[i - 1])]
        assert len(starts) == len(paths), (len(starts), len(paths))
        return PreparedCall(row, r, ids, inputs["pixel_values"].to(torch.bfloat16), inputs["image_grid_thw"], paths,
                            starts[first_now], first_now, pre_ms)

    def encode(self, pc: PreparedCall, first_image: int) -> torch.Tensor:
        """ViT features of images [first_image, n), in prompt order."""
        segs = geo.vision_segments(pc.input_ids.cpu(), pc.grid.cpu(), None, image_token_id=self.fe._image_token_id,
                                   video_token_id=self.fe._video_token_id, spatial_merge_size=self.fe._merge)
        feats, off = [], 0
        for k, sg in enumerate(segs):
            n = sg["patches"]
            if k >= first_image:
                grid = torch.tensor([sg["grid"]], dtype=torch.long)
                vcos, vsin = geo.vision_rope_cos_sin_cached(grid, self.fe._vcos_cache, self.fe._vsin_cache,
                                                            spatial_merge_size=self.fe._merge)
                pos = geo.vision_pos_embeds(grid, self.fe.vision.pos_embed,
                                            num_grid_per_side=self.fe._num_grid_per_side,
                                            spatial_merge_size=self.fe._merge, device=self.fe.device)
                emb, _ = self.fe.vision.forward(pc.pixel_values[off:off + n], pos, vcos, vsin)
                feats.append(emb)
            off += n
        return torch.cat(feats, dim=0).to(torch.bfloat16)

    def inject(self, pc: PreparedCall, feats: torch.Tensor, first_image_token: int, n_rows: int) -> torch.Tensor:
        """Write image features into the embedding slots of image tokens [first_image_token, ...) and the MRoPE rows
        of the whole prompt; return the language model's ids (image tokens -> their slots)."""
        mask = pc.input_ids == self.fe._image_token_id
        n_img = int(mask.sum())
        slot = self.llm.embedding_slot_base
        self.llm.write_embedding_rows(slot + first_image_token, feats)
        lm_ids = pc.input_ids.clone()
        lm_ids[mask] = slot + torch.arange(n_img, device=pc.input_ids.device)
        cos, sin = self.fe._rope_rows(pc.input_ids, pc.grid, n_rows)
        self.llm.set_rope_rows(0, cos, sin)
        return lm_ids.view(1, -1)

    def image_tokens_before(self, pc: PreparedCall, pos: int) -> int:
        return int((pc.input_ids[:pos] == self.fe._image_token_id).sum())

    # ---- capsules ----
    @torch.inference_mode()
    def build_capsule(self, pc: PreparedCall, prefix_len: int, budget: int) -> tuple[dict, float]:
        S = int(pc.input_ids.shape[0])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        feats = self.encode(pc, 0)[: self.image_tokens_before(pc, prefix_len)]
        lm_ids = self.inject(pc, feats, 0, S + budget + LOOKAHEAD_ROWS)
        self.llm.prefill_long_ctx_nvfp4_agent(lm_ids[:, :prefix_len], max_new_tokens=budget, K=SPEC_K)
        cap = self.llm.snapshot_capsule()
        ms = sync_ms(t0)
        self.llm.reset_rope_rows()
        return cap, ms

    # ---- cells ----
    @torch.inference_mode()
    def run_generate(self, pc: PreparedCall, budget: int) -> tuple[list[int], dict]:
        msgs = to_messages(pc.row_abs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g = self.fe.generate(msgs, max_new_tokens=budget, chat_template_kwargs=CHAT_KWARGS)
        call = sync_ms(t0)
        t = g.timings_ms
        st = {"preprocess_ms": t["preprocess"], "vit_ms": t["vit"], "inject_ms": t["inject"], "restore_ms": 0.0,
              "prefill_ms": t["prefill"], "decode_ms": t["decode"], "call_ms": call,
              "spec_attempts": int(getattr(self.llm, "_spec_attempts", 0)),
              "spec_accepts": int(getattr(self.llm, "_spec_accepts", 0))}
        return g.token_ids, st

    @torch.inference_mode()
    def run_agent(self, pc: PreparedCall, cap: dict | None, prefix_len: int, budget: int,
                  stop_at_action: bool = False) -> tuple[list[int], dict]:
        """One call on the committed-stream path. stop_at_action: consume the decode stream chunk by chunk (one chunk
        = the tokens committed by one verify cycle) and stop after the chunk that closes the action value."""
        S = int(pc.input_ids.shape[0])
        # preprocess: the chat template and image processing of this call (its output equals pc's, prepared once)
        msgs = to_messages(pc.row_abs)
        torch.cuda.synchronize()
        tp = time.perf_counter()
        self.fe.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            processor_kwargs={"device": self.fe.device}, **CHAT_KWARGS).to(self.fe.device)
        pre = sync_ms(tp)
        t0 = time.perf_counter()
        # With a capsule only the NOW images are encoded: the BEFORE images' embedding rows (needed by the "aligned"
        # prefix, whose boundary falls inside the BEFORE part) are the same for every call of the episode and stay in
        # their slots from the capsule build, as a ViT-output cache would hold them.
        first_img = 0 if cap is None else pc.first_now_image
        feats = self.encode(pc, first_img)
        vit = sync_ms(t0)
        t1 = time.perf_counter()
        first_tok = 0 if cap is None else self.image_tokens_before(pc, pc.boundary)
        lm_ids = self.inject(pc, feats, first_tok, S + budget + LOOKAHEAD_ROWS)
        inj = sync_ms(t1)
        t2 = time.perf_counter()
        if cap is None:
            restore = 0.0
            self.llm.prefill_long_ctx_nvfp4_agent(lm_ids, max_new_tokens=budget, K=SPEC_K)
        else:
            self.llm.restore_capsule(cap)
            restore = sync_ms(t2)
            t2 = time.perf_counter()
            self.llm.append_long_ctx_nvfp4_agent(lm_ids, start_pos=prefix_len, max_new_tokens=budget, K=SPEC_K)
        prefill = sync_ms(t2)
        t3 = time.perf_counter()
        toks: list[int] = []
        for chunk in self.llm.decode_long_ctx_nvfp4_committed_stream(max_new_tokens=budget, K=SPEC_K):
            toks.extend(chunk)
            if stop_at_action and parse_action(self.tok.decode(toks, skip_special_tokens=True)) is not None:
                break
        decode = sync_ms(t3)
        self.llm.reset_rope_rows()
        st = {"preprocess_ms": pre, "vit_ms": vit, "inject_ms": inj, "restore_ms": restore,
              "prefill_ms": prefill, "decode_ms": decode,
              "call_ms": pre + vit + inj + restore + prefill + decode,
              "spec_attempts": int(getattr(self.llm, "_spec_attempts", 0)),
              "spec_accepts": int(getattr(self.llm, "_spec_accepts", 0))}
        return toks, st

    def text_of(self, toks: list[int]) -> str:
        stop = next((i for i, t in enumerate(toks) if t in self.fe._eos_token_ids), len(toks))
        return self.tok.decode(toks[:stop], skip_special_tokens=True)

    def decision_tokens(self, toks: list[int]) -> int:
        """Smallest n such that toks[:n] holds the action value and its closing quote."""
        for n in range(1, len(toks) + 1):
            t = self.tok.decode(toks[:n], skip_special_tokens=True)
            if parse_action(t) is not None:
                return n
        return len(toks)


def episode_label(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def main() -> None:
    frontend, replies, out_path, jsonls = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    assert replies in ("recorded", "own"), replies
    bench = MonitorBench(frontend)
    cells: list[CellRecord] = []
    capsules: list[CapsuleRecord] = []
    structure: list[dict] = []

    def dump() -> None:
        with open(out_path, "w") as f:
            json.dump({"structure": structure, "capsules": [asdict(c) for c in capsules],
                       "cells": [asdict(c) for c in cells]}, f, indent=1)

    for jl in jsonls:
        ep = episode_label(jl)
        rows = [json.loads(line) for line in open(jl)]
        mon = [r for r in rows if r["kind"] == "monitor"]
        img_sets = [{p["image"] for p in r["user_parts"] if "image" in p} for r in mon]
        before = sorted(set.intersection(*img_sets))
        pcs = [bench.prepare(r, REPLAY_DIR, before) for r in mon]
        B = pcs[0].boundary
        for pc in pcs:
            assert pc.boundary == B and torch.equal(pc.input_ids[:B], pcs[0].input_ids[:B]), ep
        P_al = bench.llm.capsule_aligned_len(B)
        if replies == "recorded":
            ref_texts = [r["output_text"] for r in mon]
            budgets = [min(r["output_tokens"] + 1, r["max_tokens"]) for r in mon]
        else:
            ref_texts, budgets = [], []
            for pc in pcs:
                toks, _ = bench.run_generate(pc, OWN_BUDGET)
                ref_texts.append(bench.text_of(toks))
                n = len(bench.tok.encode(ref_texts[-1], add_special_tokens=False))
                budgets.append(min(n + 1, OWN_BUDGET))
        cap_budget = max(budgets)
        caps: dict[str, tuple[dict, int]] = {}
        for name, P in (("aligned", P_al), ("exact", B)):
            bench.build_capsule(pcs[0], P, cap_budget)            # warm-up (graph capture)
            cap, ms = bench.build_capsule(pcs[0], P, cap_budget)
            caps[name] = (cap, P)
            capsules.append(CapsuleRecord(ep, name, P, ms, int(cap.get("nbytes", 0))))
        print(f"{ep}: {len(mon)} monitor calls, prompt {int(pcs[0].input_ids.shape[0])}, boundary B {B}, "
              f"aligned {P_al}, capsule build ms " + ", ".join(f"{c.prefix} {c.build_ms:.0f}" for c in capsules[-2:]),
              flush=True)

        for pc, budget, ref_text in zip(pcs, budgets, ref_texts):
            row = pc.row
            S = int(pc.input_ids.shape[0])

            def record(prefix: str, bname: str, P: int, toks: list[int], st: dict) -> None:
                text = bench.text_of(toks)
                same = (text == ref_text) if bname == "eos" else None
                dec = parse_action(text)
                cells.append(CellRecord(ep, row["step"], prefix, bname, P, S, len(toks), text=text, same_reply=same,
                                        decision=dec, same_decision=(dec == parse_action(ref_text)),
                                        recorded_decision=row.get("decision"), **st))

            bench.run_generate(pc, budget)
            toks_ref, st = bench.run_generate(pc, budget)
            record("generate", "eos", 0, toks_ref, st)
            n_dec = bench.decision_tokens(toks_ref)
            structure.append({"episode": ep, "step": row["step"], "prompt_tokens": S, "boundary": B,
                              "aligned": P_al, "image_tokens": int((pc.input_ids == bench.fe._image_token_id).sum()),
                              "image_tokens_before_boundary": bench.image_tokens_before(pc, B),
                              "reply_tokens": budget - 1, "decision_tokens": n_dec,
                              "decision": parse_action(ref_text), "recorded_decision": row.get("decision")})
            for prefix in ("none", "aligned", "exact"):
                cap, P = (None, 0) if prefix == "none" else caps[prefix]
                for bname, b, stop in (("eos", budget, False), ("stop", budget, True), ("decision", n_dec, False)):
                    bench.run_agent(pc, cap, P, b, stop)
                    toks, st = bench.run_agent(pc, cap, P, b, stop)
                    record(prefix, bname, P, toks, st)
            last = cells[-10:]
            print(f"  step {row['step']:4d} " + " | ".join(
                f"{c.prefix[:3]}/{c.budget[:3]} {c.call_ms:6.0f}ms{'' if c.same_decision else ' DIFF'}"
                f"{'' if c.same_reply in (None, True) else ' txt'}" for c in last), flush=True)
            dump()
    dump()


if __name__ == "__main__":
    main()

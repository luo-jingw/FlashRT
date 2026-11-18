"""The planner model behind every harness: HF official Qwen3.8-27B (NVFP4) on the FlashRT image frontend.

HPE and MemER planners build their own chat messages and parse their own replies; this class runs one chat request
and reports what it cost. Generation is greedy on the long-context agent path with MTP, and decoding stops at the
first end-of-sequence token (the frontend's `generate` decodes the whole token budget before truncating, so its decode
time follows the budget, not the reply).

Environment (set before import, as in experiments/monitor_call/monitor_call_bench.py):
  PLANNER_CKPT      NVFP4 checkpoint directory (with model.visual)
  PLANNER_MTP_DIR   MTP head directory
On Thor also prepend the transformers 5.x overlay to PYTHONPATH and set FLASHRT_QWEN36_THOR_LIN_PREFILL=parent,
FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND=wy_lt (see experiments/thor_bringup/run_bringup.sh).

CUDA graphs are recorded lazily per prompt length and decode position, so the first calls of a new shape are slower;
runs that use measured latency need a warm-up pass.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

os.environ.setdefault("FLASHRT_QWEN36_MTP_CKPT_DIR", os.environ.get("PLANNER_MTP_DIR", ""))
os.environ.setdefault("FLASHRT_QWEN36_LONG_CTX_THRESHOLD", "4096")
os.environ.setdefault("FLASHRT_QWEN36_LONG_KV_CACHE", "fp8")
os.environ.setdefault("FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ", "1")
os.environ.setdefault("FLASHRT_QWEN36_TQ_PREFILL_GRAPH", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

from flash_rt.frontends.torch import _qwen3_vl_geometry as geo  # noqa: E402
from flash_rt.frontends.torch.qwen36_vl_rtx import Qwen36VlTorchFrontendRtx  # noqa: E402
from flash_rt.frontends.torch.qwen36_vl_thor import Qwen36VlTorchFrontendThor  # noqa: E402

FRONTENDS = {"rtx": Qwen36VlTorchFrontendRtx, "thor": Qwen36VlTorchFrontendThor}
THINK_END = "</think>"
SPEC_K = 6                 # the frontend's prompt-length table, as in monitor_call_bench.py
LOOKAHEAD_ROWS = 32        # qwen36_vl_rtx._SPEC_LOOKAHEAD_ROWS
MIN_MAX_SEQ = 8448         # below this a short prompt does not take the long-context route


@dataclass(frozen=True)
class Generation:
    answer: str                    # text after </think> (the whole reply when thinking is off)
    thinking: str                  # text before </think>; empty when thinking is off
    input_tokens: int
    output_tokens: int
    image_tokens: int
    preprocess_ms: float           # chat template and image preprocessing
    vit_ms: float
    prefill_ms: float              # image-row injection and language-model prefill
    decode_ms: float               # until the end-of-sequence token
    hit_budget: bool               # stopped at max_new_tokens without an end-of-sequence token


def _ms(t0: float) -> float:
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


class QwenVlBackend:
    def __init__(self, frontend: str, max_seq: int = 16384, max_image_tokens: int = 8192) -> None:
        self.fe = FRONTENDS[frontend](os.environ["PLANNER_CKPT"], max_seq=max(MIN_MAX_SEQ, max_seq),
                                      max_image_tokens=max_image_tokens)
        self.llm = self.fe.llm
        self.tok = self.fe.processor.tokenizer
        self.stop_ids = sorted(self.fe._eos_token_ids)

    def _encode(self, ids: torch.Tensor, pixel_values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        segs = geo.vision_segments(ids.cpu(), grid.cpu(), None, image_token_id=self.fe._image_token_id,
                                   video_token_id=self.fe._video_token_id, spatial_merge_size=self.fe._merge)
        feats, off = [], 0
        for sg in segs:
            n = sg["patches"]
            g = torch.tensor([sg["grid"]], dtype=torch.long)
            vcos, vsin = geo.vision_rope_cos_sin_cached(g, self.fe._vcos_cache, self.fe._vsin_cache,
                                                        spatial_merge_size=self.fe._merge)
            pos = geo.vision_pos_embeds(g, self.fe.vision.pos_embed, num_grid_per_side=self.fe._num_grid_per_side,
                                        spatial_merge_size=self.fe._merge, device=self.fe.device)
            emb, _ = self.fe.vision.forward(pixel_values[off:off + n], pos, vcos, vsin)
            feats.append(emb)
            off += n
        return torch.cat(feats, dim=0).to(torch.bfloat16)

    @torch.inference_mode()
    def generate(self, messages: list[dict[str, object]], *, thinking: bool, max_new_tokens: int) -> Generation:
        """messages: chat messages in the Qwen-VL format; images as {"type": "image", "image": PIL.Image}."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        inputs = self.fe.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            processor_kwargs={"device": self.fe.device}, enable_thinking=thinking).to(self.fe.device)
        ids = inputs["input_ids"][0]
        pre = _ms(t0)
        S = int(ids.shape[0])
        n_img = int((ids == self.fe._image_token_id).sum())
        t1 = time.perf_counter()
        feats = (self._encode(ids, inputs["pixel_values"].to(torch.bfloat16), inputs["image_grid_thw"])
                 if n_img else None)
        vit = _ms(t1)
        t2 = time.perf_counter()
        lm_ids = ids.clone()
        if n_img:
            mask = ids == self.fe._image_token_id
            slot = self.llm.embedding_slot_base
            self.llm.write_embedding_rows(slot, feats)
            lm_ids[mask] = slot + torch.arange(n_img, device=ids.device)
            cos, sin = self.fe._rope_rows(ids, inputs["image_grid_thw"], S + max_new_tokens + LOOKAHEAD_ROWS)
            self.llm.set_rope_rows(0, cos, sin)
        try:
            self.llm.prefill_long_ctx_nvfp4_agent(lm_ids.view(1, -1), max_new_tokens=max_new_tokens, K=SPEC_K)
            prefill = _ms(t2)
            t3 = time.perf_counter()
            toks: list[int] = []
            for chunk in self.llm.decode_long_ctx_nvfp4_committed_stream(max_new_tokens=max_new_tokens, K=SPEC_K,
                                                                           stop_token_ids=self.stop_ids):
                toks.extend(chunk)
                if any(t in self.stop_ids for t in chunk):
                    break
            decode = _ms(t3)
        finally:
            if n_img:
                self.llm.reset_rope_rows()
        stop = next((i for i, t in enumerate(toks) if t in self.stop_ids), None)
        gen = toks if stop is None else toks[:stop]
        text = self.tok.decode(gen, skip_special_tokens=True)
        if THINK_END in text:
            thinking_text, answer = text.split(THINK_END, 1)
        else:
            thinking_text, answer = ("", text) if not thinking else (text, "")
        return Generation(answer.strip(), thinking_text.strip(), S, len(gen), n_img, pre, vit, prefill, decode,
                          stop is None)

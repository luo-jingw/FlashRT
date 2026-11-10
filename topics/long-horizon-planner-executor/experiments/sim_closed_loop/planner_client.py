"""The FlashRT planner (Qwen3.8-27B NVFP4, MTP, thinking off) answering harness calls in a closed loop.

One call = chat template + ViT + embedding / MRoPE injection + prefill on the committed-stream path + a decode stream
read chunk by chunk (one chunk = the tokens one verify cycle commits), stopped at EOS, or (stop="action") after the chunk
that closes the reply's action value. A monitor prefix capsule (system prompt + BEFORE images, cut at the long-prefill
chunk boundary, token-identical to a full prefill) can be built once per episode and restored by later monitor calls,
which then encode only the NOW images and prefill the rest
(`../monitor_call/README.md`).
"""
import re
import time
from dataclasses import dataclass

import torch

from flash_rt.frontends.torch import _qwen3_vl_geometry as geo
from flash_rt.frontends.torch.qwen36_vl_rtx import Qwen36VlTorchFrontendRtx

SPEC_K = 6            # the frontend's prompt-length table
LOOKAHEAD_ROWS = 32   # qwen36_vl_rtx._SPEC_LOOKAHEAD_ROWS
ACTION_RE = re.compile(r'"action"\s*:\s*"([^"]*)"')


@dataclass
class PlannerReply:
    text: str
    new_tokens: int
    prompt_tokens: int
    restored_tokens: int   # prefix tokens restored from a capsule (0 without)
    preprocess_ms: float
    vit_ms: float
    inject_ms: float
    restore_ms: float
    prefill_ms: float
    decode_ms: float
    call_ms: float


@dataclass
class MonitorPrefix:
    capsule: dict
    tokens: int            # restored prefix length (chunk-aligned)
    image_tokens: int      # image tokens inside the restored prefix
    first_image: int       # first image not covered by the cached features (the first NOW image)
    build_ms: float


def _sync_ms(t0: float) -> float:
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


class PlannerClient:
    def __init__(self, ckpt: str, max_seq: int, max_image_tokens: int, chat_kwargs: dict) -> None:
        self.fe = Qwen36VlTorchFrontendRtx(ckpt, max_seq=max_seq, max_image_tokens=max_image_tokens)
        self.llm = self.fe.llm
        self.tok = self.fe.processor.tokenizer
        self.chat_kwargs = chat_kwargs
        torch.cuda.synchronize()

    def _inputs(self, msgs: list[dict]):
        return self.fe.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            processor_kwargs={"device": self.fe.device}, **self.chat_kwargs).to(self.fe.device)

    def _image_starts(self, ids: torch.Tensor) -> list[int]:
        m = (ids == self.fe._image_token_id).cpu()
        return [i for i in range(len(m)) if m[i] and (i == 0 or not m[i - 1])]

    def _encode(self, ids: torch.Tensor, pixel_values: torch.Tensor, grid: torch.Tensor,
                first_image: int) -> torch.Tensor:
        segs = geo.vision_segments(ids.cpu(), grid.cpu(), None, image_token_id=self.fe._image_token_id,
                                   video_token_id=self.fe._video_token_id, spatial_merge_size=self.fe._merge)
        feats, off = [], 0
        for k, sg in enumerate(segs):
            n = sg["patches"]
            if k >= first_image:
                g = torch.tensor([sg["grid"]], dtype=torch.long)
                vcos, vsin = geo.vision_rope_cos_sin_cached(g, self.fe._vcos_cache, self.fe._vsin_cache,
                                                            spatial_merge_size=self.fe._merge)
                pos = geo.vision_pos_embeds(g, self.fe.vision.pos_embed, num_grid_per_side=self.fe._num_grid_per_side,
                                            spatial_merge_size=self.fe._merge, device=self.fe.device)
                emb, _ = self.fe.vision.forward(pixel_values[off:off + n], pos, vcos, vsin)
                feats.append(emb)
            off += n
        return torch.cat(feats, dim=0).to(torch.bfloat16)

    def _inject(self, ids: torch.Tensor, grid: torch.Tensor, feats: torch.Tensor, first_image_token: int,
                n_rows: int) -> torch.Tensor:
        mask = ids == self.fe._image_token_id
        slot = self.llm.embedding_slot_base
        if feats.shape[0]:
            self.llm.write_embedding_rows(slot + first_image_token, feats)
        lm_ids = ids.clone()
        lm_ids[mask] = slot + torch.arange(int(mask.sum()), device=ids.device)
        cos, sin = self.fe._rope_rows(ids, grid, n_rows)
        self.llm.set_rope_rows(0, cos, sin)
        return lm_ids.view(1, -1)

    @torch.inference_mode()
    def build_monitor_prefix(self, msgs: list[dict], first_now_image: int, budget: int) -> MonitorPrefix:
        """Capsule of a monitor prompt's episode-constant prefix, cut at the chunk boundary at or before the first NOW
        image. `msgs` is any monitor call of the episode (its NOW images are not part of the prefix)."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        inp = self._inputs(msgs)
        ids = inp["input_ids"][0]
        B = self._image_starts(ids)[first_now_image]
        P = self.llm.capsule_aligned_len(B)
        n_img = int((ids[:B] == self.fe._image_token_id).sum())
        feats = self._encode(ids, inp["pixel_values"].to(torch.bfloat16), inp["image_grid_thw"], 0)[:n_img]
        lm_ids = self._inject(ids, inp["image_grid_thw"], feats, 0, int(ids.shape[0]) + budget + LOOKAHEAD_ROWS)
        self.llm.prefill_long_ctx_nvfp4_agent(lm_ids[:, :P], max_new_tokens=budget, K=SPEC_K)
        cap = self.llm.snapshot_capsule()
        ms = _sync_ms(t0)
        self.llm.reset_rope_rows()
        return MonitorPrefix(cap, P, n_img, first_now_image, ms)

    @torch.inference_mode()
    def call(self, msgs: list[dict], max_tokens: int, *, stop: str = "eos",
             prefix: MonitorPrefix | None = None) -> PlannerReply:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        inp = self._inputs(msgs)
        ids = inp["input_ids"][0]
        S = int(ids.shape[0])
        pre = _sync_ms(t0)
        t1 = time.perf_counter()
        n_img = int((ids == self.fe._image_token_id).sum())
        if n_img:
            feats = self._encode(ids, inp["pixel_values"].to(torch.bfloat16), inp["image_grid_thw"],
                                 0 if prefix is None else prefix.first_image)
        vit = _sync_ms(t1)
        t2 = time.perf_counter()
        first_tok = 0 if prefix is None else prefix.image_tokens
        lm_ids = (self._inject(ids, inp["image_grid_thw"], feats, first_tok, S + max_tokens + LOOKAHEAD_ROWS)
                  if n_img else ids.view(1, -1))
        inj = _sync_ms(t2)
        t3 = time.perf_counter()
        restore = 0.0
        if prefix is None:
            self.llm.prefill_long_ctx_nvfp4_agent(lm_ids, max_new_tokens=max_tokens, K=SPEC_K)
        else:
            self.llm.restore_capsule(prefix.capsule)
            restore = _sync_ms(t3)
            t3 = time.perf_counter()
            self.llm.append_long_ctx_nvfp4_agent(lm_ids, start_pos=prefix.tokens, max_new_tokens=max_tokens, K=SPEC_K)
        prefill = _sync_ms(t3)
        t4 = time.perf_counter()
        eos = set(self.fe._eos_token_ids)
        toks: list[int] = []
        for chunk in self.llm.decode_long_ctx_nvfp4_committed_stream(max_new_tokens=max_tokens, K=SPEC_K,
                                                                     stop_token_ids=eos):
            toks.extend(chunk)
            if any(t in eos for t in chunk):
                break
            if stop == "action" and ACTION_RE.search(self.tok.decode(toks, skip_special_tokens=True)):
                break
        decode = _sync_ms(t4)
        if n_img:
            self.llm.reset_rope_rows()
        cut = next((i for i, t in enumerate(toks) if t in eos), len(toks))
        text = self.tok.decode(toks[:cut], skip_special_tokens=True)
        return PlannerReply(text, cut, S, 0 if prefix is None else prefix.tokens, pre, vit, inj, restore, prefill,
                            decode, pre + vit + inj + restore + prefill + decode)


def action_of(text: str) -> str | None:
    m = ACTION_RE.search(text)
    return None if m is None else m.group(1)


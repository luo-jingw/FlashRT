"""FlashRT -- image input for the Qwen3.5-architecture 27B NVFP4 frontend on RTX SM120 (Qwen3.6-27B, Qwen3.8-27B).

Composes ``Qwen36TorchFrontendRtx`` (language model, unchanged forward) with ``Qwen3VlVisionRtx`` (the Qwen3-VL ViT;
the Qwen3.5 vision tower is the same network without DeepStack). Per request:

1. the checkpoint's processor renders the chat and patchifies the images;
2. the ViT encodes each image (one call per image, its own attention window);
3. the features go into the language model's reserved embedding rows, and the prompt's image-token positions are
   remapped to those rows' token ids;
4. the RoPE rows of the prompt get interleaved-MRoPE positions (t, h, w), the rows after it 1-D positions continued
   from max(position) + 1, written into the language model's RoPE tables in place;
5. the language model's MTP speculative generation runs unchanged; the RoPE tables are restored afterwards.

The checkpoint directory holds the NVFP4 language model and ``model.visual.*`` (BF16) in ``model.safetensors``, plus
``config.json``, tokenizer and processor files. The MTP head comes from ``FLASHRT_QWEN36_MTP_CKPT_DIR``, as for the
text frontend. Images only; video input raises ``NotImplementedError``.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import torch

from flash_rt.frontends.torch import _qwen3_vl_geometry as geo
from flash_rt.frontends.torch._qwen36_mrope import text_rope_rows
from flash_rt.frontends.torch._qwen3_vl_vision_rtx import Qwen3VlVisionRtx
from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

_SPEC_LOOKAHEAD_ROWS = 32


@dataclass
class VlGeneration:
    token_ids: list[int]
    text: str
    timings_ms: dict[str, float]
    num_image_tokens: int
    prompt_len: int
    route: str


class Qwen36VlTorchFrontendRtx:
    """Image + text generation on the Qwen3.6-architecture NVFP4 frontend."""

    def __init__(self, checkpoint_path: str, *, device: str = 'cuda:0', max_seq: int, max_image_tokens: int,
                 max_pixels: int | None = None) -> None:
        from transformers import AutoProcessor

        cfg = json.load(open(os.path.join(checkpoint_path, 'config.json')))
        tc = cfg['text_config']
        rp = tc['rope_parameters']
        vc = cfg['vision_config']
        if vc.get('deepstack_visual_indexes'):
            raise ValueError('a vision tower with DeepStack layers is not supported by this frontend')
        self.device = device
        self.max_image_tokens = int(max_image_tokens)
        self._image_token_id = int(cfg['image_token_id'])
        self._video_token_id = int(cfg['video_token_id'])
        self._vision_start_token_id = int(cfg['vision_start_token_id'])
        self._merge = int(vc['spatial_merge_size'])
        self._num_grid_per_side = int(vc['num_position_embeddings'] ** 0.5)
        self._rotary_dim = int(tc['head_dim'] * rp['partial_rotary_factor'])
        self._rope_theta = float(rp['rope_theta'])
        self._mrope_section = tuple(rp['mrope_section'])
        gen_path = os.path.join(checkpoint_path, 'generation_config.json')
        eos = json.load(open(gen_path)).get('eos_token_id') if os.path.isfile(gen_path) else tc.get('eos_token_id')
        self._eos_token_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()

        self.llm = Qwen36TorchFrontendRtx(checkpoint_path, device=device, max_seq=max_seq, quant='nvfp4',
                                          extra_embedding_rows=self.max_image_tokens)
        self.vision = Qwen3VlVisionRtx(checkpoint_path, device=device, config=vc)
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)
        geo.set_processor_max_pixels(self.processor, max_pixels)
        self._vcos_cache, self._vsin_cache = geo.build_vision_rope_cache(
            max_hw=self.max_image_tokens * self._merge, head_dim=vc['hidden_size'] // vc['num_heads'], device=device)

    def _encode_images(self, input_ids: torch.Tensor, pixel_values: torch.Tensor,
                       image_grid_thw: torch.Tensor) -> torch.Tensor:
        """ViT features of every image, in prompt order: (number of image tokens, hidden) bf16."""
        segs = geo.vision_segments(input_ids.cpu(), image_grid_thw.cpu(), None, image_token_id=self._image_token_id,
                                   video_token_id=self._video_token_id, spatial_merge_size=self._merge)
        feats, off = [], 0
        for sg in segs:
            n = sg['patches']
            grid = torch.tensor([sg['grid']], dtype=torch.long)
            vcos, vsin = geo.vision_rope_cos_sin_cached(grid, self._vcos_cache, self._vsin_cache,
                                                        spatial_merge_size=self._merge)
            pos = geo.vision_pos_embeds(grid, self.vision.pos_embed, num_grid_per_side=self._num_grid_per_side,
                                        spatial_merge_size=self._merge, device=self.device)
            emb, _ = self.vision.forward(pixel_values[off:off + n], pos, vcos, vsin)
            feats.append(emb)
            off += n
        return torch.cat(feats, dim=0).to(torch.bfloat16)

    def _rope_rows(self, input_ids: torch.Tensor, image_grid_thw: torch.Tensor, n_rows: int
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE rows for cache positions [0, n_rows): MRoPE positions over the prompt, then 1-D positions continued
        from max(prompt position) + 1."""
        pos = geo.mrope_position_ids(input_ids.cpu(), image_grid_thw.cpu(), None, image_token_id=self._image_token_id,
                                     video_token_id=self._video_token_id,
                                     vision_start_token_id=self._vision_start_token_id,
                                     spatial_merge_size=self._merge)
        S = int(input_ids.shape[0])
        nxt = int(pos.max()) + 1
        tail = torch.arange(nxt, nxt + n_rows - S, dtype=pos.dtype).view(1, -1).expand(3, -1)
        all_pos = torch.cat([pos, tail], dim=1).to(self.device)
        return text_rope_rows(all_pos, rotary_dim=self._rotary_dim, rope_theta=self._rope_theta,
                              mrope_section=self._mrope_section)

    def generate(self, messages: list, *, max_new_tokens: int,
                 chat_template_kwargs: dict | None = None) -> VlGeneration:
        """Greedy generation for a chat ``messages`` list whose user content may hold ``{"type": "image"}`` items.
        Output tokens stop at the first end-of-sequence token."""
        llm = self.llm
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt',
            processor_kwargs={'device': self.device}, **(chat_template_kwargs or {})).to(self.device)
        if inputs.get('pixel_values_videos') is not None:
            raise NotImplementedError('video input is not supported by this frontend')
        input_ids = inputs['input_ids'][0]
        S = int(input_ids.shape[0])
        n_rows = S + int(max_new_tokens) + _SPEC_LOOKAHEAD_ROWS
        if n_rows > llm.max_seq:
            raise ValueError(f'prompt {S} + max_new_tokens {max_new_tokens} + {_SPEC_LOOKAHEAD_ROWS} lookahead rows '
                             f'exceeds max_seq {llm.max_seq}')
        grid = inputs.get('image_grid_thw')
        image_mask = input_ids == self._image_token_id
        n_img = int(image_mask.sum())
        if n_img > self.max_image_tokens:
            raise ValueError(f'{n_img} image tokens exceed max_image_tokens {self.max_image_tokens}')
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        lm_ids = input_ids
        if n_img:
            feats = self._encode_images(input_ids, inputs['pixel_values'].to(torch.bfloat16), grid)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            slot = llm.embedding_slot_base
            llm.write_embedding_rows(slot, feats)
            lm_ids = input_ids.clone()
            lm_ids[image_mask] = slot + torch.arange(n_img, device=input_ids.device)
            cos, sin = self._rope_rows(input_ids, grid, n_rows)
            llm.set_rope_rows(0, cos, sin)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
        else:
            t2 = t3 = t1
        try:
            out = llm.generate_own_speculative_KN_nvfp4(lm_ids.view(1, -1), max_new_tokens=int(max_new_tokens))
        finally:
            if n_img:
                llm.reset_rope_rows()
        gen = out[0, S:].tolist()
        stop = next((i for i, t in enumerate(gen) if t in self._eos_token_ids), len(gen))
        gen = gen[:stop]
        timings = {'preprocess': (t1 - t0) * 1e3, 'vit': (t2 - t1) * 1e3, 'inject': (t3 - t2) * 1e3,
                   'prefill': float(llm._long_ctx_prefill_ms), 'decode': float(llm._long_ctx_decode_ms)}
        return VlGeneration(token_ids=gen, text=self.processor.tokenizer.decode(gen, skip_special_tokens=True),
                            timings_ms=timings, num_image_tokens=n_img, prompt_len=S,
                            route=str(getattr(llm, '_long_ctx_route', '')))

"""Real ImageWAM/FLUX.2 text-context encoding via live Qwen3 (real VAE
+ text-context wiring plan's own deferred "Live Qwen3" item, closed
2026-09-15).

Ported directly from `imagewam.py`'s own real `_encode_flux2_prompts`
(read at `/home/ljw/projects/pi0.5/tmp/ImageWAM/src/imagewam/models/backbones/imagewam.py`):
Qwen3's own chat template (`enable_thinking=False`, `add_generation_prompt=True`)
-> tokenize (`padding="max_length"`, `truncation=True`, `max_length=512`
-- confirmed against `flux2.text_encoder.MAX_LENGTH` and
`_imagewam_thor_spec.py`'s own declared `context` shape `(512, ...)`,
NOT the `x0=128` this project's own bench/test dims had assumed
throughout, see `opportunities.md`'s own correction) -> Qwen3ForCausalLM
forward with `output_hidden_states=True` -> concatenate 3 specific
hidden layers (`flux2.text_encoder.OUTPUT_LAYERS_QWEN3 = [9, 18, 27]`,
confirmed by reading that module directly) -> `(B, L, 3*hidden_dim)`.
For `Qwen/Qwen3-4B` (`hidden_size=2560`), `3*2560=7680`, matching
`JOINT_ATTENTION_DIM` exactly.

`transformers.Qwen3ForCausalLM` is available in this project's own
isolated venv (`FlashRT/.venv`, see `PROJECT.md`) but there is no
local Qwen3-4B checkpoint by default -- `load_real_text_encoder` uses
`transformers.AutoModelForCausalLM.from_pretrained`, which downloads
from the Hub on first use unless `model_spec` already points at a
local snapshot directory.
"""
from __future__ import annotations

import torch

DEV = "cuda"
FP16 = torch.float16
BF16 = torch.bfloat16

_OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
_MAX_LENGTH = 512


def load_real_text_encoder(model_spec: str = "Qwen/Qwen3-4B", *, device: str = DEV,
                            dtype: torch.dtype = torch.bfloat16):
    """Loads the real Qwen3 text encoder + tokenizer, matching
    `imagewam.py`'s own `_load_flux2_text_encoder_for_inference` exactly.
    `model_spec` may be an HF repo id (downloads on first use) or a
    local snapshot directory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_spec, torch_dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_spec)
    return model, tokenizer


@torch.no_grad()
def encode_prompts(model, tokenizer, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Real prompts -> `(context, context_mask)`, matching
    `imagewam.py`'s own `_encode_flux2_prompts` exactly (chat template,
    `enable_thinking=False`; tokenize to a fixed `max_length=512`;
    forward with `output_hidden_states=True`; concatenate layers
    `[9, 18, 27]`). `context`: `(B, 512, 3*hidden_dim)` **BF16** CUDA
    (real Thor measurement, opportunities.md OPT-001 "FP16 residual
    overflow": real token positions -- e.g. the chat-template's own
    first special token, a well-known LLM "attention sink" -- reach
    absmax~16000 in the model's native bf16; once fed through
    `ImageWAMTorchFrontendThor`'s real trained `txt_in` weight and
    accumulated across the real 25-layer backbone's residual stream,
    the running sum legitimately reaches ~120000, which FP16's ~65504
    ceiling cannot hold at all -- this function used to downcast to
    FP16 here, which is what silently produced that overflow).
    `context_mask`: `(B, 512)` bool CUDA (`1` for real tokens, `0` for
    padding -- this project's own `context_mask` input, declared in
    `_imagewam_thor_spec.py` but never previously load-bearing)."""
    device = next(model.parameters()).device
    all_input_ids, all_attention_masks = [], []
    for prompt in prompts:
        messages = [{"role": "user", "content": str(prompt)}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                                  truncation=True, max_length=_MAX_LENGTH)
        all_input_ids.append(model_inputs["input_ids"])
        all_attention_masks.append(model_inputs["attention_mask"])

    input_ids = torch.cat(all_input_ids, dim=0).to(device)
    attention_mask = torch.cat(all_attention_masks, dim=0).to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                     output_hidden_states=True, use_cache=False)
    hidden = torch.stack([outputs.hidden_states[k] for k in _OUTPUT_LAYERS_QWEN3], dim=1)
    hidden = hidden.permute(0, 2, 1, 3).reshape(hidden.shape[0], hidden.shape[2], -1)  # b c l d -> b l (c d)
    return hidden.to(dtype=BF16), attention_mask.to(dtype=torch.bool)

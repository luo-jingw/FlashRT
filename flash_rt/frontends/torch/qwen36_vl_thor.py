"""FlashRT -- image input for the Qwen3.5-architecture 27B NVFP4 frontend on Jetson Thor (SM110).

Same request flow as ``Qwen36VlTorchFrontendRtx`` (processor, one ViT call per image, image features in the
language model's reserved embedding rows, interleaved-MRoPE rows written into its RoPE tables, MTP speculative
generation). Two components differ:

- language model: ``Qwen36TorchFrontendThor`` (a subclass of the RTX frontend that keeps its embedding table, RoPE
  tables and their write / reset methods, and replaces the attention backend and the K-row forward for SM110);
- ViT: ``Qwen3VlVisionRtx`` with BF16 linears and SDPA attention (Thor's Qwen3-VL kernel module has no FP8 block-128
  ViT GEMM and FA2 is not built on Thor), as in ``qwen3_vl_thor``.
"""
from __future__ import annotations

from flash_rt.frontends.torch._qwen3_vl_vision_rtx import Qwen3VlVisionRtx
from flash_rt.frontends.torch.qwen36_thor import Qwen36TorchFrontendThor
from flash_rt.frontends.torch.qwen36_vl_rtx import Qwen36VlTorchFrontendRtx


class Qwen36VlTorchFrontendThor(Qwen36VlTorchFrontendRtx):
    """Image + text generation on the Qwen3.6-architecture NVFP4 frontend, Thor."""

    def _build_language_model(self, checkpoint_path: str, device: str, max_seq: int) -> Qwen36TorchFrontendThor:
        """The Thor NVFP4 language model with ``max_image_tokens`` reserved embedding rows."""
        return Qwen36TorchFrontendThor(checkpoint_path, device=device, max_seq=max_seq, quant='nvfp4',
                                       extra_embedding_rows=self.max_image_tokens)

    def _build_vision(self, checkpoint_path: str, device: str, config: dict) -> Qwen3VlVisionRtx:
        """The ViT with BF16 linears and SDPA attention."""
        return Qwen3VlVisionRtx(checkpoint_path, device=device, config=config, fp8=False, attention_backend='sdpa')

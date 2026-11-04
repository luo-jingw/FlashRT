"""FlashRT - Qwen3.6-27B NVFP4 Spark frontend (SM121).

Spark reuses the RTX Qwen3.6 compute path, including its MTP prompt-tail
K/V prefill (NVFP4 k/v projections for the FP8-source MTP head), and
overrides only the GB10-measured policies: the prompt-tail length, the
long-context speculative K and the FP8-KV XQA buckets.
"""

from __future__ import annotations

import os

from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx


class Qwen36TorchFrontendSpark(Qwen36TorchFrontendRtx):
    """Qwen3.6 NVFP4 frontend for DGX Spark / GB10 (SM121).

    This class is intentionally narrow: it keeps the RTX attention backend,
    all SM120/SM121 GEMM dispatch and the MTP tail prefill, and replaces
    only the measured bucket policies.
    """

    def _long_mtp_prefill_tail_for_prompt(self, prompt_len: int) -> int:
        raw = os.environ.get(
            'FLASHRT_QWEN36_LONG_MTP_PREFILL_TAIL', 'auto') or 'auto'
        if raw.lower() != 'auto':
            return max(0, int(raw))
        mtp = self._weights.ptrs.get('mtp') if self._weights else None
        if not isinstance(mtp, dict):
            return 0
        prompt_len = int(prompt_len)
        if prompt_len >= 128 and prompt_len < 512:
            return min(128, prompt_len)
        if prompt_len < 512:
            return 0
        if prompt_len < 768:
            return 512
        if prompt_len < 3072:
            return 2048
        if prompt_len < 6144:
            return 512
        if prompt_len < 24576:
            return 4096
        return 2048

    def _long_tq_effective_k(
            self, prompt_len: int, K: int,
            max_new_tokens: int | None = None) -> int:
        target_k = super()._long_tq_effective_k(
            prompt_len, K, max_new_tokens)
        if os.environ.get('FLASHRT_QWEN36_TQ_SPEC_K', ''):
            return target_k
        prompt_len = int(prompt_len)
        if 6144 <= prompt_len < 12288:
            target_k = 7
        elif 12288 <= prompt_len < 24576:
            target_k = 6
        elif 24576 <= prompt_len < 49152:
            target_k = 7
        caller_k = int(K)
        if caller_k < 6:
            return min(caller_k, target_k)
        return target_k

    def _fp8_xqa_auto_bucket_enabled(
            self, q_seq: int | None, end_pos: int) -> bool:
        """Spark-measured FP8-KV XQA bucket policy.

        GB10/SM121 differs from the RTX 5090 policy inherited by the parent:
        XQA is faster for the 2K and 16K buckets, while the staged BF16 path is
        faster around 8K for the measured K/tail policy.
        """
        end_pos = int(end_pos)
        if end_pos < 256 and q_seq is not None and int(q_seq) <= 7:
            return True
        if end_pos < 6144:
            return True
        if end_pos < 12288:
            return False
        return True

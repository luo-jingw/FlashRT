"""`text_trim=True` and the consumers that describe one captured graph at
the max dims: `runtime_surface()`, `pipeline_resources()` and
`export_model_runtime()` refuse a trimmed frontend with a `ValueError`
(opportunities.md OPT-030, "Constraints on consumers of a trimmed
frontend"). Without the refusal, after a trimmed `set_prompt`:

- `runtime_surface()` would report `context_rows = dims["x0"]` (the
  buffer size) and the max dims in the setup identity;
- `pipeline_resources()` would pair the max dims with the trimmed RoPE
  table, which has fewer rows (an out-of-bounds read by the native
  pipeline);
- the export would adopt one length's graph while its prompt verb can
  switch the frontend to another length's graph.

Each guard is checked on a trimmed frontend, with the untrimmed frontend
of the same dims as the control.
"""
from __future__ import annotations

import pytest
import torch

from flash_rt.frontends.torch.imagewam_thor import ImageWAMTorchFrontendThor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TEXT_LEN = 16
DIMS = dict(x0=TEXT_LEN + 1, a0=TEXT_LEN + 1 + 10, total=TEXT_LEN + 1 + 10 + 4,
            joint_attention_dim=16, proprio_dim=3)


def _frontend(text_trim: bool) -> ImageWAMTorchFrontendThor:
    torch.manual_seed(0)
    fe = ImageWAMTorchFrontendThor(precision="fp16", dims_override=dict(DIMS), text_trim=text_trim)
    mask = torch.zeros(TEXT_LEN, dtype=torch.bool)
    mask[:5] = True
    fe.set_prompt(context=torch.randn(TEXT_LEN, DIMS["joint_attention_dim"]).to(torch.bfloat16),
                  context_mask=mask)
    return fe


def test_runtime_surface_refuses_text_trim():
    with pytest.raises(ValueError, match="text_trim=True"):
        _frontend(True).runtime_surface()
    surface = _frontend(False).runtime_surface()
    assert surface.context_rows == DIMS["x0"]


def test_pipeline_resources_refuses_text_trim():
    with pytest.raises(ValueError, match="text_trim=True"):
        _frontend(True).pipeline_resources()
    resources = _frontend(False).pipeline_resources()
    assert resources.dims.x0 == DIMS["x0"]


def test_export_model_runtime_refuses_text_trim():
    pytest.importorskip("flash_rt.runtime.exec", exc_type=ImportError)
    pytest.importorskip("flash_rt.runtime.export", exc_type=ImportError)
    with pytest.raises(ValueError, match="text_trim=True"):
        _frontend(True).export_model_runtime()
    runtime = _frontend(False).export_model_runtime()
    assert runtime is not None

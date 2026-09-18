"""`blockscaled_ref.py`: Hadamard rotation math (exact in fp32), element
grids, UE4M3 scale rules, nibble packing, and the CUTLASS scale-factor
byte layout -- at ImageWAM's real GEMM widths. Runs anywhere torch
runs; the bit-exact comparison against the CUDA quantizers lives in
`tests/test_imagewam_e0m3_hadamard.py` (Thor only)."""
import pytest
import torch

from flash_rt.models.imagewam.blockscaled_ref import (
    BLOCK,
    decode_codes,
    dequantize_blocks,
    e0m3_round,
    e2m1_round,
    fwht16_butterfly,
    hadamard_matrix,
    pack_codes,
    pack_scales,
    quantize_blocks,
    rotate_k_blocks,
    sf_offsets,
    sf_size_bytes,
    ue4m3_round,
    unpack_codes,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
# Every K an ImageWAM GEMM reads (3072 hidden, 9216 mlp, 1024/4096
# ActionDiT, 7680 txt_in, 12288 merged linear2).
REAL_K = (3072, 9216, 1024, 4096, 7680, 12288)


def _outlier_matrix(rows: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(rows, k, generator=g)
    x[:, torch.randint(0, k, (max(1, k // 256),), generator=g)] *= 40.0
    return x.to(DEV)


@pytest.mark.parametrize("n", [16, 32, 64, 128, 256, 512])
def test_hadamard_orthonormal(n):
    h = hadamard_matrix(n, device=DEV)
    err = (h @ h.t() - torch.eye(n, device=DEV)).abs().max().item()
    sym = (h - h.t()).abs().max().item()
    print(f"H{n}: max|HH^T-I|={err:.3e} max|H-H^T|={sym:.1e}")
    assert err < 1e-6 and sym == 0.0


@pytest.mark.parametrize("k", REAL_K)
@pytest.mark.parametrize("block", [16, 128, 512])
def test_block_rotation_preserves_gemm(k, block):
    """<Hx, Hw> = <x, w> for the block-diagonal rotation, fp32 math."""
    if k % block:
        pytest.skip(f"K={k} not a multiple of {block}")
    x = _outlier_matrix(64, k, 0)
    w = _outlier_matrix(96, k, 1) * 0.03
    ref = (x.double() @ w.double().t())
    rot = rotate_k_blocks(x, block) @ rotate_k_blocks(w, block).t()
    plain = x.float() @ w.float().t()
    diff = (rot.double() - ref)
    rel_l2 = (diff.norm() / ref.norm()).item()
    plain_rel = ((plain.double() - ref).norm() / ref.norm()).item()
    print(f"K={k} block={block}: rotated fp32 max_abs={diff.abs().max().item():.3e} rel_l2={rel_l2:.3e} "
          f"(unrotated fp32 GEMM rel_l2={plain_rel:.3e})")
    # Same order as fp32 accumulation noise of the unrotated GEMM.
    assert rel_l2 < 5e-6
    # Rotation is an involution (symmetric orthonormal H).
    back = rotate_k_blocks(rotate_k_blocks(x, block), block)
    assert (back - x).abs().max().item() < 1e-4 * x.abs().max().item()


def test_fwht16_butterfly_matches_matrix():
    x = _outlier_matrix(128, 3072, 2)
    a = fwht16_butterfly(x)
    b = rotate_k_blocks(x, 16)
    err = (a - b).abs().max().item()
    print(f"butterfly vs H16 matrix: max_abs={err:.3e} (|x|max={x.abs().max().item():.1f})")
    assert err < 1e-5 * x.abs().max().item()


def test_ue4m3_round():
    x = torch.tensor([0.0, 1e-12, 2.0 ** -9, 1.5 * 2.0 ** -9, 2.0 ** -6, 0.0137, 1.0625, 460.0, 1e9, -3.0], device=DEV)
    got = ue4m3_round(x).tolist()
    # 1.5*2^-9 ties to even (2^-8); 0.0137 -> subnormal 7*2^-9;
    # 1.0625 ties to even (1.0); saturates at 448; negatives clamp to 0.
    want = [0.0, 0.0, 2.0 ** -9, 2.0 ** -8, 2.0 ** -6, 7 * 2.0 ** -9, 1.0, 448.0, 448.0, 0.0]
    print("ue4m3:", got)
    assert got == want


def test_element_grids():
    v = torch.tensor([0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.01, 99.0, -0.2, -2.6], device=DEV)
    assert e2m1_round(v).tolist() == [0.0, 0.5, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 6.0, -0.0, -3.0]
    u = torch.tensor([0.5, 1.5, 2.5, 6.49, 6.5, 7.6, -0.4, -3.5, -9.0], device=DEV)
    got = e0m3_round(u).tolist()
    print("e0m3:", got)
    assert got == [0.0, 2.0, 2.0, 6.0, 6.0, 7.0, 0.0, -4.0, -7.0]
    codes = quantize_blocks(-0.01 * torch.ones(1, 16, device=DEV), "e0m3").codes
    assert int(codes.max()) <= 0x0F


@pytest.mark.parametrize("fmt,rule", [("e2m1", "amax"), ("e2m1", "mse"), ("e0m3", "amax")])
def test_quantize_blocks_properties(fmt, rule):
    x = _outlier_matrix(256, 3072, 3) * 0.05
    q = quantize_blocks(x, fmt, rule)
    qmax = 6.0 if fmt == "e2m1" else 7.0
    assert q.codes.dtype == torch.uint8 and int(q.codes.max()) <= 0x0F
    assert torch.equal(q.scale_bytes.view(torch.float8_e4m3fn).float(), q.scales)
    vals = decode_codes(q.codes, fmt)
    assert float(vals.abs().max()) <= qmax
    deq = dequantize_blocks(q)
    rel = ((deq - x.float()).norm() / x.float().norm()).item()
    print(f"{fmt}/{rule}: rel_l2 quant error on N(0,.05)+outliers = {rel:.4f}")
    assert rel < 0.25


def test_zero_scale_blocks_encode_like_kernels():
    x = torch.zeros(2, 32, device=DEV)
    x[0, :16] = 1e-4
    x[0, 3] = -1e-4
    x[1, 16:] = 1.0
    for fmt, zero_code in (("e2m1", 7), ("e0m3", 0)):
        q = quantize_blocks(x, fmt)
        assert float(q.scales[0, 0]) == 0.0 and float(q.scales[0, 1]) == 0.0
        assert float(dequantize_blocks(q)[0].abs().max()) == 0.0
        # All-zero block: 0 * inf = NaN -> e2m1 falls through to 6, e0m3 converts to 0.
        assert int(q.codes[0, 16]) == zero_code


def test_pack_codes_nibble_order_and_roundtrip():
    codes = torch.randint(0, 16, (8, 64), dtype=torch.uint8, device=DEV)
    packed = pack_codes(codes)
    assert packed.shape == (8, 32)
    assert int(packed[0, 0]) == int(codes[0, 0]) | (int(codes[0, 1]) << 4)
    assert torch.equal(unpack_codes(packed), codes)


@pytest.mark.parametrize("rows,k", [(9216, 3072), (3072, 9216), (27648, 3072), (1024, 4096),
                                    (17408, 1024), (905, 3072), (513, 3072), (392, 9216), (64, 1024),
                                    (64, 4096), (3, 7680 + 16)])
def test_sf_offsets_layout(rows, k):
    off = sf_offsets(rows, k, device=DEV)
    size = sf_size_bytes(rows, k)
    flat = off.reshape(-1)
    assert int(flat.min()) >= 0 and int(flat.max()) < size
    assert torch.unique(flat).numel() == flat.numel()
    # Hand-derived entries of the ((32,4),(16,4)):((16,4),(0,1)) atom.
    assert int(off[0, 0]) == 0
    assert int(off[1, 0]) == 16
    assert int(off[0, 1]) == 1
    if rows > 32:
        assert int(off[32, 0]) == 4
    if k >= 128:
        assert int(off[0, 4]) == 512
    if rows > 128:
        assert int(off[128, 0]) == 512 * -(-k // 64)
    print(f"rows={rows} K={k}: sf bytes={size} used={flat.numel()}")


def test_pack_scales_places_every_byte():
    rows, k = 200, 3072
    sb = torch.randint(1, 255, (rows, k // BLOCK), dtype=torch.uint8, device=DEV)
    buf = pack_scales(sb)
    off = sf_offsets(rows, k, device=DEV)
    assert torch.equal(buf[off.reshape(-1)].reshape(rows, k // BLOCK), sb)
    assert int((buf != 0).sum()) == rows * (k // BLOCK)


def test_scale_rule_division_semantics():
    """E0M3 divides by 7 exactly (`__fdiv_rn`): amax = 175/128 gives the
    exact tie 25/128, which rounds to the even UE4M3 value 0.1875. NVFP4
    multiplies by fp32(1/6) (fast-math lowering of `amax / 6.f`)."""
    x = torch.zeros(1, 16, device=DEV)
    x[0, 0] = 1.3671875
    assert float(quantize_blocks(x, "e0m3").scales[0, 0]) == 0.1875
    x6 = torch.full((1, 16), 0.1, device=DEV)
    want = ue4m3_round(torch.tensor([0.1], device=DEV) * torch.tensor(1.0 / 6.0, device=DEV))
    assert float(quantize_blocks(x6, "e2m1").scales[0, 0]) == float(want[0])

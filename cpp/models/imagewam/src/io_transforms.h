// Host-side ImageWAM IO transforms shared by the native verbs.
//
// Each function reproduces, bit for bit, the float32 torch sequence the
// Python frontend runs (flash_rt/models/imagewam/dataset_stats.py and
// ImageWAMTorchFrontendThor.stage_proprio / read_actions). The translation
// unit is compiled with floating-point contraction disabled so `x * s + o`
// stays two roundings, as it is in torch.
#ifndef FLASHRT_CPP_MODELS_IMAGEWAM_IO_TRANSFORMS_H
#define FLASHRT_CPP_MODELS_IMAGEWAM_IO_TRANSFORMS_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace flashrt {
namespace models {
namespace imagewam {

// One dataset field's min/max affine map (`scale`, `offset` per channel).
// Empty vectors mean identity.
struct AffineField {
    std::vector<float> scale;
    std::vector<float> offset;

    bool identity() const { return scale.empty(); }
};

// torch: clamp(x * scale + offset, -5, 5), channel-wise over `n` values;
// a copy when `field` is identity (no normalizer loaded, no clamp).
void normalize_state(const float* raw, const AffineField& field, float* out, std::size_t n);

// torch: (x - offset) / scale, in place over `rows` x `cols` values.
void denormalize_actions_inplace(float* values, const AffineField& field,
                                 std::size_t rows, std::size_t cols);

// float32 -> bfloat16 bits, round to nearest even (c10::BFloat16; NaN -> 0x7FC0).
std::uint16_t float_to_bf16_rne(float x);

}  // namespace imagewam
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_IMAGEWAM_IO_TRANSFORMS_H

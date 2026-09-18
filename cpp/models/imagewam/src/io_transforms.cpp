#include "io_transforms.h"

#include <cmath>
#include <cstring>

namespace flashrt {
namespace models {
namespace imagewam {

namespace {
constexpr float kStateClamp = 5.0f;
}  // namespace

void normalize_state(const float* raw, const AffineField& field, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        float x = raw[i];
        if (!field.identity()) {
            const float scaled = x * field.scale[i];
            x = scaled + field.offset[i];
            // torch.clamp keeps NaN; the comparisons below do too.
            if (x < -kStateClamp) x = -kStateClamp;
            if (x > kStateClamp) x = kStateClamp;
        }
        out[i] = x;
    }
}

void denormalize_actions_inplace(float* values, const AffineField& field,
                                 std::size_t rows, std::size_t cols) {
    if (field.identity()) return;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            float& v = values[r * cols + c];
            const float shifted = v - field.offset[c];
            v = shifted / field.scale[c];
        }
    }
}

std::uint16_t float_to_bf16_rne(float x) {
    if (std::isnan(x)) return 0x7FC0u;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    const std::uint32_t rounding_bias = ((bits >> 16) & 1u) + 0x7FFFu;
    return static_cast<std::uint16_t>((bits + rounding_bias) >> 16);
}

}  // namespace imagewam
}  // namespace models
}  // namespace flashrt

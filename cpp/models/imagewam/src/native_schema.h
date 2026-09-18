// The `io="native"` model-runtime schema of ImageWAM: the ports, region and
// stage the native verbs implement. Rendered in the runtime builder's
// canonical identity record format so a producer-built declaration can be
// compared with it line for line (tests/gate_imagewam_native_schema_parity.py).
#ifndef FLASHRT_CPP_MODELS_IMAGEWAM_NATIVE_SCHEMA_H
#define FLASHRT_CPP_MODELS_IMAGEWAM_NATIVE_SCHEMA_H

#include <cstdint>
#include <string>
#include <vector>

namespace flashrt {
namespace models {
namespace imagewam {

// Declaration buffer order of the export: img_raw, context, action_latent.
enum class NativeBuffer : int64_t { kNone = -1, kImgRaw = 0, kContext = 1, kActionLatent = 2 };

enum class NativePort { kImageTokens, kProprio, kNoise, kActions, kActionsRaw };

struct NativePortSpec {
    NativePort role;
    std::string name;
    uint32_t modality;
    uint32_t dtype;
    uint32_t layout;
    uint32_t direction;
    uint32_t update;
    uint32_t required;
    std::vector<int64_t> shape;
    NativeBuffer buffer;
    uint64_t bytes;
};

struct NativeDims {
    uint32_t img_len;
    uint32_t token_dim;
    uint32_t num_action;
    uint32_t action_dim;
    uint32_t proprio_dim;  // 0 = no proprio port
};

struct NativeSchema {
    std::vector<NativePortSpec> ports;  // declaration order
    uint64_t region_bytes;              // rollout_boundary = the action_latent window
};

NativeSchema build_native_schema(const NativeDims& dims);

// region / port / stage records, one per line, in builder order.
std::string render_schema_records(const NativeSchema& schema);

}  // namespace imagewam
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_IMAGEWAM_NATIVE_SCHEMA_H

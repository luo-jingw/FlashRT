#include "native_schema.h"

#include "flashrt/model_runtime.h"
#include "flashrt/runtime.h"

#include <cstdio>
#include <string>

namespace flashrt {
namespace models {
namespace imagewam {

NativeSchema build_native_schema(const NativeDims& d) {
    const uint64_t token_bytes = uint64_t(d.img_len) * d.token_dim * 2;
    const uint64_t chunk_bytes = uint64_t(d.num_action) * d.action_dim * 4;
    const std::vector<int64_t> chunk = {int64_t(d.num_action), int64_t(d.action_dim)};
    NativeSchema s;
    s.ports.push_back({NativePort::kImageTokens, "image_tokens", FRT_RT_MOD_TENSOR,
                       FRT_RT_DTYPE_BF16, FRT_RT_LAYOUT_FLAT, FRT_RT_PORT_IN, FRT_RT_PORT_SWAP, 1,
                       {int64_t(d.img_len), int64_t(d.token_dim)}, NativeBuffer::kImgRaw,
                       token_bytes});
    if (d.proprio_dim > 0) {
        s.ports.push_back({NativePort::kProprio, "proprio", FRT_RT_MOD_STATE, FRT_RT_DTYPE_F32,
                           FRT_RT_LAYOUT_FLAT, FRT_RT_PORT_IN, FRT_RT_PORT_STAGED, 1,
                           {int64_t(d.proprio_dim)}, NativeBuffer::kNone, 0});
    }
    s.ports.push_back({NativePort::kNoise, "noise", FRT_RT_MOD_TENSOR, FRT_RT_DTYPE_F32,
                       FRT_RT_LAYOUT_FLAT, FRT_RT_PORT_IN, FRT_RT_PORT_SWAP, 1, chunk,
                       NativeBuffer::kActionLatent, chunk_bytes});
    s.ports.push_back({NativePort::kActions, "actions", FRT_RT_MOD_ACTION, FRT_RT_DTYPE_F32,
                       FRT_RT_LAYOUT_FLAT, FRT_RT_PORT_OUT, FRT_RT_PORT_STAGED, 0, chunk,
                       NativeBuffer::kNone, chunk_bytes});
    s.ports.push_back({NativePort::kActionsRaw, "actions_raw", FRT_RT_MOD_TENSOR,
                       FRT_RT_DTYPE_F32, FRT_RT_LAYOUT_FLAT, FRT_RT_PORT_OUT, FRT_RT_PORT_SWAP, 0,
                       chunk, NativeBuffer::kActionLatent, chunk_bytes});
    s.region_bytes = chunk_bytes;
    return s;
}

std::string render_schema_records(const NativeSchema& s) {
    std::string out;
    char line[256];
    std::snprintf(line, sizeof(line), "region:0:rollout_boundary:0:%llu:%u\n",
                  static_cast<unsigned long long>(s.region_bytes),
                  unsigned(FRT_RT_REGION_SNAPSHOT | FRT_RT_REGION_RESTORE));
    out += line;
    for (size_t i = 0; i < s.ports.size(); ++i) {
        const NativePortSpec& p = s.ports[i];
        std::snprintf(line, sizeof(line), "port:%zu:%s:%u:%u:%u:%u:%u:%u:", i, p.name.c_str(),
                      p.modality, p.dtype, p.layout, p.direction, p.update, p.required);
        out += line;
        for (size_t k = 0; k < p.shape.size(); ++k) {
            std::snprintf(line, sizeof(line), "%s%lld", k ? "," : "",
                          static_cast<long long>(p.shape[k]));
            out += line;
        }
        std::snprintf(line, sizeof(line), ":%lld:0:%llu\n", static_cast<long long>(p.buffer),
                      static_cast<unsigned long long>(p.bytes));
        out += line;
    }
    out += "stage:0:0:\n";
    return out;
}

}  // namespace imagewam
}  // namespace models
}  // namespace flashrt

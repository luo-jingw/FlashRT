// ================================================================
// FlashRT — fused Euler-step + fp32→fp16 cast, ImageWAM denoise
// step boundary (standalone proof, not wired into pipeline_thor.py)
//
// Replaces the adjacent producer/consumer pair that crosses every
// non-final ImageWAM denoise step boundary:
//
//   step i   (end):   fvk.gpu_euler_step(action_latent, velocity, ...)
//   step i+1 (start): fvk.gpu_cast_fp32_to_fp16(action_latent, action_latent_fp16, ...)
//
// with ONE kernel launch that does both in a single pass: writes the
// updated fp32 `action_latent` (Euler math, unchanged) AND, in the
// same thread, emits the fp16 cast of that SAME just-written value
// into a second output buffer.
//
// Math matches `euler_step_kernel` + `cast_fp32_fp16_kernel`
// (csrc/kernels/elementwise.cu) bit-for-bit:
//   new_val   = actions[idx] + dt * __half2float(velocity[vel_offset + idx]);
//   actions[idx]      = new_val;                 // == gpu_euler_step's write
//   actions_fp16[idx] = __float2half(new_val);    // == gpu_cast_fp32_to_fp16 reading that same new_val back
// No clamping in either original kernel, so none is added here.
// ================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void fused_euler_step_and_cast_kernel(
    float* __restrict__ actions,
    const __half* __restrict__ velocity,
    __half* __restrict__ actions_fp16_out,
    float dt, int n, int vel_elem_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float new_val = actions[idx] + dt * __half2float(velocity[vel_elem_offset + idx]);
    actions[idx] = new_val;
    actions_fp16_out[idx] = __float2half(new_val);
}

void fused_euler_step_and_cast(
    float* actions, const __half* velocity, __half* actions_fp16_out,
    int T, int action_dim, float dt, int vel_elem_offset,
    cudaStream_t stream) {
    int n = T * action_dim;
    fused_euler_step_and_cast_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        actions, velocity, actions_fp16_out, dt, n, vel_elem_offset);
}

// ---- Torch-tensor entry point for the standalone JIT test ----
void fused_euler_step_and_cast_torch(
    torch::Tensor actions, torch::Tensor velocity, torch::Tensor actions_fp16_out,
    int64_t T, int64_t action_dim, double dt, int64_t vel_elem_offset) {
    TORCH_CHECK(actions.is_cuda() && velocity.is_cuda() && actions_fp16_out.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(actions.scalar_type() == torch::kFloat32, "actions must be fp32");
    TORCH_CHECK(velocity.scalar_type() == torch::kFloat16, "velocity must be fp16");
    TORCH_CHECK(actions_fp16_out.scalar_type() == torch::kFloat16, "actions_fp16_out must be fp16");
    TORCH_CHECK(actions.is_contiguous() && velocity.is_contiguous() && actions_fp16_out.is_contiguous(),
                "all tensors must be contiguous");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_euler_step_and_cast(
        actions.data_ptr<float>(),
        reinterpret_cast<const __half*>(velocity.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(actions_fp16_out.data_ptr<at::Half>()),
        static_cast<int>(T), static_cast<int>(action_dim),
        static_cast<float>(dt), static_cast<int>(vel_elem_offset), stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_euler_step_and_cast", &fused_euler_step_and_cast_torch,
          "Fused Euler step (fp32 in-place) + fp32->fp16 cast into a second buffer");
}

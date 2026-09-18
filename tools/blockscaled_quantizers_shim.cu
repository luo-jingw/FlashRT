// C entry points over the repository's block-scaled quantizers, for
// tools/check_blockscaled_quantizers_sm90.py. The tool compiles this file
// together with the unmodified quantizer sources for a non-Blackwell GPU
// (the quantizers use no tcgen05 instructions) and compares their bytes
// with flash_rt/models/imagewam/blockscaled_ref.py. Each call synchronizes.
#include <cstdint>
#include <cuda_runtime.h>
#include "quantize/quantize_e0m3_sfa.cuh"
#include "quantize/quantize_fp4_sfa.cuh"
#include "fused_fp4/pi05_e0m3_act.cuh"
extern "C" {
int shim_e0m3_w(uintptr_t s, uintptr_t p, uintptr_t f, int N, int D, int sfb) {
  int rc = flash_rt::fp4::quantize_e0m3_dynamic_sfa_fp16((const void*)s, (void*)p, (void*)f, N, D, sfb != 0, 0);
  cudaDeviceSynchronize(); return rc; }
int shim_e0m3_vec(uintptr_t s, uintptr_t p, uintptr_t f, int N, int D, int rht) {
  int rc = flash_rt::fp4::quantize_e0m3_dynamic_sfa_fp16_vec((const void*)s, (void*)p, (void*)f, N, D, false, rht, 0);
  cudaDeviceSynchronize(); return rc; }
int shim_nvfp4(uintptr_t s, uintptr_t p, uintptr_t f, int N, int D, int sfb) {
  int rc = flash_rt::fp4::quantize_fp4_dynamic_sfa_fp16((const void*)s, (void*)p, (void*)f, N, D, sfb != 0, 0);
  cudaDeviceSynchronize(); return rc; }
int shim_nvfp4_mse(uintptr_t s, uintptr_t p, uintptr_t f, int N, int D, int sfb) {
  int rc = flash_rt::fp4::quantize_fp4_dynamic_sfa_mse_fp16((const void*)s, (void*)p, (void*)f, N, D, sfb != 0, 0);
  cudaDeviceSynchronize(); return rc; }
}

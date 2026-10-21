// NVFP4 (W4A4) SM120 CUTLASS GEMM tile / operand-order variants at ImageWAM ActionDiT shapes.
//
// Every variant computes D (M_tok x N_out, row-major) = alpha * A_act(M_tok x K) * W(N_out x K)^T
// from the same packed FP4 operands and swizzled UE4M3 scale factors that the
// production path uses (`quantize_bf16_to_nvfp4_swizzled` / `bf16_weight_to_nvfp4_swizzled`
// layout = CUTLASS Sm1xxBlockScaledConfig<16>).
//   normal : A = activation, B = weight, D row-major.
//   swap   : A = weight, B = activation (D' = W A^T, N_out x M_tok), D' column-major,
//            which is D row-major.
// Output: max |D_variant - D_reference| over all elements (reference = production
// pingpong 128x128x256, bf16 out) and mean kernel time over ITERS back-to-back launches.
//
// build: nvcc -std=c++17 -O3 --expt-relaxed-constexpr -gencode=arch=compute_120a,code=sm_120a
//        -I<cutlass>/include -I<cutlass>/tools/util/include nvfp4_tile_bench.cu -o nvfp4_tile_bench
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)

using ElementSF = cutlass::float_ue4m3_t;
using ElementPair = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;

template <class TileShape_, class Schedule_, class LayoutD_, class ElementD_, class Sched_ = cutlass::gemm::PersistentScheduler,
          int Splits_ = 0>
struct Cfg {
  static constexpr int Splits = Splits_;  // 0 = scheduler default; >0 = stream-K scheduler in SplitK mode
  using TileShape = TileShape_;
  using ClusterShape = Shape<_1, _1, _1>;
  using ElementD = ElementD_;
  using ElementC = ElementD_;
  using LayoutD = LayoutD_;
  static constexpr int AlignA = 32, AlignB = 32;
  static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      ElementC, LayoutD, AlignD, ElementD, LayoutD, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementPair, cutlass::layout::RowMajor, AlignA,
      ElementPair, cutlass::layout::ColumnMajor, AlignB, float, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
      Schedule_>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, Sched_>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

struct Problem {
  const void *act, *act_sf, *w, *w_sf;
  void* D;
  int M, N, K;  // M = tokens, N = output features
  float alpha;
};

template <class C>
struct Runner {
  typename C::Gemm gemm;
  typename C::Gemm::Arguments saved;
  void* ws = nullptr;
  bool ok = false;
  static constexpr bool kStreamK = !std::is_same_v<typename C::Kernel::TileSchedulerTag, cutlass::gemm::PersistentScheduler>;

  bool init(const Problem& p, bool swap, cudaStream_t s) {
    using G = typename C::Gemm;
    using StrideA = typename G::GemmKernel::StrideA;
    using StrideB = typename G::GemmKernel::StrideB;
    using StrideD = typename G::GemmKernel::StrideD;
    int M = swap ? p.N : p.M, N = swap ? p.M : p.N, K = p.K;
    const void* A = swap ? p.w : p.act;
    const void* SFA = swap ? p.w_sf : p.act_sf;
    const void* B = swap ? p.act : p.w;
    const void* SFB = swap ? p.act_sf : p.w_sf;
    auto shp = make_shape(M, N, K, 1);
    auto lsfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(shp);
    auto lsfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(shp);
    using AA = typename G::GemmKernel::CollectiveMainloop::ArrayElementA;
    using AB = typename G::GemmKernel::CollectiveMainloop::ArrayElementB;
    StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));
    typename G::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        {reinterpret_cast<AA const*>(A), cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1)),
         reinterpret_cast<AB const*>(B), cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1)),
         reinterpret_cast<ElementSF const*>(SFA), lsfa, reinterpret_cast<ElementSF const*>(SFB), lsfb},
        {{p.alpha, 0.0f}, nullptr, sd, reinterpret_cast<typename C::ElementD*>(p.D), sd}};
    if constexpr (C::Splits > 0) {
      args.scheduler.splits = C::Splits;
      args.scheduler.decomposition_mode = decltype(args.scheduler.decomposition_mode)::SplitK;
    }
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    wsz = G::get_workspace_size(args);
    if (wsz) CK(cudaMalloc(&ws, wsz));
    saved = args;
    if (gemm.initialize(args, ws, s) != cutlass::Status::kSuccess) return false;
    ok = true;
    return true;
  }
  // stream-K: the reduction workspace is re-initialized (zeroed) before every launch
  size_t wsz = 0;
  // stream-K: the reduction workspace is zeroed (one memset) before every launch
  void run(cudaStream_t s) { if constexpr (kStreamK) { if (wsz) cudaMemsetAsync(ws, 0, wsz, s); } gemm.run(s); }
  ~Runner() { if (ws) cudaFree(ws); }
};

using RM = cutlass::layout::RowMajor;
using CM = cutlass::layout::ColumnMajor;
using PP = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using CO = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using BF = cutlass::bfloat16_t;
using HF = cutlass::half_t;

// production reference + candidates
using SK = cutlass::gemm::StreamKScheduler;
using V_ref      = Cfg<Shape<_128, _128, _256>, PP, RM, BF>;            // production default (pingpong)
using V_coop     = Cfg<Shape<_128, _128, _256>, CO, RM, BF>;            // production "plain"
using V_sk       = Cfg<Shape<_128, _128, _256>, CO, RM, BF, SK>;        // stream-K, heuristic decomposition
using V_split2   = Cfg<Shape<_128, _128, _256>, CO, RM, BF, SK, 2>;
using V_split4   = Cfg<Shape<_128, _128, _256>, CO, RM, BF, SK, 4>;
using V_split8   = Cfg<Shape<_128, _128, _256>, CO, RM, BF, SK, 8>;
using V_split16  = Cfg<Shape<_128, _128, _256>, CO, RM, BF, SK, 16>;
using V_split8_f16 = Cfg<Shape<_128, _128, _256>, CO, RM, HF, SK, 8>;
using V_swap     = Cfg<Shape<_128, _128, _256>, PP, CM, BF>;            // swap-AB, same tile
using V_ref_f16  = Cfg<Shape<_128, _128, _256>, PP, RM, HF>;            // production tile, fp16 output
using V_sk_f16   = Cfg<Shape<_128, _128, _256>, CO, RM, HF, SK>;

template <class C>
void bench(const char* name, bool swap, const Problem& p0, const std::vector<float>& ref, int iters, cudaStream_t s,
           FILE* out) {
  Problem p = p0;
  size_t n = (size_t)p.M * p.N;
  size_t bytes = n * sizeof(typename C::ElementD);
  CK(cudaMalloc(&p.D, bytes));
  CK(cudaMemset(p.D, 0, bytes));
  Runner<C> r;
  if (!r.init(p, swap, s)) {
    std::fprintf(out, "{\"variant\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"ok\":false}\n", name, p.M, p.N, p.K);
    cudaFree(p.D);
    return;
  }
  for (int i = 0; i < 10; ++i) r.run(s);
  CK(cudaStreamSynchronize(s));
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  // time a CUDA graph of `iters` back-to-back launches (no host enqueue cost)
  cudaGraph_t graph; cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < iters; ++i) r.run(s);
  CK(cudaStreamEndCapture(s, &graph));
  CK(cudaGraphInstantiate(&exec, graph, 0));
  CK(cudaGraphLaunch(exec, s)); CK(cudaStreamSynchronize(s));
  cudaEventRecord(e0, s);
  CK(cudaGraphLaunch(exec, s));
  cudaEventRecord(e1, s);
  CK(cudaEventSynchronize(e1));
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  CK(cudaGetLastError());
  std::vector<uint16_t> h(n);
  CK(cudaMemcpy(h.data(), p.D, bytes, cudaMemcpyDeviceToHost));
  double maxd = 0, maxr = 0;
  for (size_t i = 0; i < n; ++i) {
    float v;
    if constexpr (std::is_same_v<typename C::ElementD, BF>) {
      uint32_t u = (uint32_t)h[i] << 16; std::memcpy(&v, &u, 4);
    } else {
      __half_raw hr; hr.x = h[i]; v = __half2float(__half(hr));
    }
    if (!ref.empty()) maxd = std::max(maxd, (double)std::fabs(v - ref[i]));
    maxr = std::max(maxr, (double)std::fabs(v));
  }
  double us = 1000.0 * ms / iters;
  double wbytes = (double)p.N * p.K / 2 + (double)p.N * p.K / 16;  // packed weight + SF
  std::fprintf(out, "{\"variant\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"ok\":true,\"us\":%.3f,"
                    "\"weight_GBps\":%.1f,\"max_abs_diff_vs_ref\":%.6g,\"max_abs_out\":%.6g}\n",
               name, p.M, p.N, p.K, us, wbytes / (us * 1e-6) / 1e9, maxd, maxr);
  std::fflush(out);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
  cudaFree(p.D);
}

static size_t sf_bytes(int rows, int K) {  // swizzled SF bytes, 128-row x 4-block atoms
  int nrb = (rows + 127) / 128, ncb = (K / 16 + 3) / 4;
  return (size_t)nrb * ncb * 512;
}

int main(int argc, char** argv) {
  int iters = argc > 1 ? std::atoi(argv[1]) : 500;
  bool splitk_only = argc > 2 && std::string(argv[2]) == "splitk";
  bool proj_mode = argc > 2 && std::string(argv[2]) == "proj";
  std::vector<std::pair<int, int>> NK = {{9216, 1024}, {1024, 3072}, {8192, 1024}, {1024, 4096},
                                         {17408, 1024}, {1024, 7168}};
  std::vector<int> Ms = {64, 128, 256, 640};
  if (proj_mode) Ms = {64, 128, 192, 256, 320, 384, 448, 512, 576, 640};
  cudaStream_t s; CK(cudaStreamCreate(&s));
  std::mt19937 rng(0);
  for (auto [N, K] : NK) {
    if (splitk_only && N != 1024) continue;
    for (int M : Ms) {
      size_t a_b = (size_t)M * K / 2, w_b = (size_t)N * K / 2;
      size_t asf = sf_bytes(M, K), wsf = sf_bytes(N, K);
      std::vector<uint8_t> ha(a_b), hw(w_b), hasf(asf), hwsf(wsf);
      for (auto& x : ha) x = rng() & 0xFF;
      for (auto& x : hw) x = rng() & 0xFF;
      // UE4M3 scale bytes in [0.5, 2): exponent 6..7 -> bytes 0x30..0x3F
      for (auto& x : hasf) x = 0x30 + (rng() & 0x0F);
      for (auto& x : hwsf) x = 0x30 + (rng() & 0x0F);
      void *da, *dw, *dasf, *dwsf;
      CK(cudaMalloc(&da, a_b)); CK(cudaMalloc(&dw, w_b)); CK(cudaMalloc(&dasf, asf)); CK(cudaMalloc(&dwsf, wsf));
      CK(cudaMemcpy(da, ha.data(), a_b, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dw, hw.data(), w_b, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dasf, hasf.data(), asf, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dwsf, hwsf.data(), wsf, cudaMemcpyHostToDevice));
      Problem p{da, dasf, dw, dwsf, nullptr, M, N, K, 1.0f / 64};
      // reference output
      std::vector<float> ref;
      {
        Problem q = p; size_t n = (size_t)M * N;
        CK(cudaMalloc(&q.D, n * 2));
        Runner<V_ref> r;
        if (!r.init(q, false, s)) { std::fprintf(stderr, "ref init failed M=%d N=%d K=%d\n", M, N, K); return 1; }
        r.run(s); CK(cudaStreamSynchronize(s));
        std::vector<uint16_t> h(n); CK(cudaMemcpy(h.data(), q.D, n * 2, cudaMemcpyDeviceToHost));
        ref.resize(n);
        for (size_t i = 0; i < n; ++i) { uint32_t u = (uint32_t)h[i] << 16; std::memcpy(&ref[i], &u, 4); }
        cudaFree(q.D);
      }
      if (proj_mode) {
        bench<V_ref>("ref_pp_128x128x256", false, p, ref, iters, s, stdout);
        bench<V_coop>("coop_128x128x256", false, p, ref, iters, s, stdout);
        bench<V_sk>("streamK_coop", false, p, ref, iters, s, stdout);
        if (N == 1024) {
          bench<V_split2>("splitK2_coop", false, p, ref, iters, s, stdout);
          bench<V_split4>("splitK4_coop", false, p, ref, iters, s, stdout);
        }
        cudaFree(da); cudaFree(dw); cudaFree(dasf); cudaFree(dwsf);
        continue;
      }
      if (splitk_only) {
        bench<V_ref>("ref_pp_128x128x256", false, p, ref, iters, s, stdout);
        bench<V_split2>("splitK2_coop", false, p, ref, iters, s, stdout);
        bench<V_split4>("splitK4_coop", false, p, ref, iters, s, stdout);
        bench<V_split8>("splitK8_coop", false, p, ref, iters, s, stdout);
        bench<V_split16>("splitK16_coop", false, p, ref, iters, s, stdout);
        bench<V_split8_f16>("splitK8_coop_f16out", false, p, ref, iters, s, stdout);
        cudaFree(da); cudaFree(dw); cudaFree(dasf); cudaFree(dwsf);
        continue;
      }
      bench<V_ref>("ref_pp_128x128x256", false, p, ref, iters, s, stdout);
      bench<V_coop>("coop_128x128x256", false, p, ref, iters, s, stdout);
      bench<V_sk>("streamK_coop", false, p, ref, iters, s, stdout);
      // bench<V_split2>("splitK2_coop", false, p, ref, iters, s, stdout);
      // bench<V_split4>("splitK4_coop", false, p, ref, iters, s, stdout);
      // bench<V_split8>("splitK8_coop", false, p, ref, iters, s, stdout);
      bench<V_swap>("swapAB_pp_128x128x256", true, p, ref, iters, s, stdout);
      bench<V_ref_f16>("ref_pp_f16out", false, p, ref, iters, s, stdout);
      bench<V_sk_f16>("streamK_coop_f16out", false, p, ref, iters, s, stdout);
      cudaFree(da); cudaFree(dw); cudaFree(dasf); cudaFree(dwsf);
    }
  }
  return 0;
}

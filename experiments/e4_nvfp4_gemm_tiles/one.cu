// Standalone experiment (not part of the repository build): SM120 NVFP4
// block-scaled CUTLASS GEMM, BF16 out, same element/layout/SF config as
// csrc/gemm/fp4/cutlass_nvfp4_w4a16_gemm_sm120.cu, instantiated at several
// TileShape / schedule / scheduler combinations so a narrower N tile can be
// measured against the repository's 128x128x256 tiles. Exposed through
// extern "C" for ctypes. Returns cutlass::Status as int (0 = success).
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include <cuda_runtime.h>
#include <map>
#include <tuple>

using namespace cute;

using ElementA = cutlass::float_e2m1_t;
using ElementD = cutlass::bfloat16_t;
using ElementSF = cutlass::float_ue4m3_t;
using ElementPair = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignAB = 32;
constexpr int AlignD = 8;
using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;

template <class TileShape, class Schedule, class Scheduler>
struct GemmCfg {
  using ClusterShape = Shape<_1, _1, _1>;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      ElementD, LayoutD, AlignD, ElementD, LayoutD, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementPair, LayoutA, AlignAB, ElementPair, LayoutB, AlignAB, float,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
      Schedule>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, Scheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <class Cfg>
int run(int tag, const void* A, const void* B, void* D, int M, int N, int K,
        const void* SFA, const void* SFB, float alpha, cudaStream_t stream) {
  using Gemm = typename Cfg::Gemm;
  using SA = typename Gemm::GemmKernel::StrideA;
  using SB = typename Gemm::GemmKernel::StrideB;
  using SD = typename Gemm::GemmKernel::StrideD;
  auto pmnkl = cute::make_shape(M, N, K, 1);
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {reinterpret_cast<typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA const*>(A),
       cutlass::make_cute_packed_stride(SA{}, cute::make_shape(M, K, 1)),
       reinterpret_cast<typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB const*>(B),
       cutlass::make_cute_packed_stride(SB{}, cute::make_shape(N, K, 1)),
       reinterpret_cast<ElementSF const*>(SFA), Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(pmnkl),
       reinterpret_cast<ElementSF const*>(SFB), Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(pmnkl)},
      {{alpha, 0.0f}, nullptr, cutlass::make_cute_packed_stride(SD{}, cute::make_shape(M, N, 1)),
       reinterpret_cast<ElementD*>(D), cutlass::make_cute_packed_stride(SD{}, cute::make_shape(M, N, 1))}};
  // Per-(tag, shape) workspace, allocated on first use (outside graph capture).
  static std::map<std::tuple<int, int, int, int>, std::pair<void*, size_t>> ws;
  auto key = std::make_tuple(tag, M, N, K);
  size_t need = Gemm::get_workspace_size(args);
  auto it = ws.find(key);
  if (it == ws.end() || it->second.second < need) {
    void* p = nullptr;
    if (need) cudaMalloc(&p, need);
    ws[key] = {p, need};
  }
  Gemm gemm;
  auto st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) return 100 + static_cast<int>(st);
  st = gemm.initialize(args, ws[key].first, stream);
  if (st != cutlass::Status::kSuccess) return 200 + static_cast<int>(st);
  return static_cast<int>(gemm.run(stream));
}

using Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using Ping = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using Persist = cutlass::gemm::PersistentScheduler;
using StreamK = cutlass::gemm::StreamKScheduler;
#define DEF(name, tag, TM, TN, TK, SCHED, SCHEDULER)                                              \
  extern "C" int name(const void* A, const void* B, void* D, int M, int N, int K, const void* SFA, \
                      const void* SFB, float alpha, cudaStream_t s) {                             \
    return run<GemmCfg<Shape<_##TM, _##TN, _##TK>, SCHED, SCHEDULER>>(tag, A, B, D, M, N, K, SFA, \
                                                                       SFB, alpha, s);            \
  }
DEF(t128x64x128, 9, 128,64,128, Coop, Persist)

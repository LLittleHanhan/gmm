// ============================================================================
// Hopper (SM90a) grouped GEMM — FORWARD ONLY (path one).
//
// Extracted and trimmed from:
//   nezha/cpplus/backend/pytorch/hpc_ops/hopper_grouped_gemm.cu.cc
//
// What was kept
// -------------
//   * CUTLASS 3.x PtrArray-TmaWarpSpecializedPingpong grouped GEMM kernel
//   * Template struct ScheduleConfig<dtype, trans_a, trans_b> (+ FP32 spec)
//   * Template function CutlassGroupedGemm<trans_a, trans_b, Element>
//   * The "path one" dispatch:
//       A: [num_tokens, dim],  B: [num_experts, dim, dim_out]   ->  D: [num_tokens, dim_out]
//     for dtype = fp32 / bf16 / fp16
//
// What was removed
// ----------------
//   * CutlassGroupedGemmVariableK and all variable-K dispatch functions
//     (those implement D[i] = A_i^T @ B_i used for MoE backward).
//   * The trans_a=true explicit template instantiations (backward-only).
//   * VLOG / debug prints.
//
// The kernel is registered as torch op:  grouped_gemm::hopper_gmm_fwd
// ============================================================================

#include "hopper_gmm_forward.h"

#include <mutex>

#include "cute/tensor.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/complex.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

namespace hpc_ops_hopper {

// ---------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------
template <bool trans_a>
using GroupedGemmInputALayout =
    std::conditional_t<trans_a, ::cutlass::layout::ColumnMajor, ::cutlass::layout::RowMajor>;

template <bool trans_b>
using GroupedGemmInputBLayout =
    std::conditional_t<trans_b, ::cutlass::layout::ColumnMajor, ::cutlass::layout::RowMajor>;

using ProblemShapeType = cute::Shape<int, int, int>;
using ProblemShape = cutlass::gemm::GroupProblemShape<ProblemShapeType>;  // <M,N,K> per group

// ---------------------------------------------------------------------------
// Gemm type factory — given a ScheduleConfig, build the full Gemm adapter.
// ---------------------------------------------------------------------------
template <typename ScheduleConfig>
struct GemmGivenSchedule {
  using ElementA = typename ScheduleConfig::DataType;
  using ElementB = typename ScheduleConfig::DataType;
  using ElementC = typename ScheduleConfig::DataType;

  using LayoutA = typename ScheduleConfig::LayoutA;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  using LayoutB = typename ScheduleConfig::LayoutB;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  using LayoutC = typename ScheduleConfig::LayoutC;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;

  using TileShape = typename ScheduleConfig::TileShape;
  using ClusterShape = typename ScheduleConfig::ClusterShape;
  using KernelSchedule = typename ScheduleConfig::KernelSchedule;
  using EpilogueSchedule = typename ScheduleConfig::EpilogueSchedule;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC*, AlignmentC, ElementC, LayoutC*, AlignmentC, EpilogueSchedule,
      cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutA*, AlignmentA, ElementB, LayoutB*, AlignmentB,
      ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Default: bf16/fp16 — TileShape 128x128x128, ClusterShape 1x2x1
template <typename DataType_, bool trans_a, bool trans_b>
struct ScheduleConfig {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

  using LayoutA = GroupedGemmInputALayout<trans_a>;
  using LayoutB = GroupedGemmInputBLayout<trans_b>;
  using LayoutC = cutlass::layout::RowMajor;
  using DataType = DataType_;
};

// FP32 specialisation — TileShape 64x128x32, ClusterShape 1x1x1
template <bool trans_a, bool trans_b>
struct ScheduleConfig<float, trans_a, trans_b> {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_64, cute::_128, cute::_32>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using LayoutA = GroupedGemmInputALayout<trans_a>;
  using LayoutB = GroupedGemmInputBLayout<trans_b>;
  using LayoutC = cutlass::layout::RowMajor;
  using DataType = float;
};

template <typename DataType_, bool trans_a, bool trans_b>
using GemmGrouped = typename GemmGivenSchedule<ScheduleConfig<DataType_, trans_a, trans_b>>::Gemm;

// ---------------------------------------------------------------------------
// Argument builder
// ---------------------------------------------------------------------------
template <typename GemmT, typename ElementA, typename ElementB, typename ElementC, typename StrideA,
          typename StrideB, typename StrideC>
typename GemmT::Arguments MakeArguments(int num_experts, void* problem_sizes_host,
                                        void* problem_sizes, const ElementA** ptr_A,
                                        StrideA* stride_A, const ElementB** ptr_B,
                                        StrideB* stride_B, ElementC** ptr_C, StrideC* stride_C,
                                        float alpha, float beta, int device, int math_sm_count) {
  cutlass::KernelHardwareInfo kernel_hw_info =
      cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename GemmT::GemmKernel>(
          device, math_sm_count);

  typename GemmT::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha = alpha;
  fusion_args.beta = beta;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

  arguments = typename GemmT::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, reinterpret_cast<ProblemShapeType*>(problem_sizes),
       reinterpret_cast<ProblemShapeType const*>(problem_sizes_host)},
      {ptr_A, stride_A, ptr_B, stride_B},
      {
          fusion_args,
          (beta > 0.0) ? (const ElementC**)ptr_C : nullptr,  // NOLINT(*)
          stride_C,
          ptr_C,
          stride_C,
      },
      kernel_hw_info};
  return arguments;
}

template <typename T>
inline __device__ __host__ T ROUND_UP(T m, T n) {
  return (m + n - 1) / n * n;
}

int64_t inline getGemmCoordSize(int64_t num_gemms) {
  return (int64_t)(ROUND_UP(num_gemms * sizeof(ProblemShapeType), 128UL));
}
int64_t inline getPtrSize(int64_t num_gemms) {
  return (int64_t)(ROUND_UP(num_gemms * sizeof(half*), 128UL));
}
int64_t inline getLddSize(int64_t num_gemms) {
  return (int64_t)(ROUND_UP(num_gemms * sizeof(int64_t), 128UL));
}

// 4 MB pinned/host scratch buffer, shared across calls (single-threaded on host side).
static constexpr size_t kCPUWorkSpaceSize = 4 * 1024 * 1024;
static char* getHostWorkspace() {
  static std::once_flag flag;
  static std::shared_ptr<char> workspace;
  std::call_once(flag, [&]() {
    workspace = std::shared_ptr<char>(
        reinterpret_cast<char*>(std::malloc(kCPUWorkSpaceSize)),
        [](char* p) { if (p) std::free(p); });
    if (!workspace) throw std::bad_alloc();
  });
  return workspace.get();
}

// Returns the total GPU workspace size (bytes) required for a given
// (trans_a, trans_b, dtype) / num_gemms combination.
template <bool trans_a, bool trans_b, typename Element>
size_t GetGroupedGemmWorkspaceSize(int num_gemms) {
  using Gemm = GemmGrouped<Element, trans_a, trans_b>;
  typename Gemm::Arguments arguments;
  size_t kernel_workspace_size = Gemm::get_workspace_size(arguments);
  auto gemm_coord_size = getGemmCoordSize(num_gemms);
  auto ptr_size = getPtrSize(num_gemms);
  auto ldd_size = getLddSize(num_gemms);
  auto param_workspace_size = 3 * ptr_size + 3 * ldd_size + gemm_coord_size;
  return static_cast<size_t>(param_workspace_size) + kernel_workspace_size;
}

// ---------------------------------------------------------------------------
// Core kernel driver — path one
//   For each expert i (i=0..num_gemms-1):
//       D[i] = alpha * A[i] @ B[i] + beta * C[i]
//   with A[i] : [m_i, k], B[i] : [k, n] (or transposed via trans_a/trans_b).
// ---------------------------------------------------------------------------
template <bool trans_a, bool trans_b, typename Element>
void CutlassGroupedGemm(const std::vector<torch::Tensor>& A,
                        const std::vector<torch::Tensor>& B,
                        std::vector<torch::Tensor>& D,
                        torch::Tensor& workspace, float alpha, float beta, int num_gemms,
                        cudaStream_t stream, int device, int math_sm_count) {
  using Gemm = GemmGrouped<Element, trans_a, trans_b>;
  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;

  typename Gemm::Arguments arguments;
  size_t kernel_workspace_size = Gemm::get_workspace_size(arguments);
  auto gemm_coord_size = getGemmCoordSize(num_gemms);
  auto ptr_size = getPtrSize(num_gemms);
  auto ldd_size = getLddSize(num_gemms);
  auto param_workspace_size = 3 * ptr_size + 3 * ldd_size + gemm_coord_size;

  TORCH_CHECK(
      param_workspace_size < kCPUWorkSpaceSize,
      "Insufficient kCPUWorkSpaceSize: required=", static_cast<int64_t>(param_workspace_size),
      ", available=", static_cast<int64_t>(kCPUWorkSpaceSize));

  auto total_workspace_size = param_workspace_size + kernel_workspace_size;
  TORCH_CHECK(total_workspace_size <= static_cast<size_t>(workspace.numel()),
              "Insufficient GPU workspace: required=", static_cast<int64_t>(total_workspace_size),
              ", available=", static_cast<int64_t>(workspace.numel()));

  char* workspace_ptr = reinterpret_cast<char*>(workspace.data_ptr());
  char* host_workspace = getHostWorkspace();

  // Host-side layout:
  //   [ problem_sizes | ptr_A | ptr_B | ptr_C | lda | ldb | ldc ]
  ProblemShapeType* problem_sizes_host = reinterpret_cast<ProblemShapeType*>(host_workspace);
  ElementA** ptr_A_host = reinterpret_cast<ElementA**>(host_workspace + gemm_coord_size);
  ElementB** ptr_B_host = reinterpret_cast<ElementB**>(host_workspace + gemm_coord_size + ptr_size);
  ElementC** ptr_C_host = reinterpret_cast<ElementC**>(host_workspace + gemm_coord_size + 2 * ptr_size);
  int64_t* lda_host = reinterpret_cast<int64_t*>(host_workspace + gemm_coord_size + 3 * ptr_size + 0 * ldd_size);
  int64_t* ldb_host = reinterpret_cast<int64_t*>(host_workspace + gemm_coord_size + 3 * ptr_size + 1 * ldd_size);
  int64_t* ldc_host = reinterpret_cast<int64_t*>(host_workspace + gemm_coord_size + 3 * ptr_size + 2 * ldd_size);

  for (int i = 0; i < num_gemms; i++) {
    const auto& inputA = A[i];
    const auto& inputB = B[i];
    auto& outputD = D[i];

    const int m = trans_a ? inputA.size(1) : inputA.size(0);
    const int k = trans_a ? inputA.size(0) : inputA.size(1);
    const int n = trans_b ? inputB.size(0) : inputB.size(1);

    problem_sizes_host[i] = ProblemShapeType(m, n, k);

    ptr_A_host[i] = reinterpret_cast<ElementA*>(inputA.data_ptr());
    ptr_B_host[i] = reinterpret_cast<ElementB*>(inputB.data_ptr());
    ptr_C_host[i] = reinterpret_cast<ElementC*>(outputD.data_ptr());

    lda_host[i] = LayoutA::packed({m, k}).stride(0);
    ldb_host[i] = LayoutB::packed({k, n}).stride(0);
    ldc_host[i] = LayoutC::packed({m, n}).stride(0);
  }

  cudaMemcpyAsync(workspace_ptr, host_workspace, param_workspace_size,
                  cudaMemcpyHostToDevice, stream);

  char* param_workspace_ptr = workspace_ptr;
  ProblemShapeType* problem_sizes_device = reinterpret_cast<ProblemShapeType*>(param_workspace_ptr);
  const ElementA** ptr_A = reinterpret_cast<const ElementA**>(param_workspace_ptr + gemm_coord_size);
  const ElementB** ptr_B = reinterpret_cast<const ElementB**>(param_workspace_ptr + gemm_coord_size + 1 * ptr_size);
  ElementC** ptr_C = reinterpret_cast<ElementC**>(param_workspace_ptr + gemm_coord_size + 2 * ptr_size);

  StrideA* lda = reinterpret_cast<StrideA*>(param_workspace_ptr + gemm_coord_size + 3 * ptr_size + 0 * ldd_size);
  StrideB* ldb = reinterpret_cast<StrideB*>(param_workspace_ptr + gemm_coord_size + 3 * ptr_size + 1 * ldd_size);
  StrideC* ldc = reinterpret_cast<StrideC*>(param_workspace_ptr + gemm_coord_size + 3 * ptr_size + 2 * ldd_size);

  char* kernel_workspace_ptr = workspace_ptr + param_workspace_size;

  arguments = MakeArguments<Gemm, ElementA, ElementB, ElementC, StrideA, StrideB, StrideC>(
      num_gemms, problem_sizes_host, problem_sizes_device, ptr_A, lda, ptr_B, ldb, ptr_C, ldc,
      alpha, beta, device, math_sm_count);

  Gemm gemm;
  TORCH_CHECK(gemm.can_implement(arguments) == cutlass::Status::kSuccess,
              "CUTLASS grouped GEMM can_implement() failed, num_gemms=", num_gemms);
  TORCH_CHECK(gemm.initialize(arguments, kernel_workspace_ptr) == cutlass::Status::kSuccess,
              "CUTLASS grouped GEMM initialize() failed, num_gemms=", num_gemms);
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess,
              "CUTLASS grouped GEMM run() failed, num_gemms=", num_gemms);
}

// Dispatcher for BF16/FP16 forward — trans_a is always false on path one.
static void cutlass_grouped_gemm(const std::vector<torch::Tensor>& A,
                                 const std::vector<torch::Tensor>& B,
                                 std::vector<torch::Tensor>& D, int num_gemms,
                                 bool transb,
                                 torch::Tensor& workspace, int device,
                                 int math_sm_count, cudaStream_t stream) {
  TORCH_CHECK(!A.empty(), "Input tensor list A must not be empty");
  const auto& inputA = A[0];
  float alpha = 1.0f, beta = 0.0f;

  auto dispatch = [&](auto tag) {
    using T = decltype(tag);
    if (!transb) {
      CutlassGroupedGemm<false, false, T>(A, B, D, workspace, alpha, beta, num_gemms,
                                          stream, device, math_sm_count);
    } else {
      CutlassGroupedGemm<false, true, T>(A, B, D, workspace, alpha, beta, num_gemms,
                                         stream, device, math_sm_count);
    }
  };

  if (inputA.scalar_type() == torch::kBFloat16) dispatch(cutlass::bfloat16_t{});
  else if (inputA.scalar_type() == torch::kFloat16) dispatch(cutlass::half_t{});
  else TORCH_CHECK(false, "Unsupported dtype: expected BF16 or FP16");
}

// Dispatcher for FP32 forward — hard-coded to (trans_a=false, trans_b=true).
static void cutlass_fp32_grouped_gemm(const std::vector<torch::Tensor>& A,
                                      const std::vector<torch::Tensor>& B,
                                      std::vector<torch::Tensor>& D, int num_gemms,
                                      torch::Tensor& workspace, int device,
                                      int math_sm_count, cudaStream_t stream) {
  TORCH_CHECK(!A.empty(), "Input tensor list A must not be empty");
  float alpha = 1.0f, beta = 0.0f;
  CutlassGroupedGemm<false, true, float>(A, B, D, workspace, alpha, beta, num_gemms,
                                         stream, device, math_sm_count);
}

// ---------------------------------------------------------------------------
// Public entry point — forward-only (path one)
// ---------------------------------------------------------------------------
static torch::Tensor g_workspace;
static std::mutex g_workspace_mutex;

torch::Tensor hopper_gmm_forward(torch::Tensor A, torch::Tensor B,
                                 torch::Tensor batch_sizes,
                                 bool transb) {
  TORCH_CHECK(A.is_cuda() && A.is_contiguous(), "A must be a contiguous CUDA tensor");
  TORCH_CHECK(B.is_cuda() && B.is_contiguous(), "B must be a contiguous CUDA tensor");
  TORCH_CHECK(A.ndimension() == 2, "A must be 2D");
  TORCH_CHECK(B.ndimension() == 3,
              "B must be 3D [num_experts, dim, dim_out] (forward-only build)");
  TORCH_CHECK(batch_sizes.is_cpu() && batch_sizes.ndimension() == 1
              && batch_sizes.scalar_type() == torch::kInt64,
              "batch_sizes must be a 1D CPU int64 tensor");
  TORCH_CHECK(A.scalar_type() == B.scalar_type(),
              "A and B must have the same dtype");

  int device = A.get_device();
  const bool is_fp32 = (A.scalar_type() == torch::kFloat32);

  // SM90 check
  {
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    TORCH_CHECK(major >= 9,
                "hopper_gmm_forward requires SM90 (Hopper) or newer GPU, "
                "got compute capability ", major, ".", minor);
  }
  int math_sm_count = 0;
  cudaDeviceGetAttribute(&math_sm_count, cudaDevAttrMultiProcessorCount, device);

  const int64_t num_experts = batch_sizes.size(0);
  const int64_t dim = A.size(1);
  const int64_t dim_out = transb ? B.size(1) : B.size(2);

  TORCH_CHECK(B.size(0) == num_experts,
              "B.size(0) must equal num_experts (batch_sizes.size(0))");
  TORCH_CHECK((transb ? B.size(2) : B.size(1)) == dim,
              "B.", (transb ? "size(2)" : "size(1)"), " must equal A.size(1) (dim), got ",
              (transb ? B.size(2) : B.size(1)), " vs ", dim);

  // Workspace size
  size_t required_workspace_size = 0;
  if (is_fp32) {
    // fp32 path is always (false, true)
    required_workspace_size = GetGroupedGemmWorkspaceSize<false, true, float>(
        static_cast<int>(num_experts));
  } else if (A.scalar_type() == torch::kBFloat16) {
    required_workspace_size = transb
        ? GetGroupedGemmWorkspaceSize<false, true, cutlass::bfloat16_t>(static_cast<int>(num_experts))
        : GetGroupedGemmWorkspaceSize<false, false, cutlass::bfloat16_t>(static_cast<int>(num_experts));
  } else if (A.scalar_type() == torch::kFloat16) {
    required_workspace_size = transb
        ? GetGroupedGemmWorkspaceSize<false, true, cutlass::half_t>(static_cast<int>(num_experts))
        : GetGroupedGemmWorkspaceSize<false, false, cutlass::half_t>(static_cast<int>(num_experts));
  } else {
    TORCH_CHECK(false, "Unsupported dtype: only FP32 / BF16 / FP16 are supported");
  }

  // Grow-only global workspace
  {
    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    if (!g_workspace.defined() || g_workspace.device().index() != device
        || static_cast<size_t>(g_workspace.numel()) < required_workspace_size) {
      g_workspace = torch::empty({static_cast<int64_t>(required_workspace_size)},
                                 torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));
    }
  }

  // Zero-init output so batch_sizes[i]==0 experts produce zero rows.
  auto D = torch::zeros({A.size(0), dim_out}, A.options());

  // fp32: B must be ColumnMajor, so transpose to [E, dim_out, dim] contiguous first.
  torch::Tensor B_t;
  if (is_fp32) {
    B_t = B.permute({0, 2, 1}).contiguous();  // [E, dim_out, dim]
  }

  const int64_t* bs_ptr = batch_sizes.data_ptr<int64_t>();
  std::vector<torch::Tensor> A_list, B_list, D_list;
  A_list.reserve(num_experts);
  B_list.reserve(num_experts);
  D_list.reserve(num_experts);

  int64_t offset = 0;
  int actual_num_gemms = 0;
  for (int64_t i = 0; i < num_experts; ++i) {
    int64_t m_i = bs_ptr[i];
    if (m_i == 0) continue;
    A_list.push_back(A.narrow(0, offset, m_i));
    if (is_fp32) {
      B_list.push_back(B_t[i]);
    } else {
      B_list.push_back(B[i]);
    }
    D_list.push_back(D.narrow(0, offset, m_i));
    offset += m_i;
    actual_num_gemms++;
  }

  TORCH_CHECK(offset == A.size(0),
              "Sum of batch_sizes (", offset, ") must equal A.size(0) (", A.size(0), ")");

  if (actual_num_gemms == 0) return D;

  auto stream = c10::cuda::getCurrentCUDAStream(device).stream();

  if (is_fp32) {
    cutlass_fp32_grouped_gemm(A_list, B_list, D_list, actual_num_gemms,
                              g_workspace, device, math_sm_count, stream);
  } else {
    cutlass_grouped_gemm(A_list, B_list, D_list, actual_num_gemms,
                         transb, g_workspace, device, math_sm_count, stream);
  }

  return D;
}

// ---------------------------------------------------------------------------
// Explicit template instantiations — only the ones path one actually uses.
//   * bf16/fp16 forward supports both NN (trans_b=false) and NT (trans_b=true)
//   * fp32 forward is fixed to (false, true)
// ---------------------------------------------------------------------------
template void CutlassGroupedGemm<false, false, cutlass::half_t>(
    const std::vector<torch::Tensor>&, const std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&, torch::Tensor&, float, float, int, cudaStream_t, int, int);
template void CutlassGroupedGemm<false, true, cutlass::half_t>(
    const std::vector<torch::Tensor>&, const std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&, torch::Tensor&, float, float, int, cudaStream_t, int, int);
template void CutlassGroupedGemm<false, false, cutlass::bfloat16_t>(
    const std::vector<torch::Tensor>&, const std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&, torch::Tensor&, float, float, int, cudaStream_t, int, int);
template void CutlassGroupedGemm<false, true, cutlass::bfloat16_t>(
    const std::vector<torch::Tensor>&, const std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&, torch::Tensor&, float, float, int, cudaStream_t, int, int);
template void CutlassGroupedGemm<false, true, float>(
    const std::vector<torch::Tensor>&, const std::vector<torch::Tensor>&,
    std::vector<torch::Tensor>&, torch::Tensor&, float, float, int, cudaStream_t, int, int);

}  // namespace hpc_ops_hopper

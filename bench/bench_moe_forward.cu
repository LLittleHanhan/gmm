// ============================================================================
// Hopper (SM90a) grouped GEMM — FORWARD-ONLY 性能 bench
//
// 纯 C++ / CUTLASS, 不依赖 PyTorch。用 nvcc 直接编成可执行文件。
//
// MoE problem size (按路由均分):
//     E_route   = 640
//     E_shared  = 80
//     bs        = 4000, num_token = 80, top_k = 3
//     N         = 1280, K = 320
//
//     总路由 token = bs * num_token * top_k = 960_000
//     均分 M/expert = 960_000 / 640 = 1500
//
//     共享专家 M/expert = bs * num_token   = 320_000
//
// 测试场景:
//   route  NN/NT     E=640,  M/expert=1500
//   shared NN/NT     E=80,   M/expert=320000
//   e2e              route_NN + shared_NN 串行
//
// 用法:
//   bash build_bench.sh        # 编译
//   ./bench_moe_forward                 # 默认 bf16
//   ./bench_moe_forward --dtype fp16
//   ./bench_moe_forward --dtype fp32    # 只跑 NT (fp32 限制)
//   ./bench_moe_forward --warmup 20 --iters 100
// ============================================================================

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

// ---------------------------------------------------------------------------
#define CUDA_CHECK(stmt)                                                     \
  do {                                                                       \
    cudaError_t e = (stmt);                                                  \
    if (e != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(e));                                   \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

#define CUTLASS_CHECK(stmt)                                                  \
  do {                                                                       \
    cutlass::Status s = (stmt);                                              \
    if (s != cutlass::Status::kSuccess) {                                    \
      std::fprintf(stderr, "CUTLASS error %s:%d: %d\n", __FILE__, __LINE__,  \
                   static_cast<int>(s));                                     \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

// ---------------------------------------------------------------------------
// ScheduleConfig — 与 csrc/hopper_gmm_forward.cu 保持一致
// ---------------------------------------------------------------------------
template <bool trans_a>
using GroupedGemmInputALayout =
    std::conditional_t<trans_a, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
template <bool trans_b>
using GroupedGemmInputBLayout =
    std::conditional_t<trans_b, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;

using ProblemShapeType = cute::Shape<int, int, int>;
using ProblemShape = cutlass::gemm::GroupProblemShape<ProblemShapeType>;

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

// fp32 特化
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

template <typename DataType_, bool trans_a, bool trans_b>
using GemmGrouped = typename GemmGivenSchedule<ScheduleConfig<DataType_, trans_a, trans_b>>::Gemm;

// ---------------------------------------------------------------------------
// 辅助
// ---------------------------------------------------------------------------
template <typename T>
inline T round_up(T m, T n) {
  return (m + n - 1) / n * n;
}

static int64_t get_coord_size(int64_t n) {
  return round_up<int64_t>(n * sizeof(ProblemShapeType), 128);
}
static int64_t get_ptr_size(int64_t n) {
  return round_up<int64_t>(n * sizeof(void*), 128);
}
static int64_t get_ldd_size(int64_t n) {
  return round_up<int64_t>(n * sizeof(int64_t), 128);
}

// 填充随机数 — host → device, 只做一次用于 sanity / 防止 bogus fastpath
template <typename T>
static void fill_random(T* d_ptr, size_t n, uint64_t seed = 1234) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
  std::vector<T> ht(n);
  for (size_t i = 0; i < n; ++i) {
    float v = dist(rng);
    ht[i] = T(v);  // cutlass::bfloat16_t / half_t / float 都支持 T(float)
  }
  CUDA_CHECK(cudaMemcpy(d_ptr, ht.data(), n * sizeof(T), cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// 每个 (dtype, trans_b) 组合的一次 grouped-GEMM 运行器
//   - 均匀 M/expert (MoE 路由均分或共享专家都适用)
//   - A: [M_total, K] row-major
//   - B: [E, K, N] row-major   (trans_b=false, NN)
//      或 B: [E, N, K] row-major (trans_b=true,  NT)
//   - D: [M_total, N] row-major
// ---------------------------------------------------------------------------
template <typename Element, bool trans_b>
struct GroupedGemmRunner {
  using Gemm = GemmGrouped<Element, /*trans_a=*/false, trans_b>;
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;

  int num_experts;
  int m_per;     // 每专家 M
  int k;
  int n;

  // 设备端集中分配
  ElementA* dA = nullptr;     // [E*m_per, k]
  ElementB* dB = nullptr;     // [E, k*n]  (或 [E, n*k])
  ElementC* dD = nullptr;     // [E*m_per, n]
  char* d_workspace = nullptr;
  size_t workspace_bytes = 0;

  // host-side meta
  std::vector<ProblemShapeType> h_problem_sizes;
  std::vector<ElementA const*> h_ptr_A;
  std::vector<ElementB const*> h_ptr_B;
  std::vector<ElementC*>       h_ptr_C;
  std::vector<StrideA>         h_lda;
  std::vector<StrideB>         h_ldb;
  std::vector<StrideC>         h_ldc;

  // device-side pointer arrays
  ProblemShapeType* d_problem_sizes = nullptr;
  ElementA const** d_ptr_A = nullptr;
  ElementB const** d_ptr_B = nullptr;
  ElementC**       d_ptr_C = nullptr;
  StrideA* d_lda = nullptr;
  StrideB* d_ldb = nullptr;
  StrideC* d_ldc = nullptr;

  int device = 0;
  int sm_count = 0;

  void setup(int E, int M_per, int K, int N, int dev, int sm) {
    num_experts = E; m_per = M_per; k = K; n = N; device = dev; sm_count = sm;

    size_t a_elems = static_cast<size_t>(E) * M_per * K;
    size_t b_elems = static_cast<size_t>(E) * K * N;   // NN 或 NT 总量一样
    size_t d_elems = static_cast<size_t>(E) * M_per * N;

    CUDA_CHECK(cudaMalloc(&dA, a_elems * sizeof(ElementA)));
    CUDA_CHECK(cudaMalloc(&dB, b_elems * sizeof(ElementB)));
    CUDA_CHECK(cudaMalloc(&dD, d_elems * sizeof(ElementC)));

    fill_random<ElementA>(dA, a_elems, 0x12345);
    fill_random<ElementB>(dB, b_elems, 0x67890);
    CUDA_CHECK(cudaMemset(dD, 0, d_elems * sizeof(ElementC)));

    // 填 per-expert meta
    h_problem_sizes.resize(E);
    h_ptr_A.resize(E); h_ptr_B.resize(E); h_ptr_C.resize(E);
    h_lda.resize(E);   h_ldb.resize(E);   h_ldc.resize(E);

    size_t a_stride_elems = static_cast<size_t>(M_per) * K;
    size_t b_stride_elems = static_cast<size_t>(K) * N;    // NN: K*N;  NT: N*K 也是 N*K (总数同)
    size_t d_stride_elems = static_cast<size_t>(M_per) * N;

    for (int i = 0; i < E; ++i) {
      h_problem_sizes[i] = ProblemShapeType(M_per, N, K);
      h_ptr_A[i] = dA + i * a_stride_elems;
      h_ptr_B[i] = dB + i * b_stride_elems;
      h_ptr_C[i] = dD + i * d_stride_elems;
      h_lda[i] = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M_per, K, 1));
      h_ldb[i] = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
      h_ldc[i] = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M_per, N, 1));
    }

    // workspace for per-expert ptrs
    CUDA_CHECK(cudaMalloc(&d_problem_sizes, E * sizeof(ProblemShapeType)));
    CUDA_CHECK(cudaMalloc(&d_ptr_A, E * sizeof(ElementA*)));
    CUDA_CHECK(cudaMalloc(&d_ptr_B, E * sizeof(ElementB*)));
    CUDA_CHECK(cudaMalloc(&d_ptr_C, E * sizeof(ElementC*)));
    CUDA_CHECK(cudaMalloc(&d_lda,   E * sizeof(StrideA)));
    CUDA_CHECK(cudaMalloc(&d_ldb,   E * sizeof(StrideB)));
    CUDA_CHECK(cudaMalloc(&d_ldc,   E * sizeof(StrideC)));

    CUDA_CHECK(cudaMemcpy(d_problem_sizes, h_problem_sizes.data(),
                          E * sizeof(ProblemShapeType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ptr_A, h_ptr_A.data(), E * sizeof(ElementA*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ptr_B, h_ptr_B.data(), E * sizeof(ElementB*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ptr_C, h_ptr_C.data(), E * sizeof(ElementC*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lda, h_lda.data(), E * sizeof(StrideA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ldb, h_ldb.data(), E * sizeof(StrideB), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ldc, h_ldc.data(), E * sizeof(StrideC), cudaMemcpyHostToDevice));

    // workspace size
    typename Gemm::Arguments args = make_args_(0.0f);
    workspace_bytes = Gemm::get_workspace_size(args);
    if (workspace_bytes > 0) CUDA_CHECK(cudaMalloc(&d_workspace, workspace_bytes));
  }

  typename Gemm::Arguments make_args_(float /*unused*/) {
    cutlass::KernelHardwareInfo kernel_hw_info =
        cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename Gemm::GemmKernel>(
            device, sm_count);

    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args;
    fusion_args.alpha = 1.0f;
    fusion_args.beta = 0.0f;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta  = {cute::_0{}, cute::_0{}, 0};

    arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {num_experts, d_problem_sizes, h_problem_sizes.data()},
        {d_ptr_A, d_lda, d_ptr_B, d_ldb},
        {
            fusion_args,
            nullptr,            // C ptrs (beta=0 所以不用)
            d_ldc,
            d_ptr_C,
            d_ldc,
        },
        kernel_hw_info};
    return arguments;
  }

  // 初始化并 run 一次 (返回 gemm 对象，后续 run 可以直接复用 kernel)
  Gemm gemm;
  typename Gemm::Arguments args;
  bool initialized = false;

  void init_gemm() {
    args = make_args_(0.0f);
    CUTLASS_CHECK(gemm.can_implement(args));
    CUTLASS_CHECK(gemm.initialize(args, d_workspace));
    initialized = true;
  }

  void run(cudaStream_t stream) {
    if (!initialized) init_gemm();
    CUTLASS_CHECK(gemm.run(stream));
  }

  void teardown() {
    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dD) cudaFree(dD);
    if (d_workspace) cudaFree(d_workspace);
    if (d_problem_sizes) cudaFree(d_problem_sizes);
    if (d_ptr_A) cudaFree(d_ptr_A);
    if (d_ptr_B) cudaFree(d_ptr_B);
    if (d_ptr_C) cudaFree(d_ptr_C);
    if (d_lda) cudaFree(d_lda);
    if (d_ldb) cudaFree(d_ldb);
    if (d_ldc) cudaFree(d_ldc);
  }
};

// ---------------------------------------------------------------------------
// 计时
// ---------------------------------------------------------------------------
struct Stats { float median_ms; float mean_ms; float min_ms; };

template <typename Fn>
static Stats time_it(Fn&& fn, int warmup, int iters) {
  cudaStream_t stream = 0;
  for (int i = 0; i < warmup; ++i) fn(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<cudaEvent_t> s(iters), e(iters);
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventCreate(&s[i]));
    CUDA_CHECK(cudaEventCreate(&e[i]));
  }
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(s[i], stream));
    fn(stream);
    CUDA_CHECK(cudaEventRecord(e[i], stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<float> t(iters);
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventElapsedTime(&t[i], s[i], e[i]));
    CUDA_CHECK(cudaEventDestroy(s[i]));
    CUDA_CHECK(cudaEventDestroy(e[i]));
  }
  std::sort(t.begin(), t.end());
  float med = t[iters / 2];
  float mn = t.front();
  float mean = std::accumulate(t.begin(), t.end(), 0.0f) / iters;
  return {med, mean, mn};
}

static double tflops_of(int64_t m_total, int n, int k, float ms) {
  return 2.0 * m_total * n * k / (ms * 1e-3) / 1e12;
}

// ---------------------------------------------------------------------------
// 单个场景跑 + 打印
// ---------------------------------------------------------------------------
template <typename Element, bool trans_b>
static Stats run_scenario(const char* tag, int E, int M_per, int K, int N,
                          int warmup, int iters, int device, int sm_count,
                          size_t elem_bytes) {
  GroupedGemmRunner<Element, trans_b> runner;
  runner.setup(E, M_per, K, N, device, sm_count);
  runner.init_gemm();

  // sanity: 跑一次同步看有无错误
  runner.run(0);
  CUDA_CHECK(cudaDeviceSynchronize());

  Stats st = time_it([&](cudaStream_t s) { runner.run(s); }, warmup, iters);

  int64_t m_total = static_cast<int64_t>(E) * M_per;
  double tf = tflops_of(m_total, N, K, st.median_ms);
  // A + B + D bytes
  double bytes = (double)m_total * K * elem_bytes
               + (double)E * K * N * elem_bytes
               + (double)m_total * N * elem_bytes;
  double gbps = bytes / (st.median_ms * 1e-3) / 1e9;

  std::printf("  [%-6s] %s E=%4d M/expert=%6d M_total=%9ld K=%4d N=%4d  "
              "median=%7.3f ms  mean=%7.3f ms  min=%7.3f ms  "
              "%7.2f TFLOPS  %7.1f GB/s\n",
              tag, trans_b ? "NT" : "NN",
              E, M_per, (long)m_total, K, N,
              st.median_ms, st.mean_ms, st.min_ms, tf, gbps);

  runner.teardown();
  return st;
}

// fp32 专用: 只有 NT (trans_a=false, trans_b=true) 布局。
// 这里走 trans_b=true, 但 B 的数据我们仍然按 [E, K, N] 分配足够大内存 (setup 里 size 正确即可)。

// ---------------------------------------------------------------------------
// 命令行参数
// ---------------------------------------------------------------------------
struct Args {
  std::string dtype = "bf16";
  int warmup = 20;
  int iters = 100;

  int E_route  = 640;
  int E_shared = 80;
  int bs = 4000, num_token = 80, top_k = 3;
  int N = 1280, K = 320;
};

static Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "--%s requires a value\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (s == "--dtype") a.dtype = next("dtype");
    else if (s == "--warmup") a.warmup = std::stoi(next("warmup"));
    else if (s == "--iters") a.iters = std::stoi(next("iters"));
    else if (s == "--E-route") a.E_route = std::stoi(next("E-route"));
    else if (s == "--E-shared") a.E_shared = std::stoi(next("E-shared"));
    else if (s == "--bs") a.bs = std::stoi(next("bs"));
    else if (s == "--num-token") a.num_token = std::stoi(next("num-token"));
    else if (s == "--top-k") a.top_k = std::stoi(next("top-k"));
    else if (s == "--N") a.N = std::stoi(next("N"));
    else if (s == "--K") a.K = std::stoi(next("K"));
    else if (s == "-h" || s == "--help") {
      std::printf("usage: %s [--dtype bf16|fp16|fp32] [--warmup N] [--iters N]\n"
                  "          [--E-route 640] [--E-shared 80] [--bs 4000] [--num-token 80] [--top-k 3]\n"
                  "          [--N 1280] [--K 320]\n", argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
      std::exit(2);
    }
  }
  return a;
}

// ---------------------------------------------------------------------------
// 每个 dtype 分发
// ---------------------------------------------------------------------------
template <typename Element>
static void run_all(const Args& a, int device, int sm_count, size_t elem_bytes) {
  int total_routed = a.bs * a.num_token * a.top_k;
  int m_route = total_routed / a.E_route;
  int m_shared = a.bs * a.num_token;

  std::printf(">>> 路由专家 (均分 M/expert = %d)\n", m_route);
  run_scenario<Element, /*trans_b=*/false>("route", a.E_route, m_route, a.K, a.N,
                                           a.warmup, a.iters, device, sm_count, elem_bytes);
  if (!std::is_same_v<Element, float>) {
    run_scenario<Element, /*trans_b=*/true >("route", a.E_route, m_route, a.K, a.N,
                                             a.warmup, a.iters, device, sm_count, elem_bytes);
  }

  if (a.E_shared > 0) {
    std::printf(">>> 共享专家 (每专家 M = bs*num_token = %d)\n", m_shared);
    run_scenario<Element, /*trans_b=*/false>("shared", a.E_shared, m_shared, a.K, a.N,
                                             a.warmup, a.iters, device, sm_count, elem_bytes);
    if (!std::is_same_v<Element, float>) {
      run_scenario<Element, /*trans_b=*/true >("shared", a.E_shared, m_shared, a.K, a.N,
                                               a.warmup, a.iters, device, sm_count, elem_bytes);
    }
  }

  // e2e: route_NN + shared_NN 串行
  std::printf(">>> 端到端 (route_NN + shared_NN 串行)\n");
  GroupedGemmRunner<Element, false> r_route; r_route.setup(a.E_route, m_route, a.K, a.N, device, sm_count); r_route.init_gemm();
  GroupedGemmRunner<Element, false> r_share;
  if (a.E_shared > 0) { r_share.setup(a.E_shared, m_shared, a.K, a.N, device, sm_count); r_share.init_gemm(); }

  auto e2e = [&](cudaStream_t s) {
    r_route.run(s);
    if (a.E_shared > 0) r_share.run(s);
  };
  e2e(0); CUDA_CHECK(cudaDeviceSynchronize());
  Stats st = time_it(e2e, a.warmup, a.iters);
  int64_t m_total = static_cast<int64_t>(a.E_route) * m_route + (int64_t)a.E_shared * m_shared;
  double tf = tflops_of(m_total, a.N, a.K, st.median_ms);
  std::printf("  [e2e  ] -- M_total=%9ld                          "
              "median=%7.3f ms  mean=%7.3f ms  min=%7.3f ms  %7.2f TFLOPS\n",
              (long)m_total, st.median_ms, st.mean_ms, st.min_ms, tf);

  r_route.teardown();
  if (a.E_shared > 0) r_share.teardown();
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  Args a = parse(argc, argv);

  int device = 0;
  CUDA_CHECK(cudaSetDevice(device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  if (prop.major < 9) {
    std::fprintf(stderr, "need SM90+ (Hopper), got sm_%d%d (%s)\n",
                 prop.major, prop.minor, prop.name);
    return 1;
  }
  int sm_count = prop.multiProcessorCount;

  std::printf("==========================================================================\n");
  std::printf("hopper_gmm_forward — MoE forward 性能测试 (pure C++/CUTLASS, no torch)\n");
  std::printf("device : %s  (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, sm_count);
  std::printf("dtype  : %s    warmup=%d  iters=%d\n", a.dtype.c_str(), a.warmup, a.iters);
  std::printf("problem: bs=%d  num_token=%d  top_k=%d  E_route=%d  E_shared=%d  N=%d  K=%d\n",
              a.bs, a.num_token, a.top_k, a.E_route, a.E_shared, a.N, a.K);
  int total_routed = a.bs * a.num_token * a.top_k;
  std::printf("         total_routed_tokens = %d    路由均分 M/expert = %d\n",
              total_routed, total_routed / a.E_route);
  std::printf("==========================================================================\n");

  if (a.dtype == "bf16" || a.dtype == "bfloat16") {
    run_all<cutlass::bfloat16_t>(a, device, sm_count, 2);
  } else if (a.dtype == "fp16" || a.dtype == "half") {
    run_all<cutlass::half_t>(a, device, sm_count, 2);
  } else if (a.dtype == "fp32" || a.dtype == "float" || a.dtype == "float32") {
    run_all<float>(a, device, sm_count, 4);
  } else {
    std::fprintf(stderr, "unknown dtype: %s (use bf16/fp16/fp32)\n", a.dtype.c_str());
    return 2;
  }

  std::printf("==========================================================================\n");
  return 0;
}

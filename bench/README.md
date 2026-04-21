# bench_moe_forward — 纯 C++/CUTLASS 性能测试

不依赖 PyTorch / pybind，只用 `nvcc` + CUTLASS 编一个可执行文件，直接跑 MoE 场景的 grouped GEMM 性能。

## 测试问题规模

对应业务 MoE：
```
E_route  = 640   # 路由专家
E_shared = 80    # 共享专家
bs       = 4000, num_token = 80, top_k = 3
N        = 1280, K = 320
```

路由专家（均分）: `M/expert = 4000*80*3 / 640 = 1500`
共享专家         : `M/expert = 4000*80 = 320000`

## 编译 + 运行

```bash
cd hopper_gmm_forward_standalone/bench
CUTLASS_HOME=$HOME/cutlass bash build_bench.sh                       # 只编译
CUTLASS_HOME=$HOME/cutlass bash build_bench.sh --run                 # 编完即跑 (bf16)
CUTLASS_HOME=$HOME/cutlass bash build_bench.sh --run --dtype fp16
CUTLASS_HOME=$HOME/cutlass bash build_bench.sh --run --dtype fp32
CUTLASS_HOME=$HOME/cutlass bash build_bench.sh --run --warmup 30 --iters 200
```

> 首次编译较慢（~1 min）：CUTLASS 3.x 的 grouped kernel 模板吃 nvcc 时间很凶。

## 可执行文件参数

```
./bench_moe_forward
    [--dtype bf16|fp16|fp32]     默认 bf16
    [--warmup 20] [--iters 100]
    [--E-route 640] [--E-shared 80]
    [--bs 4000] [--num-token 80] [--top-k 3]
    [--N 1280] [--K 320]
```

- `fp32` 只跑 `NT` 布局（TileShape 64×128×32, Cluster 1×1×1），和原 `.cu` 路径一一致。
- `bf16/fp16` 会同时跑 `NN` 和 `NT` 两种 `trans_b`。
- 结果包含 `median / mean / min (ms)`, `TFLOPS (by 2*M*N*K)`, `GB/s (A+B+D)`。

## 与 `csrc/hopper_gmm_forward.cu` 的一致性

`bench_moe_forward.cu` 里的 `ScheduleConfig` / `GemmGivenSchedule` / FP32 特化 tile 配置
全部复制自原文件，保证同一套 kernel：
- bf16/fp16: `TileShape = 128×128×128`, `ClusterShape = 1×2×1`
- fp32     : `TileShape = 64×128×32`,  `ClusterShape = 1×1×1`
- schedule : `KernelPtrArrayTmaWarpSpecializedPingpong`
- epilogue : `PtrArrayTmaWarpSpecializedPingpong`

## 注意

1. `libcuda` 在驱动里（`/usr/lib64/libcuda.so.1`）或 CUDA toolkit 的 stubs
   目录（`$CUDA_HOME/lib64/stubs/libcuda.so`）。如果链接报 `cannot find -lcuda`：
   ```bash
   export LIBRARY_PATH=$CUDA_HOME/lib64/stubs:$LIBRARY_PATH
   ```

2. Hopper TMA 路径要求对齐严格：当前配置要求
   `M_per * K * sizeof(elem)` 和 `K * N * sizeof(elem)` 都能满足 128-bit / 16-byte 对齐。
   默认 `K=320, N=1280` 对 bf16/fp16 足够；自定义极端尺寸可能会 `can_implement()` 失败。

3. 这里 bench 每专家 `M` 相同（路由均分 / 共享全 token）。若要模拟路由不均，
   需要改 `GroupedGemmRunner::setup` 使每个 expert 的 `M` 不同。

# hopper_gmm_forward_standalone

**最小可运行** 的 Hopper (SM90a) grouped GEMM **前向** 工程。从 `nezha/cpplus/backend/pytorch/hpc_ops/hopper_grouped_gemm.cu.cc` 中只抽取 "路径一 forward"：

```
A: [num_tokens, dim]           ┐
B: [num_experts, dim, dim_out] │ → D: [num_tokens, dim_out]
batch_sizes: [num_experts]     ┘
```

已删除：variable-K (`CutlassGroupedGemmVariableK`)、反向用的 `trans_a=True` 实例化、所有 debug 日志。保留下来的 kernel 与原 nezha 的路径一行为逐行对应。

## 目录

```
hopper_gmm_forward_standalone/
├── README.md
├── setup.py                    # 独立编译（torch.utils.cpp_extension + nvcc sm_90a）
├── build_and_test.sh           # 一键编译+测试
├── csrc/
│   ├── hopper_gmm_forward.h    # 对外接口声明
│   ├── hopper_gmm_forward.cu   # 精简版 kernel（保留路径一）
│   └── bindings.cc             # TORCH_LIBRARY("hopper_gmm_fwd") 注册
├── hopper_gmm_forward/
│   └── __init__.py             # Python wrapper, 加载 _C
└── tests/
    └── test_forward.py         # fp32/bf16/fp16 正确性测试
```

## 支持的路径

| dtype | trans_b=False | trans_b=True |
|---|---|---|
| fp32   | 内部 `B.permute(0,2,1).contiguous()` 后始终走 (false,true)，用户无需关心 | 同左 |
| bf16   | ✅ NN | ✅ NT |
| fp16   | ✅ NN | ✅ NT |

对外始终只接受 **3D B**：`[E, dim, dim_out]`（或 `trans_b=True` 时 `[E, dim_out, dim]`）。

## 依赖

- CUDA ≥ 12.0，nvcc 能编 sm_90a
- PyTorch ≥ 2.4
- CUTLASS ≥ 3.5（推荐内网 `dylancai/cutlass v4.3.5` 或官方 `v3.8.0`）

## 编译 + 测试

```bash
cd hopper_gmm_forward_standalone
CUTLASS_HOME=$HOME/cutlass bash build_and_test.sh

# 只编不测：
CUTLASS_HOME=$HOME/cutlass bash build_and_test.sh --no-test
```

## 使用

```python
import torch
from hopper_gmm_forward import hopper_gmm_fwd

a = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
b = torch.randn(8,   128, 256, device="cuda", dtype=torch.bfloat16)
batch_sizes = torch.tensor([64] * 8, dtype=torch.int64)  # CPU int64

c = hopper_gmm_fwd(a, b, batch_sizes)   # [512, 256]
```

算子还被注册为 `torch.ops.hopper_gmm_fwd.gmm`，可在 `torch.compile` / PT2 导出里直接使用（已提供 `register_fake`）。

## 与 `grouped_gemm_standalone` 的关系

`grouped_gemm_standalone` 是整套三个实现（Ampere + Hopper 完整版 + Triton K-grouped），本目录只是把其中 **Hopper forward 那一段 kernel** 单独拿出来，代码量约 400 行（原文件 1097 行），编译产物更小、编译更快，适合只需要 MoE forward 推理场景。

#!/usr/bin/env bash
# 纯 nvcc 编译 bench_moe_forward.cu (CUTLASS 3.x, SM90a), 不依赖 PyTorch。
#
# 用法:
#   CUTLASS_HOME=/path/to/cutlass bash build_bench.sh
#   CUTLASS_HOME=/path/to/cutlass bash build_bench.sh --run
#   CUTLASS_HOME=/path/to/cutlass bash build_bench.sh --run --dtype fp16
#
# 环境变量:
#   CUTLASS_HOME  必填
#   CUDA_HOME     可选, 默认探测 /usr/local/cuda
#   NVCC          可选, 默认 $CUDA_HOME/bin/nvcc 或 PATH 里的 nvcc
#   ARCH          可选, 默认 90a (Hopper)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

: "${CUTLASS_HOME:?CUTLASS_HOME is not set; point it to a CUTLASS checkout}"
if [[ ! -f "$CUTLASS_HOME/include/cutlass/cutlass.h" ]]; then
    echo "[ERR] $CUTLASS_HOME/include/cutlass/cutlass.h not found" >&2
    exit 1
fi

# 定位 CUDA / nvcc
if [[ -z "${CUDA_HOME:-}" ]]; then
    if [[ -d /usr/local/cuda ]]; then
        CUDA_HOME=/usr/local/cuda
    elif command -v nvcc >/dev/null 2>&1; then
        CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    else
        echo "[ERR] cannot locate CUDA. set CUDA_HOME." >&2
        exit 1
    fi
fi
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
if [[ ! -x "$NVCC" ]]; then
    NVCC="$(command -v nvcc || true)"
fi
if [[ -z "$NVCC" || ! -x "$NVCC" ]]; then
    echo "[ERR] nvcc not found. set NVCC or CUDA_HOME." >&2
    exit 1
fi

ARCH="${ARCH:-90a}"
OUT="$HERE/bench_moe_forward"

echo "=========================================="
echo "  Building bench_moe_forward (sm_${ARCH})"
echo "  CUTLASS  = $CUTLASS_HOME"
echo "  CUDA     = $CUDA_HOME"
echo "  nvcc     = $NVCC"
echo "=========================================="

# 如需更详细日志, 把 --ptxas-options=-v 等打开即可.
NVCC_FLAGS=(
    -O3
    -std=c++17
    -gencode "arch=compute_${ARCH},code=sm_${ARCH}"
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -DCUTE_USE_PACKED_TUPLE=1
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1
    -DCUTLASS_VERSIONS_GENERATED
    -DCUTLASS_TEST_LEVEL=0
    -DENABLE_BF16
    -D__CUDA_NO_HALF_OPERATORS__
    -D__CUDA_NO_HALF_CONVERSIONS__
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__
    -D__CUDA_NO_HALF2_OPERATORS__
    -I "$CUTLASS_HOME/include"
    -I "$CUTLASS_HOME/tools/util/include"
    -lcuda
    -lcudart
)

set -x
"$NVCC" "${NVCC_FLAGS[@]}" "$HERE/bench_moe_forward.cu" -o "$OUT"
{ set +x; } 2>/dev/null

echo
echo "built: $OUT"
echo

if [[ "${1:-}" == "--run" ]]; then
    shift
    echo "=========================================="
    echo "  Running: $OUT $*"
    echo "=========================================="
    exec "$OUT" "$@"
fi

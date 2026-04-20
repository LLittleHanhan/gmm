#!/usr/bin/env bash
# Build + test the minimal forward-only Hopper grouped GEMM.
#
# Usage:
#   CUTLASS_HOME=/path/to/cutlass bash build_and_test.sh
#   CUTLASS_HOME=/path/to/cutlass bash build_and_test.sh --no-test
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

: "${CUTLASS_HOME:?CUTLASS_HOME is not set; point it to a CUTLASS checkout}"

if [[ ! -f "$CUTLASS_HOME/include/cutlass/cutlass.h" ]]; then
    echo "[ERR] $CUTLASS_HOME/include/cutlass/cutlass.h not found" >&2
    exit 1
fi

echo "=========================================="
echo "  Building hopper_gmm_forward (SM90a)"
echo "  CUTLASS = $CUTLASS_HOME"
echo "=========================================="

export TORCH_CUDA_ARCH_LIST="9.0a"
python setup.py build_ext --inplace

echo
ls -lh hopper_gmm_forward/_C*.so 2>/dev/null || true

if [[ "${1:-}" == "--no-test" ]]; then
    exit 0
fi

echo
echo "=========================================="
echo "  Running forward test"
echo "=========================================="
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
python tests/test_forward.py

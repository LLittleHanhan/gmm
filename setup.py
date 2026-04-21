"""Build script for the minimal forward-only Hopper grouped GEMM.

Usage:
    CUTLASS_HOME=/path/to/cutlass python setup.py build_ext --inplace
    # or
    CUTLASS_HOME=/path/to/cutlass pip install -e .
"""

import os
import sys
from pathlib import Path

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _locate_cutlass() -> str:
    candidates = []
    if os.environ.get("CUTLASS_HOME"):
        candidates.append(os.environ["CUTLASS_HOME"])
    candidates += [
        "/usr/local/cutlass",
        "/opt/cutlass",
        os.path.expanduser("~/cutlass"),
        str(Path(__file__).resolve().parent / "third_party" / "cutlass"),
    ]
    for c in candidates:
        if c and (Path(c) / "include" / "cutlass" / "cutlass.h").exists():
            return c
    raise RuntimeError(
        "CUTLASS not found. Set CUTLASS_HOME to a checkout with "
        "include/cutlass/cutlass.h. Tried: {}".format(candidates))


HERE = Path(__file__).resolve().parent
CSRC = HERE / "csrc"

cutlass_home = _locate_cutlass()
print(f"[hopper_gmm_forward] CUTLASS = {cutlass_home}", file=sys.stderr)

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")

ext = CUDAExtension(
    name="hopper_gmm_forward._C",
    sources=[
        str(CSRC / "bindings.cc"),
        str(CSRC / "hopper_gmm_forward.cu"),
    ],
    include_dirs=[
        str(CSRC),
        str(Path(cutlass_home) / "include"),
        str(Path(cutlass_home) / "tools" / "util" / "include"),
    ],
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=1"],
        "nvcc": [
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-DENABLE_BF16",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
        ],
    },
    # CUTLASS 3.x Hopper 路径会用到 CUDA Driver API (cuDriverGetVersion,
    # cuTensorMapEncodeTiled 等), 必须显式链 libcuda (stub 或驱动里的都行).
    libraries=["cuda"],
)

setup(
    name="hopper_gmm_forward",
    version="0.1.0",
    description="Minimal forward-only Hopper grouped GEMM extracted from nezha.",
    packages=find_packages(include=["hopper_gmm_forward", "hopper_gmm_forward.*"]),
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch>=2.4"],
)

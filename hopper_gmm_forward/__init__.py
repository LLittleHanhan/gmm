"""Minimal forward-only Hopper grouped GEMM.

Usage::

    from hopper_gmm_forward import hopper_gmm_fwd
    c = hopper_gmm_fwd(a, b, batch_sizes)  # c = a @ b, grouped by batch_sizes
"""

import glob
import os
import warnings

import torch

_HAS_C = False
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_c_ext() -> bool:
    """Locate and load the compiled _C*.so via torch.ops.load_library.

    The extension registers ops under the ``hopper_gmm_fwd`` namespace using
    ``TORCH_LIBRARY``; it does NOT expose a ``PyInit__C`` symbol, so we must
    load it as a torch op library rather than importing it as a Python module.
    """
    # Match _C.so, _C.cpython-<abi>-<plat>.so, etc.
    patterns = [
        os.path.join(_PKG_DIR, "_C*.so"),
        os.path.join(_PKG_DIR, "_C*.pyd"),
    ]
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p))
    if not candidates:
        return False
    # Prefer the shortest / most specific filename if multiple exist.
    candidates.sort(key=len)
    try:
        torch.ops.load_library(candidates[0])
    except Exception as e:  # noqa: BLE001
        warnings.warn(
            f"hopper_gmm_forward: failed to load {candidates[0]}: {e}",
            RuntimeWarning,
        )
        return False
    return True


if _load_c_ext():
    _HAS_C = True
else:
    warnings.warn(
        "hopper_gmm_forward._C not built; did you run `setup.py build_ext --inplace`? "
        f"No _C*.so found in {_PKG_DIR}",
        RuntimeWarning,
    )


def hopper_gmm_fwd(a: torch.Tensor,
                   b: torch.Tensor,
                   batch_sizes: torch.Tensor,
                   trans_b: bool = False) -> torch.Tensor:
    """Forward-only grouped GEMM on Hopper (SM90a).

    Args:
        a: [num_tokens, dim]  contiguous CUDA, fp32/bf16/fp16.
        b: [num_experts, dim, dim_out] (or [num_experts, dim_out, dim] if
           ``trans_b=True``) contiguous CUDA, same dtype as ``a``.
        batch_sizes: [num_experts] int64 tensor on CPU.
        trans_b: whether ``B`` is transposed (bf16/fp16 only; fp32 handled
            internally).

    Returns:
        Tensor of shape [num_tokens, dim_out], same dtype/device as ``a``.
    """
    if not _HAS_C:
        raise RuntimeError("hopper_gmm_forward._C is not available. Build with setup.py.")
    return torch.ops.hopper_gmm_fwd.gmm(a, b, batch_sizes, trans_b)


@torch.library.register_fake("hopper_gmm_fwd::gmm")
def _hopper_gmm_fwd_fake(a, b, batch_sizes, trans_b):
    num_tokens = a.shape[0]
    n = b.shape[1] if trans_b else b.shape[2]
    return torch.empty(num_tokens, n, dtype=a.dtype, device=a.device)


__all__ = ["hopper_gmm_fwd"]

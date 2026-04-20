"""Minimal forward-only Hopper grouped GEMM.

Usage::

    from hopper_gmm_forward import hopper_gmm_fwd
    c = hopper_gmm_fwd(a, b, batch_sizes)  # c = a @ b, grouped by batch_sizes
"""

import warnings

import torch

try:
    from . import _C  # registers torch.ops.hopper_gmm_fwd.gmm
    _HAS_C = True
except ImportError as e:
    _HAS_C = False
    warnings.warn(
        "hopper_gmm_forward._C not built; did you run `setup.py build_ext --inplace`? "
        f"Import error: {e}",
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

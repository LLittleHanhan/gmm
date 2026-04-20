"""Forward correctness test for the minimal Hopper grouped GEMM."""

import torch

from hopper_gmm_forward import hopper_gmm_fwd


def ref_gmm(a, b, batch_sizes, trans_b=False):
    bs = batch_sizes.tolist()
    out, start = [], 0
    for i, size in enumerate(bs):
        rhs = b[i].t() if trans_b else b[i]
        out.append(a[start:start + size] @ rhs)
        start += size
    return torch.cat(out)


def _is_sm90() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _run_case(num_experts, token_per_expert, dim, dim_out, dtype, trans_b=False):
    torch.manual_seed(0)
    bs = torch.tensor([token_per_expert] * num_experts, dtype=torch.int64)
    total = int(bs.sum().item())

    a = (torch.randn(total, dim, device="cuda", dtype=torch.float32)
         / (dim * dim_out) ** 0.5).to(dtype)
    if trans_b:
        b = (torch.randn(num_experts, dim_out, dim, device="cuda", dtype=torch.float32)
             / (dim * dim_out) ** 0.5).to(dtype)
    else:
        b = (torch.randn(num_experts, dim, dim_out, device="cuda", dtype=torch.float32)
             / (dim * dim_out) ** 0.5).to(dtype)

    out = hopper_gmm_fwd(a, b, bs, trans_b=trans_b)
    ref = ref_gmm(a, b, bs, trans_b=trans_b)

    diff = (out.float() - ref.float()).abs().max().item()
    tol = 1e-3 if dtype == torch.float32 else 1e-2
    assert torch.allclose(out.float(), ref.float(), rtol=tol, atol=tol), \
        f"mismatch (dtype={dtype}, trans_b={trans_b}): max_diff={diff}"
    return diff


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return
    if not _is_sm90():
        print("SKIP: requires Hopper (SM90+) GPU")
        return

    cases = [
        # (num_experts, tokens_per_expert, dim, dim_out)
        (4,  128, 128, 256),
        (8,   64, 256, 512),
        (16, 128, 128, 128),
    ]

    print("=" * 60)
    print("hopper_gmm_forward — forward correctness")
    print("=" * 60)
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        for E, T, D, N in cases:
            # NN (trans_b=False) — all dtypes
            d = _run_case(E, T, D, N, dtype, trans_b=False)
            print(f"  NN dtype={str(dtype):18s} E={E:3d} T={T:4d} "
                  f"DxN={D}x{N}   max_diff={d:.3e}")
            # NT (trans_b=True) — only bf16/fp16 (fp32 forces trans_b=True internally via permute)
            if dtype != torch.float32:
                d = _run_case(E, T, D, N, dtype, trans_b=True)
                print(f"  NT dtype={str(dtype):18s} E={E:3d} T={T:4d} "
                      f"DxN={D}x{N}   max_diff={d:.3e}")

    # Edge case: some batch_sizes = 0
    bs = torch.tensor([0, 32, 0, 64, 0], dtype=torch.int64)
    dim, dim_out = 128, 256
    total = int(bs.sum().item())
    a = torch.randn(total, dim, device="cuda", dtype=torch.float16)
    b = torch.randn(len(bs), dim, dim_out, device="cuda", dtype=torch.float16)
    out = hopper_gmm_fwd(a, b, bs)
    assert out.shape == (total, dim_out)
    print(f"  zero-size experts OK  shape={tuple(out.shape)}")

    print("\nAll hopper_gmm_forward tests PASSED")


if __name__ == "__main__":
    main()

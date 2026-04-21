"""MoE forward 性能测试 (hopper grouped GEMM)

MoE problem size (按路由均分):
    E_route   = 640    # 路由专家
    E_shared  = 80     # 共享专家
    bs        = 4000
    num_token = 80
    top_k     = 3
    N         = 1280
    K         = 320

路由专家端：总 token 数 = bs * num_token * top_k = 4000*80*3 = 960_000
           均分到 640 个专家，每个专家 M_per = 960_000 / 640 = 1500
共享专家端：每个共享专家看到全部 token, 但这里按 "问题规模" 的理解，
           共享专家的 M_per = bs * num_token = 4000*80 = 320_000
           (共享专家不过 top_k, 因为每个 token 都会经过每个共享专家)

本脚本按 "路由均分 1500" 这一子场景进行性能测试, 覆盖:
  1) NN:  A[M_total, K] @ B[E, K, N]  -> D[M_total, N]
  2) NT:  A[M_total, K] @ B[E, N, K]^T -> D[M_total, N]

运行:
    cd hopper_gmm_forward_standalone
    python tests/bench_moe_forward.py
    # 或指定 dtype / 重复次数 / 变更 E:
    DTYPE=bf16 WARMUP=20 ITERS=100 python tests/bench_moe_forward.py
"""

from __future__ import annotations

import os
import statistics
import time

import torch

from hopper_gmm_forward import hopper_gmm_fwd


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
def _parse_dtype(s: str) -> torch.dtype:
    s = s.lower()
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float": torch.float32,
        "float32": torch.float32,
    }[s]


DTYPE   = _parse_dtype(os.environ.get("DTYPE", "bf16"))
WARMUP  = int(os.environ.get("WARMUP", 20))
ITERS   = int(os.environ.get("ITERS", 100))

# MoE 尺寸
BS         = int(os.environ.get("BS", 4000))
NUM_TOKEN  = int(os.environ.get("NUM_TOKEN", 80))
TOP_K      = int(os.environ.get("TOP_K", 3))
E_ROUTE    = int(os.environ.get("E_ROUTE", 640))
E_SHARED   = int(os.environ.get("E_SHARED", 80))
N_DIM      = int(os.environ.get("N", 1280))
K_DIM      = int(os.environ.get("K", 320))


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _is_sm90() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _tflops(m_total: int, n: int, k: int, ms: float) -> float:
    # 2*M*N*K FLOPs, ms -> s
    return 2.0 * m_total * n * k / (ms * 1e-3) / 1e12


def _bench_once(fn, warmup: int, iters: int) -> tuple[float, float, float]:
    """返回 (median_ms, mean_ms, min_ms)."""
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # 计时 (逐次 event, 便于统计中位数)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return (statistics.median(times), statistics.fmean(times), min(times))


def _make_inputs(num_experts: int, m_per: int, k: int, n: int,
                 dtype: torch.dtype, trans_b: bool):
    torch.manual_seed(0)
    total_m = num_experts * m_per
    # 按经验缩放，避免 fp16 累加溢出 (对性能无影响，仅为保险)
    scale = 1.0 / (k ** 0.5)
    a = (torch.randn(total_m, k, device="cuda", dtype=torch.float32) * scale).to(dtype)
    if trans_b:
        b = (torch.randn(num_experts, n, k, device="cuda", dtype=torch.float32) * scale).to(dtype)
    else:
        b = (torch.randn(num_experts, k, n, device="cuda", dtype=torch.float32) * scale).to(dtype)
    batch_sizes = torch.tensor([m_per] * num_experts, dtype=torch.int64)
    return a, b, batch_sizes


def _dtype_bytes(dtype: torch.dtype) -> int:
    return {torch.float32: 4, torch.bfloat16: 2, torch.float16: 2}[dtype]


def _run_scenario(tag: str,
                  num_experts: int,
                  m_per: int,
                  k: int,
                  n: int,
                  dtype: torch.dtype,
                  trans_b: bool,
                  warmup: int,
                  iters: int):
    total_m = num_experts * m_per
    a, b, bs = _make_inputs(num_experts, m_per, k, n, dtype, trans_b)

    # 一次正确性 sanity (只在第一次跑时比较数值量级)
    out = hopper_gmm_fwd(a, b, bs, trans_b=trans_b)
    assert out.shape == (total_m, n), out.shape
    assert torch.isfinite(out.float()).all().item(), "output contains NaN/Inf"

    def _call():
        hopper_gmm_fwd(a, b, bs, trans_b=trans_b)

    med, mean, mn = _bench_once(_call, warmup=warmup, iters=iters)

    tflops_med = _tflops(total_m, n, k, med)
    dbytes = _dtype_bytes(dtype)
    # 内存访问量估算: A + B + D
    bytes_io = (total_m * k + num_experts * k * n + total_m * n) * dbytes
    bw_gbs = bytes_io / (med * 1e-3) / 1e9

    layout = "NT" if trans_b else "NN"
    print(
        f"  [{tag}] {layout} E={num_experts:>4d} M/expert={m_per:>6d} "
        f"M_total={total_m:>8d}  K={k:>4d} N={n:>4d}  "
        f"dtype={str(dtype).replace('torch.',''):>8s}  "
        f"median={med:7.3f}ms  mean={mean:7.3f}ms  min={mn:7.3f}ms  "
        f"{tflops_med:7.2f} TFLOPS  {bw_gbs:7.1f} GB/s"
    )
    return med


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return
    if not _is_sm90():
        print("SKIP: requires Hopper (SM90+) GPU")
        return

    total_tokens_routed = BS * NUM_TOKEN * TOP_K          # 960_000
    m_per_route         = total_tokens_routed // E_ROUTE  # 1500
    assert total_tokens_routed % E_ROUTE == 0, \
        f"total_tokens={total_tokens_routed} 无法被 E_ROUTE={E_ROUTE} 整除"

    dev = torch.cuda.get_device_name()
    cap = torch.cuda.get_device_capability()

    print("=" * 90)
    print(f"hopper_gmm_forward — MoE forward 性能测试")
    print(f"device : {dev}  (sm_{cap[0]}{cap[1]})")
    print(f"dtype  : {DTYPE}    warmup={WARMUP}  iters={ITERS}")
    print(f"problem: bs={BS}  num_token={NUM_TOKEN}  top_k={TOP_K}  "
          f"E_route={E_ROUTE}  E_shared={E_SHARED}  N={N_DIM}  K={K_DIM}")
    print(f"         total_routed_tokens = {total_tokens_routed}")
    print(f"         路由均分 M/expert   = {m_per_route}")
    print("=" * 90)

    # ----- 路由专家, 路由均分 1500 ---------------------------------------
    print(">>> 路由专家 (均分 M/expert = 1500)")
    _run_scenario("route", E_ROUTE, m_per_route, K_DIM, N_DIM,
                  DTYPE, trans_b=False, warmup=WARMUP, iters=ITERS)
    if DTYPE != torch.float32:
        _run_scenario("route", E_ROUTE, m_per_route, K_DIM, N_DIM,
                      DTYPE, trans_b=True, warmup=WARMUP, iters=ITERS)

    # ----- 共享专家 ------------------------------------------------------
    # 共享专家每个都看到全部 (bs*num_token) 个 token
    if E_SHARED > 0:
        m_per_shared = BS * NUM_TOKEN
        print(f">>> 共享专家 (每专家 M = bs*num_token = {m_per_shared})")
        _run_scenario("shared", E_SHARED, m_per_shared, K_DIM, N_DIM,
                      DTYPE, trans_b=False, warmup=WARMUP, iters=ITERS)
        if DTYPE != torch.float32:
            _run_scenario("shared", E_SHARED, m_per_shared, K_DIM, N_DIM,
                          DTYPE, trans_b=True, warmup=WARMUP, iters=ITERS)

    # ----- 汇总: 端到端 (一次 route + 一次 shared) -----------------------
    print("=" * 90)
    t0 = time.perf_counter()
    for _ in range(WARMUP):
        pass
    # e2e: 这里单独再测一遍 NN 串行调用的总耗时 (等价于一次 MoE forward 的
    # 两段 GEMM, 未包括 gather/scatter / routing 等其它部分)
    a_r, b_r, bs_r = _make_inputs(E_ROUTE,  m_per_route,      K_DIM, N_DIM, DTYPE, False)
    if E_SHARED > 0:
        a_s, b_s, bs_s = _make_inputs(E_SHARED, BS * NUM_TOKEN, K_DIM, N_DIM, DTYPE, False)

    def _e2e():
        hopper_gmm_fwd(a_r, b_r, bs_r, trans_b=False)
        if E_SHARED > 0:
            hopper_gmm_fwd(a_s, b_s, bs_s, trans_b=False)

    med, mean, mn = _bench_once(_e2e, warmup=WARMUP, iters=ITERS)
    m_total = E_ROUTE * m_per_route + E_SHARED * BS * NUM_TOKEN
    tflops = _tflops(m_total, N_DIM, K_DIM, med)
    print(f"  [e2e ] route+shared NN    M_total={m_total:>8d}  "
          f"median={med:7.3f}ms  mean={mean:7.3f}ms  min={mn:7.3f}ms  "
          f"{tflops:7.2f} TFLOPS")
    print("=" * 90)


if __name__ == "__main__":
    main()

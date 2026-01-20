import argparse
import itertools
import logging
from typing import Tuple

import numpy as np
import torch

from flashinfer import autotune, mm_bf16
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import get_compute_capability


def suppress_autotune_logs():
    logging.disable(logging.CRITICAL)
    for name in ["flashinfer.jit", "flashinfer.jit.autotuner"]:
        lg = logging.getLogger(name)
        lg.disabled = True
        lg.propagate = False
        lg.setLevel(logging.CRITICAL + 1)
        for handler in list(lg.handlers):
            lg.removeHandler(handler)


def bench_shape(m: int, n: int, k: int, repeat_ms: int) -> Tuple[float, float]:
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    # Autotune once to pick a tactic, then benchmark without extra logging.
    with autotune():
        mm_bf16(a, b.T, out=out, out_dtype=torch.bfloat16, backend="cutlass")

    def run(a_, b_t_, out_):
        mm_bf16(a_, b_t_, out=out_, out_dtype=torch.bfloat16, backend="cutlass")

    times_ms = bench_gpu_time(
        fn=run,
        input_args=(a, b.T, out),
        dry_run_time_ms=50,
        repeat_time_ms=repeat_ms,
        cold_l2_cache=False,  # avoid extra overhead; we just want relative kernel speed
    )
    median_ms = float(np.median(times_ms))
    tflops = (2.0 * m * n * k) / (median_ms * 1e-3) / 1e12
    return median_ms, tflops


def main():
    suppress_autotune_logs()

    parser = argparse.ArgumentParser(
        description="BF16 CUTLASS GEMM small-batch benchmark (transpose trick)."
    )
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=[1, 8, 16, 32, 64, 128],
        help="M dimension values to benchmark",
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[1536, 7168, 24576],
        help="N dimension values to benchmark",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1536, 7168],
        help="K dimension values to benchmark",
    )
    parser.add_argument(
        "--repeat_ms", type=int, default=200, help="Target total repeat time."
    )
    args = parser.parse_args()

    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 10:
        print("Warning: intended for SM100/103; running anyway.")

    ms = args.m_values
    ns = args.n_values
    ks = args.k_values

    print("m     n      k      median_ms    TFLOP/s")
    print("----------------------------------------")
    for m, n, k in itertools.product(ms, ns, ks):
        median_ms, tflops = bench_shape(m, n, k, args.repeat_ms)
        print(f"{m:4d}  {n:6d}  {k:6d}  {median_ms:10.3f}   {tflops:8.2f}")


if __name__ == "__main__":
    main()

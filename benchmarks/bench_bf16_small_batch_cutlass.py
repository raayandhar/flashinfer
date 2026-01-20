import argparse
import time

import torch

from flashinfer import autotune, mm_bf16
from flashinfer.utils import get_compute_capability


def parse_int_list(arg: str):
    return [int(x) for x in arg.split(",")] if arg else []


def bench_shape(m: int, n: int, k: int, warmup: int, iters: int):
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    # One autotuned run to pick a tactic (cached for subsequent calls).
    with autotune():
        mm_bf16(a, b.T, out=out, out_dtype=torch.bfloat16, backend="cutlass")

    for _ in range(warmup):
        mm_bf16(a, b.T, out=out, out_dtype=torch.bfloat16, backend="cutlass")

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        mm_bf16(a, b.T, out=out, out_dtype=torch.bfloat16, backend="cutlass")
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_us = (end - start) * 1e6 / iters
    tflops = (2.0 * m * n * k) / (avg_us * 1e-6) / 1e12
    return avg_us, tflops


def main():
    parser = argparse.ArgumentParser(
        description="Simple BF16 CUTLASS GEMM benchmark (small batches)."
    )
    parser.add_argument(
        "--m", type=str, default="1,8,16,32,64", help="Comma-separated M sizes."
    )
    parser.add_argument(
        "--n", type=str, default="1024", help="Comma-separated N sizes."
    )
    parser.add_argument(
        "--k", type=str, default="1024", help="Comma-separated K sizes."
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=50, help="Measured iterations.")
    args = parser.parse_args()

    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 10:
        print("Warning: This benchmark is intended for SM100/103. Running anyway.")

    ms = parse_int_list(args.m)
    ns = parse_int_list(args.n)
    ks = parse_int_list(args.k)

    for m in ms:
        for n in ns:
            for k in ks:
                avg_us, tflops = bench_shape(m, n, k, args.warmup, args.iters)
                print(
                    f"m={m:5d} n={n:5d} k={k:5d} | avg={avg_us:8.2f} us | {tflops:6.2f} TFLOP/s"
                )


if __name__ == "__main__":
    main()

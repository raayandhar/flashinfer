"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Benchmark script for BF16 GEMM small-batch optimization.

This script compares performance of the standard BF16 GEMM kernel vs the
low-latency kernel optimized for small batch sizes (M <= 32).

The optimization uses a transpose trick: instead of computing D = A @ B,
we compute D^T = B^T @ A^T, which allows using smaller tile sizes and
better CTA utilization for small M values.

Usage:
    python bench_bf16_gemm_small_batch.py
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F

from flashinfer import mm_bf16, autotune
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import get_compute_capability


def check_correctness(m, n, k, out_dtype):
    """Verify that the output is numerically correct."""
    torch.manual_seed(42)
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)

    reference = torch.mm(a, b.T)
    out = torch.empty([m, n], device="cuda", dtype=out_dtype)

    with autotune():
        mm_bf16(a, b.T, None, False, out, out_dtype, "cutlass")

    cos_sim = F.cosine_similarity(
        reference.reshape(-1).float(), out.reshape(-1).float(), dim=0
    )
    return cos_sim.item()


def bench_bf16_gemm(m, n, k, out_dtype, repeat_iters=100, warmup_iters=10):
    """Benchmark BF16 GEMM with given dimensions."""
    torch.manual_seed(42)
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    out = torch.empty([m, n], device="cuda", dtype=out_dtype)

    # Warmup and autotune
    with autotune():
        for _ in range(warmup_iters):
            mm_bf16(a, b.T, None, False, out, out_dtype, "cutlass")

    torch.cuda.synchronize()

    # Benchmark
    measurements = bench_gpu_time(
        lambda: mm_bf16(a, b.T, None, False, out, out_dtype, "cutlass"),
        repeat_iters=repeat_iters,
    )
    return measurements


def bench_torch_mm(m, n, k, out_dtype, repeat_iters=100, warmup_iters=10):
    """Benchmark PyTorch's native mm for comparison."""
    torch.manual_seed(42)
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(warmup_iters):
        _ = torch.mm(a, b.T)

    torch.cuda.synchronize()

    # Benchmark
    measurements = bench_gpu_time(
        lambda: torch.mm(a, b.T),
        repeat_iters=repeat_iters,
    )
    return measurements


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BF16 GEMM small-batch optimization"
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
        "--out-dtype",
        type=str,
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="Output dtype",
    )
    parser.add_argument(
        "--repeat-iters", type=int, default=100, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--check-correctness",
        action="store_true",
        help="Also check numerical correctness",
    )
    parser.add_argument(
        "--compare-torch",
        action="store_true",
        help="Also compare with PyTorch's native mm",
    )
    args = parser.parse_args()

    # Check compute capability
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    print(f"Detected GPU: SM{compute_capability_number}")

    if not mm_bf16.is_compute_capability_supported(compute_capability_number):
        print(
            f"mm_bf16 CUTLASS backend not supported on SM{compute_capability_number}. "
            "Requires SM100 or SM103."
        )
        return

    out_dtype = torch.bfloat16 if args.out_dtype == "bfloat16" else torch.float16

    print("\n" + "=" * 100)
    print(
        f"{'M':>6} | {'N':>6} | {'K':>6} | {'FlashInfer (us)':>15} | "
        f"{'TFLOPs/s':>10} | {'Small Batch':>12}"
        + (" | {'PyTorch (us)':>15} | {'Speedup':>8}" if args.compare_torch else "")
        + (" | {'Cos Sim':>8}" if args.check_correctness else "")
    )
    print("=" * 100)

    for m in args.m_values:
        for n in args.n_values:
            for k in args.k_values:
                # Benchmark FlashInfer
                try:
                    measurements = bench_bf16_gemm(
                        m, n, k, out_dtype, num_iters=args.repeat_iters
                    )
                    median_us = (
                        np.median(measurements) * 1000
                    )  # Convert to microseconds
                    tflops = 2 * m * n * k * 1e-12 / (median_us * 1e-6)
                    # Low-latency kernel used when 8 <= m <= 32
                    is_small_batch = "Yes" if 8 <= m <= 32 else "No"

                    result_str = (
                        f"{m:>6} | {n:>6} | {k:>6} | {median_us:>15.2f} | "
                        f"{tflops:>10.2f} | {is_small_batch:>12}"
                    )

                    # Compare with PyTorch if requested
                    if args.compare_torch:
                        torch_measurements = bench_torch_mm(
                            m, n, k, out_dtype, num_iters=args.repeat_iters
                        )
                        torch_median_us = np.median(torch_measurements) * 1000
                        speedup = torch_median_us / median_us
                        result_str += f" | {torch_median_us:>15.2f} | {speedup:>8.2f}x"

                    # Check correctness if requested
                    if args.check_correctness:
                        cos_sim = check_correctness(m, n, k, out_dtype)
                        result_str += f" | {cos_sim:>8.4f}"

                    print(result_str)

                except Exception as e:
                    print(f"{m:>6} | {n:>6} | {k:>6} | ERROR: {e}")

    print("=" * 100)
    print(
        "\nNote: 'Small Batch' indicates whether the low-latency kernel (8 <= m <= 32) was used."
    )
    print("      Values m < 8 use the standard kernel to meet alignment requirements.")


if __name__ == "__main__":
    main()

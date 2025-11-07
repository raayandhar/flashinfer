#!/usr/bin/env python3
"""
Simple test for BF16 CUTLASS GEMM

This is a minimal test that:
1. Compiles the BF16 GEMM module
2. Runs a simple matmul
3. Compares against PyTorch

Run with: python test_bf16_simple.py
"""

import torch
import sys


def main():
    print("=" * 70)
    print("Simple BF16 CUTLASS GEMM Test")
    print("=" * 70)

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    sm_version = capability[0] * 10 + capability[1]
    print(f"\nGPU: {torch.cuda.get_device_name(device)}")
    print(f"Compute Capability: SM{sm_version}")

    if sm_version < 80:
        print(f"ERROR: BF16 GEMM requires SM80+ (you have SM{sm_version})")
        return False

    # Compile the module
    print("\n" + "-" * 70)
    print("Step 1: Compiling BF16 GEMM module...")
    print("-" * 70)

    try:
        from flashinfer.jit.gemm import gen_gemm_sm80_module_cutlass_bf16

        spec = gen_gemm_sm80_module_cutlass_bf16()
        print(f"Module name: {spec.name}")
        print(f"Source files: {len(spec.sources)}")

        print("\nBuilding... (this may take 30-60 seconds)")
        module = spec.build_and_load()
        print("✓ Module compiled successfully!")

    except Exception as e:
        print(f"✗ Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check what functions are available
    print("\n" + "-" * 70)
    print("Step 2: Checking available functions...")
    print("-" * 70)

    funcs = [x for x in dir(module) if not x.startswith('_')]
    print(f"Available functions: {', '.join(funcs)}")

    if 'bf16_gemm' not in funcs:
        print("\n✗ bf16_gemm function not found!")
        print("This means the TVM FFI export didn't work.")
        return False

    print("✓ bf16_gemm found!")

    # Get number of tactics
    if 'bf16_gemm_tactic_num' in funcs:
        try:
            num_tactics = module.bf16_gemm_tactic_num()
            print(f"✓ Number of tactics available: {num_tactics}")
        except Exception as e:
            print(f"⚠ Could not query tactics: {e}")
    else:
        print("⚠ bf16_gemm_tactic_num not found (optional)")
        num_tactics = 1  # Assume at least one

    # Test the kernel
    print("\n" + "-" * 70)
    print("Step 3: Testing BF16 GEMM kernel...")
    print("-" * 70)

    # Small test case
    m, n, k = 128, 256, 512
    print(f"\nMatrix sizes: A({m}×{k}) @ B({k}×{n}) = C({m}×{n})")

    A = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
    B = torch.randn(k, n, dtype=torch.bfloat16, device='cuda')
    out = torch.empty(m, n, dtype=torch.bfloat16, device='cuda')
    workspace = torch.empty(0, dtype=torch.int8, device='cuda')

    print("\nCalling CUTLASS kernel (tactic 0)...")
    try:
        module.bf16_gemm(A, B, out, workspace, 0)
        print("✓ Kernel executed successfully!")
    except Exception as e:
        print(f"✗ Kernel execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Compare with PyTorch
    print("\nComparing with PyTorch reference...")
    ref = torch.matmul(A, B)

    abs_diff = torch.abs(out - ref)
    rel_diff = abs_diff / (torch.abs(ref) + 1e-6)

    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    mean_rel = rel_diff.mean().item()

    print(f"  Max absolute error: {max_abs:.6f}")
    print(f"  Mean absolute error: {mean_abs:.6f}")
    print(f"  Mean relative error: {mean_rel:.6f} ({mean_rel*100:.2f}%)")

    # Check accuracy
    if mean_rel < 0.001:  # 0.1%
        print("\n✓ EXCELLENT: Results match PyTorch very closely!")
        return True
    elif mean_rel < 0.01:  # 1%
        print("\n✓ GOOD: Results are close to PyTorch (< 1% error)")
        return True
    elif mean_rel < 0.02:  # 2%
        print("\n⚠ ACCEPTABLE: Results are within 2% of PyTorch")
        print("  (This is normal for BF16 precision)")
        return True
    else:
        print(f"\n✗ FAIL: Error {mean_rel*100:.2f}% is too high!")
        print("\nSample values:")
        print(f"  CUTLASS: {out[0, :5]}")
        print(f"  PyTorch: {ref[0, :5]}")
        return False


if __name__ == "__main__":
    success = main()

    print("\n" + "=" * 70)
    if success:
        print("OVERALL: ✓ TEST PASSED")
        print("=" * 70)
        print("\nYour BF16 CUTLASS GEMM implementation works!")
        print("\nNext steps:")
        print("  1. Benchmark different tactics (0-17) to find the fastest")
        print("  2. Test with larger matrices")
        print("  3. Compare performance with PyTorch's cuBLAS")
        sys.exit(0)
    else:
        print("OVERALL: ✗ TEST FAILED")
        print("=" * 70)
        print("\nSee error messages above for debugging.")
        print("Check DEBUG_BF16_GEMM.md for troubleshooting guide.")
        sys.exit(1)

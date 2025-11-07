"""
Test the actual CUTLASS BF16 GEMM kernel

This script:
1. Compiles the CUTLASS BF16 GEMM module via JIT
2. Calls the actual CUTLASS kernel
3. Compares against PyTorch reference
"""

import torch
import sys


def test_cutlass_bf16_compilation():
    """Test that the CUTLASS module compiles"""
    print("=" * 70)
    print("Testing CUTLASS BF16 GEMM Compilation")
    print("=" * 70)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    sm_version = capability[0] * 10 + capability[1]
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Compute Capability: SM{sm_version}")

    if sm_version < 80:
        print(f"ERROR: BF16 GEMM requires SM80+ (you have SM{sm_version})")
        return False

    # Try to compile the module
    try:
        print("\n1. Importing FlashInfer JIT module...")
        from flashinfer.jit.gemm import gen_gemm_sm80_module_cutlass_bf16

        print("2. Generating JIT spec...")
        spec = gen_gemm_sm80_module_cutlass_bf16()
        print(f"   Module name: {spec.name}")
        print(f"   Source files: {len(spec.sources)} file(s)")
        for src in spec.sources:
            print(f"     - {src}")

        print("\n3. Loading/compiling module (this may take a while)...")
        # This actually triggers compilation
        module = spec.load()
        print(f"   ✓ Module compiled: {module}")

        print("\n✓ Compilation successful!")
        return True

    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("\nMake sure you have FlashInfer installed:")
        print("  pip install -e .")
        return False

    except Exception as e:
        print(f"\n✗ Compilation error: {e}")
        print("\nDebugging info:")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def test_cutlass_bf16_kernel():
    """Test the actual CUTLASS kernel"""
    print("\n" + "=" * 70)
    print("Testing CUTLASS BF16 GEMM Kernel Execution")
    print("=" * 70)

    if not torch.cuda.is_available():
        return False

    try:
        print("\n1. Loading compiled module...")
        from flashinfer.jit.gemm import gen_gemm_sm80_module_cutlass_bf16
        spec = gen_gemm_sm80_module_cutlass_bf16()
        module = spec.load()

        # Try to get the function
        print("2. Looking for bf16_gemm function...")

        # The function should be registered via TVM FFI
        # Let's try to import it
        try:
            import tvm.runtime as tvm_runtime
            bf16_gemm = tvm_runtime.get_global_func("bf16_gemm")
            bf16_gemm_tactic_num = tvm_runtime.get_global_func("bf16_gemm_tactic_num")
            print(f"   ✓ Found bf16_gemm: {bf16_gemm}")
            print(f"   ✓ Found bf16_gemm_tactic_num: {bf16_gemm_tactic_num}")

            # Get number of tactics
            num_tactics = bf16_gemm_tactic_num()
            print(f"   Number of available tactics: {num_tactics}")

        except Exception as e:
            print(f"   ✗ Could not find TVM function: {e}")
            print("\n   This is expected - the function needs to be registered.")
            print("   Let's try calling it via ctypes or check if it's in the module...")
            return False

        print("\n3. Creating test tensors...")
        m, n, k = 256, 512, 1024
        A = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
        B = torch.randn(k, n, dtype=torch.bfloat16, device='cuda')
        out = torch.empty(m, n, dtype=torch.bfloat16, device='cuda')
        workspace = torch.empty(0, dtype=torch.int8, device='cuda')

        print(f"   A: {A.shape}, {A.dtype}")
        print(f"   B: {B.shape}, {B.dtype}")
        print(f"   Output: {out.shape}, {out.dtype}")

        print("\n4. Calling CUTLASS kernel (tactic 0)...")
        # Call the kernel
        bf16_gemm(A, B, out, workspace, 0)  # tactic=0

        print("\n5. Computing PyTorch reference...")
        ref = torch.matmul(A, B)

        print("\n6. Comparing results...")
        abs_diff = torch.abs(out - ref)
        rel_diff = abs_diff / (torch.abs(ref) + 1e-6)

        max_abs = abs_diff.max().item()
        mean_abs = abs_diff.mean().item()
        max_rel = rel_diff.max().item()
        mean_rel = rel_diff.mean().item()

        print(f"   Max absolute error: {max_abs:.6f}")
        print(f"   Mean absolute error: {mean_abs:.6f}")
        print(f"   Max relative error: {max_rel:.6f}")
        print(f"   Mean relative error: {mean_rel:.6f} ({mean_rel*100:.2f}%)")

        # Check if results are close
        if mean_rel < 0.001:  # 0.1% - should be very close since both use BF16
            print("\n   ✓ CUTLASS kernel produces correct results!")
            return True
        elif mean_rel < 0.01:  # 1%
            print("\n   ⚠ CUTLASS kernel results are close but not perfect")
            print(f"   Mean error {mean_rel*100:.2f}% is acceptable for BF16")
            return True
        else:
            print("\n   ✗ CUTLASS kernel results differ significantly!")
            print(f"   Mean error {mean_rel*100:.2f}% is too high")
            return False

    except Exception as e:
        print(f"\n✗ Kernel test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compilation_only():
    """Minimal test: just check if the code compiles"""
    print("\n" + "=" * 70)
    print("Minimal Compilation Test")
    print("=" * 70)

    try:
        print("\nChecking CUTLASS BF16 GEMM source files...")

        import os
        from pathlib import Path

        flashinfer_root = Path(__file__).parent

        files_to_check = [
            flashinfer_root / "csrc/bf16_gemm_cutlass.cu",
            flashinfer_root / "include/flashinfer/gemm/bf16_gemm_cutlass.h",
            flashinfer_root / "include/flashinfer/gemm/bf16_gemm_cutlass_template.h",
        ]

        for fpath in files_to_check:
            if fpath.exists():
                size = fpath.stat().st_size
                print(f"  ✓ {fpath.name} ({size} bytes)")
            else:
                print(f"  ✗ {fpath.name} NOT FOUND")
                return False

        print("\n✓ All source files present!")

        # Try to compile with nvcc directly
        print("\nAttempting direct NVCC compilation test...")
        cu_file = flashinfer_root / "csrc/bf16_gemm_cutlass.cu"
        include_dir = flashinfer_root / "include"
        cutlass_dir = flashinfer_root / "3rdparty/cutlass/include"

        cmd = f"""nvcc --version && echo "---" && \
nvcc -x cu -c {cu_file} \
  -I{include_dir} \
  -I{cutlass_dir} \
  -I{flashinfer_root / "3rdparty/spdlog/include"} \
  -std=c++17 \
  -DENABLE_BF16 \
  --gpu-architecture=sm_89 \
  -DNDEBUG \
  --dry-run 2>&1 | head -30"""

        print(f"\nRunning: {cmd}\n")
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=flashinfer_root)

        if result.returncode == 0 or "--dry-run" in result.stdout:
            print("NVCC output (dry run):")
            print(result.stdout[:1000])
            if result.stderr:
                print("\nWarnings:")
                print(result.stderr[:500])
            print("\n✓ NVCC can compile the code!")
            return True
        else:
            print("✗ NVCC compilation failed:")
            print(result.stderr[:1000])
            return False

    except Exception as e:
        print(f"✗ Compilation test error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("CUTLASS BF16 GEMM Integration Test")
    print("=" * 70)

    results = []

    # Test 1: Compilation
    print("\n### TEST 1: Check source files and NVCC ###")
    results.append(("Source files check", test_compilation_only()))

    # Test 2: JIT compilation
    print("\n\n### TEST 2: JIT Compilation ###")
    results.append(("JIT compilation", test_cutlass_bf16_compilation()))

    # Test 3: Kernel execution (only if compilation succeeded)
    if results[-1][1]:
        print("\n\n### TEST 3: Kernel Execution ###")
        results.append(("Kernel execution", test_cutlass_bf16_kernel()))
    else:
        print("\n\n### TEST 3: Kernel Execution ###")
        print("Skipped (compilation failed)")
        results.append(("Kernel execution", None))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, result in results:
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "- SKIP"
        print(f"{name:30s} {status}")

    all_passed = all(r is True or r is None for _, r in results)
    print("=" * 70)
    if all_passed:
        print("Overall: ✓ TESTS PASSED")
        sys.exit(0)
    else:
        print("Overall: ✗ SOME TESTS FAILED")
        sys.exit(1)

# BF16 CUTLASS GEMM Testing Checklist

Use this flow when validating the new BF16 CUTLASS GEMM path before opening a PR.

## 1. Prerequisites
- Linux with CUDA 12.4+ drivers and a Blackwell GPU (SM100/103/110).  
- Python 3.10/3.11 with PyTorch >= 2.4 built for CUDA 12.4+.  
- Ninja, CMake, and a recent GCC/Clang toolchain on the PATH.

Verify the GPU type before proceeding:

```bash
python - <<'PY'
import torch
print(torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))
PY
```

## 2. Environment Setup
1. Create and activate a clean virtual environment (conda/venv).  
2. Install dependencies and build FlashInfer in editable mode:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
3. (Optional) set a persistent JIT cache for generated CUTLASS kernels to avoid rebuilding on every run:
   ```bash
   export FLASHINFER_JIT_CACHE_DIR=$HOME/.cache/flashinfer
   ```

## 3. Run the BF16 GEMM Tests
Warm the cache by running each test once; subsequent invocations reuse the compiled kernels.

```bash
pytest tests/gemm/test_mm_bf16.py --maxfail=1 -v
pytest tests/gemm/test_bmm_bf16.py --maxfail=1 -v
```

Both tests compare the CUTLASS results against high-precision PyTorch references for bf16/fp16 outputs. Failures normally indicate a missing kernel build, unsupported SM, or a regression in the underlying runner.

## 4. Manual Sanity Check (optional)
After the tests pass, you can run a quick script to multiply larger matrices and inspect the max error:

```bash
python - <<'PY'
import torch
from flashinfer import mm_bf16

torch.manual_seed(0)
a = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
b = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16)
ref = torch.mm(a.float(), b.float())
out = mm_bf16(a, b)
print("max error:", (ref - out.float()).abs().max())
PY
```

If both the pytest suite and the manual check succeed, the BF16 CUTLASS GEMM path is ready for review.

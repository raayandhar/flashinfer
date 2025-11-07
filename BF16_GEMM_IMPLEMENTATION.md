# BF16 CUTLASS GEMM Implementation for SM89

## Overview

This document explains the BF16 GEMM implementation and how it differs from the existing FP4 and FP8 implementations.

## Architecture Differences

### BF16 (This Implementation) - SM80+

**Target Hardware:**
- SM80: A100 (Ampere)
- SM86: RTX 30xx (Ampere)
- SM89: RTX 40xx, L40S (Ada Lovelace) ← **Your target**
- SM90: H100 (Hopper)

**CUTLASS Version:** CUTLASS 2.x API
- Uses `cutlass::gemm::kernel::DefaultGemm`
- Uses `cutlass::gemm::device::GemmUniversal`
- Traditional multi-stage pipeline (2-4 stages)

**Key Features:**
```cpp
Element Types:
  Input A:  BF16 (cutlass::bfloat16_t)
  Input B:  BF16 (cutlass::bfloat16_t)
  Output:   BF16 or FP16 (template parameter)
  Accumulator: FP32

Tile Configurations (for SM80/89):
  - 128x128x64 with 64x64x64 warp
  - 128x128x64 with 64x32x64 warp
  - 128x64x64 with 64x32x64 warp
  - 64x128x64 with 64x32x64 warp
  - 128x256x64 with 64x64x64 warp
  - 64x128x64 with 32x64x64 warp

Instruction Shape: 16x8x16 (MMA.16816)
```

### FP8 Implementation - SM100 (Blackwell)

**Target Hardware:** SM100+ only (B200, future GPUs)

**CUTLASS Version:** CUTLASS 3.x API
- Uses `cutlass::epilogue::collective::collective_builder`
- Uses `cutlass::gemm::collective::collective_builder`
- TMA (Tensor Memory Accelerator) support
- Cluster shapes (1x1x1, 2x1x1, 2x2x1, etc.)

**Key Features:**
```cpp
Element Types:
  Input A:  FP8 E4M3 (__nv_fp8_e4m3)
  Input B:  FP8 E4M3 (__nv_fp8_e4m3)
  Output:   BF16 or FP16
  Accumulator: FP32
  Scaling:  Per-tensor FP32 scale factors

Tile Configurations (bytes-based):
  - 64x64x128B
  - 64x128x128B
  - 64x256x128B
  - 128x64x128B
  - 128x128x128B
  - 128x256x128B

Advanced Features:
  - Per-tensor scaling (scale_a, scale_b)
  - TMA warp-specialized scheduling
  - Cluster launch
```

### FP4 Implementation - SM100/SM120

**Target Hardware:** SM100+ (Blackwell and beyond)

**CUTLASS Version:** CUTLASS 3.x API

**Key Features:**
```cpp
Element Types:
  Input A:  FP4 (NVFP4 or MXFP4)
  Input B:  FP4 (NVFP4 or MXFP4)
  Output:   BF16 or FP16
  Accumulator: FP32
  Quantization: Per-group or per-tensor

Tile Configurations:
  SM100: 128x64x128, 128x256x128, etc.
  SM120: 128x128x128 (fixed)

Advanced Features:
  - Multiple FP4 formats (NVFP4, MXFP4)
  - Group-wise quantization
  - Ultra-low precision (4-bit)
```

## Key Differences Summary

| Feature | BF16 (SM89) | FP8 (SM100) | FP4 (SM100) |
|---------|-------------|-------------|-------------|
| **CUTLASS API** | 2.x | 3.x | 3.x |
| **Input Precision** | 16-bit | 8-bit | 4-bit |
| **Hardware** | Ampere/Ada/Hopper | Blackwell+ | Blackwell+ |
| **Scaling** | None | Per-tensor | Per-group |
| **Tile Config** | Shape-based | Byte-based | Byte-based |
| **TMA Support** | No | Yes | Yes |
| **Cluster Launch** | No | Yes | Yes |
| **Complexity** | Simple | Medium | High |

## Why Different APIs?

### CUTLASS 2.x (BF16 - Our Implementation)
- **Mature, stable API** for SM80-SM90
- Traditional multi-stage pipeline
- Direct kernel configuration
- No special hardware features required
- **Pros:** Simple, well-tested, portable
- **Cons:** No TMA, no cluster launch, manual tuning

### CUTLASS 3.x (FP8/FP4)
- **Modern API** for SM90+, required for SM100+
- Builder pattern for kernel construction
- Leverages new hardware: TMA, cluster launch, FP8 tensor cores
- **Pros:** Better performance on new hardware, automated optimizations
- **Cons:** More complex, requires newer GPUs, larger compile times

## Testing Your Implementation

### 1. Basic Test (PyTorch Reference)
```bash
cd /home/user/flashinfer
python test_bf16_gemm.py
```

This will:
- Check your GPU compute capability
- Run reference BF16 matmuls with PyTorch
- Compare accuracy vs FP32
- Benchmark different sizes

### 2. Full Integration Test (After Compilation)

First, compile the module:
```python
from flashinfer.jit.gemm import gen_gemm_sm80_module_cutlass_bf16

# This will JIT compile the CUDA code
spec = gen_gemm_sm80_module_cutlass_bf16()
module = spec.load()  # Actual compilation happens here
```

Then use it:
```python
import torch
from flashinfer import bf16_gemm  # Once registered

m, n, k = 256, 512, 1024
A = torch.randn(m, k, dtype=torch.bfloat16, device='cuda')
B = torch.randn(k, n, dtype=torch.bfloat16, device='cuda')
out = torch.empty(m, n, dtype=torch.bfloat16, device='cuda')
workspace = torch.empty(0, dtype=torch.int8, device='cuda')

# Call the CUTLASS kernel
bf16_gemm(A, B, out, workspace, tactic=0)

# Compare with PyTorch
ref = torch.matmul(A, B)
print(f"Max diff: {torch.max(torch.abs(out - ref))}")
```

### 3. Compilation Test
```bash
# Try to build the module
cd /home/user/flashinfer
python -c "
from flashinfer.jit.gemm import gen_gemm_sm80_module_cutlass_bf16
import torch

if torch.cuda.is_available():
    print('Generating JIT spec...')
    spec = gen_gemm_sm80_module_cutlass_bf16()
    print(f'Module name: {spec.name}')
    print(f'Source files: {spec.source_files}')
    print('✓ JIT spec created successfully')
else:
    print('⚠ CUDA not available')
"
```

## Potential Issues

### 1. **Missing CUTLASS Headers**
The code uses:
```cpp
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/kernel/default_gemm.h"
```

Make sure CUTLASS is in your include path (it should be via `3rdparty/cutlass`).

### 2. **Template Instantiation**
The code instantiates templates for:
- `CutlassBf16GemmRunner<__nv_bfloat16>`
- `CutlassBf16GemmRunner<half>`

If you only need one, you can remove the other from `bf16_gemm_cutlass.cu`.

### 3. **Architecture Targeting**
The current code supports SM80-90. For SM89 specifically:
- No code changes needed! SM89 uses the same tensor core instructions as SM80
- The compiler will optimize for your specific architecture

### 4. **Workspace Size**
Unlike FP8/FP4, BF16 GEMM typically needs minimal workspace. The implementation still queries it for consistency.

## Performance Expectations

On SM89 (RTX 40xx), you should expect:
- **~150-250 TFLOPS** for large matrices (2048x2048x2048)
- **Better than FP32** (2-3x faster)
- **Slightly slower than FP16** on tensor cores (same ops, but BF16 has better numerical range)

The actual performance depends on:
- Tile configuration (tune with `tactic` parameter)
- Matrix size (larger = better utilization)
- Pipeline stages (2-4, try all)

## Next Steps

1. **Run basic test:** `python test_bf16_gemm.py`
2. **Check compilation:** Make sure CUTLASS headers are found
3. **Profile:** Use `nsys` or `ncu` to see actual kernel performance
4. **Tune:** Try different tactics (tile configs) for your workload
5. **Compare:** Benchmark against PyTorch's cublas BF16 matmul

## Differences from Tutorial Implementations

Many CUTLASS tutorials show simpler examples. This implementation is **production-grade**:

✅ Batched GEMM support
✅ Workspace management
✅ Multiple tile configurations for auto-tuning
✅ Error handling
✅ TVM FFI integration for Python
✅ Type safety (template parameters)

This is why it's longer than a simple tutorial!

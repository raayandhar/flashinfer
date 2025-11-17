"""
Test FP8 GEMM with the same dimensions as the failing BF16 test.
This will help us understand why FP8 works but BF16 doesn't.
"""
import torch
import torch.nn.functional as F
from flashinfer import autotune, bmm_fp8
from flashinfer.utils import get_compute_capability
from tests.utils_fp8 import to_float8

# Use the exact same dimensions as the failing BF16 test
b, m, n, k = 1, 48, 80, 64
input_dtype = torch.float8_e4m3fn
mat2_dtype = torch.float8_e4m3fn
res_dtype = torch.bfloat16

compute_capability = get_compute_capability(torch.device("cuda"))
print(f"Compute capability: sm{compute_capability[0]}{compute_capability[1]}")

torch.manual_seed(7)
print(f"\n{'='*60}")
print(f"Testing FP8 GEMM with dimensions: b={b}, m={m}, n={n}, k={k}")
print(f"{'='*60}\n")

# Create input tensors
input = torch.randn([b, m, k], device="cuda", dtype=torch.bfloat16)
mat2 = torch.randn([b, n, k], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)

print(f"[Python] input shape: {input.shape}, strides: {input.stride()}")
print(f"[Python] mat2 shape: {mat2.shape}, strides: {mat2.stride()}")

# Convert to FP8
input_fp8, input_inv_s = to_float8(input, dtype=input_dtype)
mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=mat2_dtype)

reference = torch.bmm(input, mat2)
print(f"[Python] Reference output shape: {reference.shape}")

res = torch.empty([b, m, n], device="cuda", dtype=res_dtype)

print(f"\n{'='*60}")
print(f"Calling bmm_fp8 without autotune...")
print(f"{'='*60}\n")

try:
    bmm_fp8(
            input_fp8,
            mat2_fp8,
            input_inv_s,
            mat2_inv_s,
            res_dtype,
            res,
            backend="cutlass",
    )
    
    cos_sim = F.cosine_similarity(reference.reshape(-1), res.reshape(-1), dim=0)
    print(f"\n{'='*60}")
    print(f"SUCCESS! Cosine similarity: {cos_sim}")
    print(f"{'='*60}")
except Exception as e:
    print(f"\n{'='*60}")
    print(f"FAILED with error: {e}")
    print(f"{'='*60}")
    import traceback
    traceback.print_exc()


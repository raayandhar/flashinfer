"""
Test FP8 GEMM with the same dimensions as the failing BF16 test.
This will help us understand why FP8 works but BF16 doesn't.
"""
import torch
import torch.nn.functional as F
from flashinfer import autotune, bmm_bf16
from flashinfer.utils import get_compute_capability

# Use the exact same dimensions as the failing BF16 test
b, m, n, k = 1, 48, 80, 64

compute_capability = get_compute_capability(torch.device("cuda"))
print(f"Compute capability: sm{compute_capability[0]}{compute_capability[1]}")

torch.manual_seed(7)
print(f"\n{'='*60}")
print(f"Testing BF16 GEMM with dimensions: b={b}, m={m}, n={n}, k={k}")
print(f"{'='*60}\n")

# Create input tensors
input = torch.randn([b, m, k], device="cuda", dtype=torch.bfloat16)
mat2 = torch.randn([b, n, k], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)

print(f"[Python] input shape: {input.shape}, strides: {input.stride()}")
print(f"[Python] mat2 shape: {mat2.shape}, strides: {mat2.stride()}")

reference = torch.bmm(input, mat2)
print(f"[Python] Reference output shape: {reference.shape}")

res = torch.empty([b, m, n], device="cuda", dtype=torch.bfloat16)

print(f"\n{'='*60}")
print(f"Calling bmm_bf16 without autotune...")
print(f"{'='*60}\n")

try:
    bmm_bf16(
        input,
        mat2,
        res,
        out_dtype=torch.bfloat16,
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


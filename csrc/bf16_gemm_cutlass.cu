/*
 * Copyright (c) 2025, FlashInfer.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "flashinfer/gemm/bf16_gemm_cutlass.h"
#include "flashinfer/gemm/bf16_gemm_cutlass_template.h"
#include "flashinfer/gemm/cutlass_gemm_configs.h"
#define FLASHINFER_TVM_FFI_SKIP_TENSOR_USING
#include "tvm_ffi_utils.h"
#undef FLASHINFER_TVM_FFI_SKIP_TENSOR_USING

using flashinfer::gemm::CutlassBf16GemmRunner;
using flashinfer::gemm::CutlassBf16GemmRunnerInterface;
using flashinfer::gemm::CutlassGemmConfig;
using flashinfer::gemm::CutlassTileConfigSM100;
using flashinfer::gemm::EpilogueScheduleType;
using flashinfer::gemm::MainloopScheduleType;

namespace flashinfer {
namespace gemm {
namespace {
__device__ __constant__ float kCutlassBf16UnitScalar = 1.0f;
}  // namespace

float const* get_bf16_unit_scalar_device_ptr() {
  void* symbol_addr = nullptr;
  cudaError_t status = cudaGetSymbolAddress(&symbol_addr, kCutlassBf16UnitScalar);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("[Bf16 Gemm Runner] failed to locate device scalar: ") +
        cudaGetErrorString(status));
  }
  return static_cast<float const*>(symbol_addr);
}

template class CutlassBf16GemmRunner<__nv_bfloat16>;
template class CutlassBf16GemmRunner<half>;
}  // namespace gemm
}  // namespace flashinfer

namespace torch_ext {

using tvm::ffi::Tensor;
using tvm::ffi::TensorView;

namespace {

bool is_col_major_matrix_view(TensorView tensor) {
  int ndim = tensor.ndim();
  if (ndim < 2) {
    return false;
  }

  int row_dim = ndim - 2;
  int col_dim = ndim - 1;

  if (tensor.stride(row_dim) != 1) {
    return false;
  }
  if (tensor.stride(col_dim) != tensor.size(row_dim)) {
    return false;
  }

  int64_t expected_stride = tensor.size(row_dim) * tensor.size(col_dim);
  for (int dim = row_dim - 1; dim >= 0; --dim) {
    if (tensor.stride(dim) != expected_stride) {
      return false;
    }
    expected_stride *= tensor.size(dim);
  }
  return true;
}

CutlassGemmConfig getBf16GemmConfig(int64_t m, int64_t n, int64_t k, int64_t tactic) {
  auto getCutlassBf16GemmConfigs = []() {
    CutlassBf16GemmRunner<__nv_bfloat16> gemmRunner;
    return gemmRunner.getConfigs();
  };
  static std::vector<CutlassGemmConfig> globalConfigs = getCutlassBf16GemmConfigs();
  TVM_FFI_ICHECK(tactic >= 0 && tactic < static_cast<int64_t>(globalConfigs.size()))
      << "tactic must be between 0 and " << globalConfigs.size();
  return globalConfigs[tactic];
}

template <typename T>
void runGemm(TensorView out, TensorView mat1, TensorView mat2, int64_t m, int64_t n, int64_t k,
             int64_t b, CutlassGemmConfig const& gemmConfig, TensorView workspace_buffer) {
  CutlassBf16GemmRunner<T> gemmRunner;

  int64_t const required_workspace_size = gemmRunner.getWorkspaceSize(m, n, k);
  int64_t const provided_workspace_size =
      workspace_buffer.numel() * get_element_size(workspace_buffer);

  auto runKernel = [&](void* workspace) {
    gemmRunner.gemm(static_cast<__nv_bfloat16*>(mat1.data_ptr()),
                    static_cast<__nv_bfloat16*>(mat2.data_ptr()), out.data_ptr(), m, n, k, b,
                    gemmConfig, reinterpret_cast<char*>(workspace), required_workspace_size,
                    get_stream(mat1.device()));
  };

  if (provided_workspace_size < required_workspace_size) {
    Tensor new_workspace =
        alloc_tensor({required_workspace_size}, DLDataType{kDLInt, 8, 1}, mat1.device());
    runKernel(new_workspace.data_ptr());
  } else {
    runKernel(workspace_buffer.data_ptr());
  }
}

void bf16_bmm_impl(TensorView mat1, TensorView mat2, TensorView out, TensorView workspace_buffer,
                   int64_t tactic) {
  CHECK_INPUT_AND_TYPE(mat1, dl_bfloat16);
  CHECK_CUDA(mat2);
  CHECK_INPUT_TYPE(mat2, dl_bfloat16);
  TVM_FFI_ICHECK(is_col_major_matrix_view(mat2))
      << "mat2 last two dimensions must be laid out in column-major order; pass b.transpose(-2, -1).";

  int64_t m, n, k, b;
  if (mat1.ndim() == 2) {
    TVM_FFI_ICHECK_EQ(mat2.ndim(), 2) << "mat2 must be a matrix";
    TVM_FFI_ICHECK_EQ(mat1.size(1), mat2.size(1))
        << "mat1 and mat2 shapes cannot be multiplied (" << mat1.size(0) << "x" << mat1.size(1)
        << " and " << mat2.size(0) << "x" << mat2.size(1) << ")";
    m = mat1.size(0);
    n = mat2.size(0);
    k = mat2.size(1);
    b = 1;
  } else if (mat1.ndim() == 3) {
    TVM_FFI_ICHECK_EQ(mat2.ndim(), 3) << "mat2 must be a batch of matrices";
    TVM_FFI_ICHECK_EQ(mat1.size(0), mat2.size(0)) << "mat1 and mat2 must have the same batch size ("
                                                  << mat1.size(0) << " and " << mat2.size(0) << ")";
    TVM_FFI_ICHECK_EQ(mat1.size(2), mat2.size(2))
        << "mat1 and mat2 shapes cannot be multiplied (" << mat1.size(1) << "x" << mat1.size(2)
        << " and " << mat2.size(1) << "x" << mat2.size(2) << ")";
    m = mat1.size(1);
    n = mat2.size(1);
    k = mat2.size(2);
    b = mat1.size(0);
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError) << "mat1 must be a matrix or a batch of matrices";
  }

  if (tactic == -1) {
    tactic = 0;
  }
  auto config = getBf16GemmConfig(m, n, k, tactic);

  std::vector<int64_t> out_shape =
      mat1.ndim() == 2 ? std::vector<int64_t>{m, n} : std::vector<int64_t>{b, m, n};
  TVM_FFI_ICHECK_EQ(out.ndim(), static_cast<int>(out_shape.size()))
      << "out must have " << out_shape.size() << " dimensions, but got " << out.ndim();
  for (int i = 0; i < static_cast<int>(out_shape.size()); ++i) {
    TVM_FFI_ICHECK_EQ(out.size(i), out_shape[i])
        << "out shape mismatch at dimension " << i << ": expected " << out_shape[i] << ", got "
        << out.size(i);
  }

  switch (encode_dlpack_dtype(out.dtype())) {
    case float16_code:
      runGemm<half>(out, mat1, mat2, m, n, k, b, config, workspace_buffer);
      break;
    case bfloat16_code:
      runGemm<__nv_bfloat16>(out, mat1, mat2, m, n, k, b, config, workspace_buffer);
      break;
    default:
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "out_dtype must be one of fp16/bf16.";
  }
}

}  // namespace

void bf16_gemm(TensorView mat1, TensorView mat2, TensorView out, TensorView workspace_buffer,
               int64_t tactic) {
  bf16_bmm_impl(mat1, mat2, out, workspace_buffer, tactic);
}

int64_t bf16_gemm_tactic_num() {
  auto getCutlassConfigs = []() {
    CutlassBf16GemmRunner<__nv_bfloat16> gemmRunner;
    return gemmRunner.getConfigs();
  };
  static int64_t totalTactics = getCutlassConfigs().size();
  return totalTactics;
}

}  // namespace torch_ext

TVM_FFI_DLL_EXPORT_TYPED_FUNC(bf16_gemm, torch_ext::bf16_gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bf16_gemm_tactic_num, torch_ext::bf16_gemm_tactic_num);

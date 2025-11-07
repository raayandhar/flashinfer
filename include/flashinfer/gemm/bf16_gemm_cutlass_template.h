/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef FLASHINFER_BF16_GEMM_CUTLASS_TEMPLATE_H_
#define FLASHINFER_BF16_GEMM_CUTLASS_TEMPLATE_H_

#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
#include "cutlass/arch/arch.h"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/layout/matrix.h"
#include "flashinfer/cutlass_utils.cuh"

#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif  // __GNUC__

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <tuple>

namespace flashinfer {
namespace gemm {

// BF16 TN gemm for SM80
// batch GEMM
// return required workspace size
template <typename T, typename arch, int32_t CTA_M_, int32_t CTA_N_, int32_t CTA_K_,
          int32_t WARP_M_, int32_t WARP_N_, int32_t WARP_K_, int32_t NUM_STAGES_>
size_t genericBf16GemmKernelLauncherSm80(__nv_bfloat16 const* A, __nv_bfloat16 const* B, T* D,
                                         int m, int n, int k, int b, CutlassGemmConfig config,
                                         char* workspacePtr, size_t const workspaceBytes,
                                         cudaStream_t stream) {

  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;

  // Convert standard types to CUTLASS types
  using ElementOutput = std::conditional_t<
      std::is_same_v<T, __nv_bfloat16>,
      cutlass::bfloat16_t,
      std::conditional_t<std::is_same_v<T, half>, cutlass::half_t, T>
  >;

  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ThreadblockShape = cutlass::gemm::GemmShape<CTA_M_, CTA_N_, CTA_K_>;
  using WarpShape = cutlass::gemm::GemmShape<WARP_M_, WARP_N_, WARP_K_>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, AlignmentC, ElementAccumulator, ElementAccumulator>;

  // DefaultGemm (non-grouped) doesn't support ComplexTransform in this CUTLASS version
  // Only DefaultGemmGrouped does - so we use alignment-only signature
  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
      ElementA,                              // Element A
      LayoutA,                               // Layout A
      AlignmentA,                            // Alignment A (no ComplexTransform!)
      ElementB,                              // Element B
      LayoutB,                               // Layout B
      AlignmentB,                            // Alignment B (no ComplexTransform!)
      ElementOutput,                         // Element C&D
      LayoutC,                               // Layout C&D
      ElementAccumulator,                    // Accumulator
      OperatorClass,                         // Operator Class
      arch,                                  // Architecture
      ThreadblockShape,                      // Threadblock shape
      WarpShape,                             // Warp shape
      InstructionShape,                      // Instruction shape
      EpilogueOp,                            // Epilogue
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,  // Swizzle
      NUM_STAGES_                            // Stages
      >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversal<GemmKernel>;

  typename Gemm::Arguments arguments;

  if (b == 1) {
    // Single GEMM
    arguments = typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k},
        /*batch_count=*/1,
        {ElementAccumulator(1.0), ElementAccumulator(0.0)},
        reinterpret_cast<ElementA const*>(A), reinterpret_cast<ElementB const*>(B),
        reinterpret_cast<ElementOutput const*>(nullptr), reinterpret_cast<ElementOutput*>(D),
        /*batch_stride_A=*/0, /*batch_stride_B=*/0,
        /*batch_stride_C=*/0, /*batch_stride_D=*/0, k, k, 0, n);
  } else {
    // Batched GEMM
    int64_t batch_stride_A = m * k;
    int64_t batch_stride_B = n * k;
    int64_t batch_stride_D = m * n;

    arguments = typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kBatched, {m, n, k}, b,
        {ElementAccumulator(1.0), ElementAccumulator(0.0)},
        reinterpret_cast<ElementA const*>(A), reinterpret_cast<ElementB const*>(B),
        reinterpret_cast<ElementOutput const*>(nullptr), reinterpret_cast<ElementOutput*>(D),
        batch_stride_A, batch_stride_B, 0, batch_stride_D, k, k, 0, n);
  }

  Gemm gemm_op;

  // Query workspace size
  size_t workspace_size = gemm_op.get_workspace_size(arguments);

  if (A == nullptr || B == nullptr || D == nullptr) {
    return workspace_size;
  }

  if (workspace_size > workspaceBytes && workspacePtr != nullptr) {
    throw std::runtime_error(
        "Insufficient workspace for BF16 GEMM. Required: " + std::to_string(workspace_size) +
        " bytes, provided: " + std::to_string(workspaceBytes) + " bytes.");
  }

  cutlass::Status status = gemm_op.initialize(arguments, workspacePtr, stream);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("Failed to initialize CUTLASS BF16 GEMM: " +
                             std::string(cutlassGetStatusString(status)));
  }

  status = gemm_op.run(stream);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("Failed to run CUTLASS BF16 GEMM: " +
                             std::string(cutlassGetStatusString(status)));
  }

  return workspace_size;
}

template <typename T, typename arch>
size_t dispatchGemmConfigSm80(__nv_bfloat16 const* A, __nv_bfloat16 const* B, T* D, int m, int n,
                              int k, int b, CutlassGemmConfig gemmConfig, char* workspacePtr,
                              size_t const workspaceBytes, cudaStream_t stream) {
  int num_stages = (gemmConfig.stages == -1) ? 2 : gemmConfig.stages;

  switch (gemmConfig.tile_config_sm80) {
    case CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x64:
      if (num_stages == 2) {
        return genericBf16GemmKernelLauncherSm80<T, arch, 128, 128, 64, 64, 64, 64, 2>(
            A, B, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      } else if (num_stages == 3) {
        return genericBf16GemmKernelLauncherSm80<T, arch, 128, 128, 64, 64, 64, 64, 3>(
            A, B, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      } else if (num_stages == 4) {
        return genericBf16GemmKernelLauncherSm80<T, arch, 128, 128, 64, 64, 64, 64, 4>(
            A, B, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      }
      break;

    case CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64:
      if (num_stages == 2) {
        return genericBf16GemmKernelLauncherSm80<T, arch, 128, 128, 64, 64, 32, 64, 2>(
            A, B, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      } else if (num_stages == 3) {
        return genericBf16GemmKernelLauncherSm80<T, arch, 128, 128, 64, 64, 32, 64, 3>(
            A, B, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      } else if (num_stages == 4) {
        return genericBf16GemmKernelLauncherSm80<T, arch, 128, 128, 64, 64, 32, 64, 4>(
            A, B, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      }
      break;

    case CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64:
      if (num_stages == 2) {
        return genericBf16GemmKernelLauncherSm80<T, arch, 128, 64, 64, 64, 32, 64, 2>(
            A, B, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      } else if (num_stages == 3) {
        return genericBf16GemmKernelLauncherSm80<T, arch, 128, 64, 64, 64, 32, 64, 3>(
            A, B, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      } else if (num_stages == 4) {
        return genericBf16GemmKernelLauncherSm80<T, arch, 128, 64, 64, 64, 32, 64, 4>(
            A, B, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      }
      break;

    case CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
      if (num_stages == 2) {
        return genericBf16GemmKernelLauncherSm80<T, arch, 64, 128, 64, 64, 32, 64, 2>(
            A, B, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      } else if (num_stages == 3) {
        return genericBf16GemmKernelLauncherSm80<T, arch, 64, 128, 64, 64, 32, 64, 3>(
            A, B, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      } else if (num_stages == 4) {
        return genericBf16GemmKernelLauncherSm80<T, arch, 64, 128, 64, 64, 32, 64, 4>(
            A, B, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      }
      break;

    case CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64:
      if (num_stages == 2) {
        return genericBf16GemmKernelLauncherSm80<T, arch, 128, 256, 64, 64, 64, 64, 2>(
            A, B, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      } else if (num_stages == 3) {
        return genericBf16GemmKernelLauncherSm80<T, arch, 128, 256, 64, 64, 64, 64, 3>(
            A, B, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      }
      break;

    case CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
      if (num_stages == 2) {
        return genericBf16GemmKernelLauncherSm80<T, arch, 64, 128, 64, 32, 64, 64, 2>(
            A, B, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      } else if (num_stages == 3) {
        return genericBf16GemmKernelLauncherSm80<T, arch, 64, 128, 64, 32, 64, 64, 3>(
            A, B, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      } else if (num_stages == 4) {
        return genericBf16GemmKernelLauncherSm80<T, arch, 64, 128, 64, 32, 64, 64, 4>(
            A, B, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
      }
      break;

    default:
      throw std::runtime_error("Unsupported tile config for BF16 GEMM on SM80");
  }

  throw std::runtime_error("Unsupported number of stages for BF16 GEMM on SM80");
}

template <typename T>
size_t dispatchToArch(__nv_bfloat16 const* A, __nv_bfloat16 const* B, void* D, int m, int n,
                      int k, int b, CutlassGemmConfig gemmConfig, char* workspacePtr,
                      size_t const workspaceBytes, cudaStream_t stream) {
  using arch = cutlass::arch::Sm80;

  // Swap M, N and A, B for TN layout
  // A row-major, B column-major, C, D row-major
  return dispatchGemmConfigSm80<T, arch>(B, A, static_cast<T*>(D), n, m, k, b, gemmConfig,
                                         workspacePtr, workspaceBytes, stream);
}

template <typename T>
void CutlassBf16GemmRunner<T>::gemm(__nv_bfloat16 const* A, __nv_bfloat16 const* B, void* D, int m,
                                    int n, int k, int b, CutlassGemmConfig gemmConfig,
                                    char* workspacePtr, size_t const workspaceBytes,
                                    cudaStream_t stream) {
  dispatchToArch<T>(A, B, D, m, n, k, b, gemmConfig, workspacePtr, workspaceBytes, stream);
}

template <typename T>
size_t CutlassBf16GemmRunner<T>::getWorkspaceSizeImpl(int m, int n, int k) {
  size_t workspace_size = 0;
  auto gemmConfigs = CutlassBf16GemmRunner<T>{}.getConfigs();
  for (auto const& gemmConfig : gemmConfigs) {
    try {
      size_t curr_workspace_size =
          dispatchToArch<T>(nullptr, nullptr, nullptr, m, n, k, 1, gemmConfig, nullptr, 0, nullptr);

      workspace_size = std::max(workspace_size, curr_workspace_size);
    } catch (std::runtime_error& e) {
      // Swallow errors when SMEM exceeds maximum allowed
      continue;
    }
  }

  return workspace_size;
}

template <typename T>
size_t CutlassBf16GemmRunner<T>::getWorkspaceSize(int m, int n, int k) {
  // Custom hash function for the MNK type
  using MNK = std::tuple<int, int, int>;

  struct MNKHash {
    size_t operator()(const MNK& mnk) const {
      auto h1 = std::hash<int>{}(std::get<0>(mnk));
      auto h2 = std::hash<int>{}(std::get<1>(mnk));
      auto h3 = std::hash<int>{}(std::get<2>(mnk));
      return h1 ^ h2 ^ h3;
    }
  };

  static std::unordered_map<MNK, size_t, MNKHash> workspace_hashmap;

  size_t workspace_size = 0;
  if (workspace_hashmap.find(std::make_tuple(m, n, k)) == workspace_hashmap.end()) {
    workspace_size = CutlassBf16GemmRunner<T>::getWorkspaceSizeImpl(m, n, k);
    workspace_hashmap[std::make_tuple(m, n, k)] = workspace_size;
  } else {
    workspace_size = workspace_hashmap[std::make_tuple(m, n, k)];
  }
  return workspace_size;
}

template <typename T>
std::vector<CutlassGemmConfig> CutlassBf16GemmRunner<T>::getConfigs() const {
  std::vector<CutlassGemmConfig> candidate_configs;

  std::vector<CutlassTileConfig> tilesSm80 = {
      CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x64,
      CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64,
      CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64,
      CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64,
      CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64,
      CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64,
  };

  std::vector<int> stages = {2, 3, 4};

  for (auto const& tile_config : tilesSm80) {
    for (auto const& stage : stages) {
      CutlassGemmConfig config(tile_config, SplitKStyle::NO_SPLIT_K, 1, stage);
      candidate_configs.push_back(config);
    }
  }

  return candidate_configs;
}

}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_BF16_GEMM_CUTLASS_TEMPLATE_H_

/***************************************************************************************************
 * Copyright (c) 2025 Intel Corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

#include <torch/all.h>
#include "csrc/utils.h"

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"

#include "gemm_xe2_policy.hpp"
#include "grouped_gemm_fc1_silu_xe2.hpp"

namespace MoE {
using namespace cute;

template <typename, typename, typename, char, char, class>
class GemmFc1SiLuCuteName;

template <
    char layoutA,
    char layoutB,
    class policy,
    typename ElementA,
    typename ElementB,
    typename ElementBI,
    typename ElementD>
void MoEFc1SiLuLauncher(
    sycl::queue& stream,
    const ElementA* activations,
    const ElementB* weights,
    const ElementBI* bias,
    ElementD* outputs,
    const int inter_size,
    const int gemm_k,
    const int* rows_per_expert,
  const int* active_expert_ids,
  const int* active_row_offsets,
  const int* active_expert_count,
    const int num_experts,
    float clamp_limit,
    int32_t* atomic_buffer) {
  using ElementA_non_CV = cutlass::platform::remove_cv_t<ElementA>;
  auto op = XE_DPAS_TT<8, float, ElementA_non_CV>{};

  using WGTile = typename policy::WGTile;
  using SGLayout = typename policy::SGLayout;
  using MMA = typename TiledMMAHelper<
      MMA_Atom<decltype(op)>,
      Layout<WGTile>,
      SGLayout>::TiledMMA;
  auto mma = MMA{};

  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto max_threads_per_workgroup = size(mma);
  static constexpr int MaxThreadsPerSM = 512;

  TORCH_CHECK(
      MaxThreadsPerSM % max_threads_per_workgroup == 0,
      "MaxThreadsPerSM must be divisible by MaxThreadsPerWorkgroup");

  sycl::range<3> local(1, 1, max_threads_per_workgroup);
  sycl::range<3> global(1, sm_count * MaxThreadsPerSM / max_threads_per_workgroup, 1);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{syclex::sub_group_size<16>, intelex::grf_size<256>};

  using GmemTiledCopyA = typename policy::GmemTiledCopyA;
  using GmemTiledCopyB = typename policy::GmemTiledCopyB;
  using GmemTiledCopyD = typename policy::GmemTiledCopyD;

  stream.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), cgh);
    cgh.parallel_for<GemmFc1SiLuCuteName<
        ElementA,
        ElementB,
        ElementD,
        layoutA,
        layoutB,
        policy>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
          MoE::MoEFc1SiLU<
              GmemTiledCopyA,
              GmemTiledCopyB,
              GmemTiledCopyD,
              layoutA,
              layoutB,
              'R'>(
              activations,
              weights,
              bias,
              outputs,
              mma,
              rows_per_expert,
              active_expert_ids,
              active_row_offsets,
              active_expert_count,
              num_experts,
              inter_size,
              gemm_k,
              clamp_limit,
              atomic_buffer,
              local_mem);
        });
  });
}

inline at::Tensor cutlass_grouped_gemm_fc1_silu_xe2_impl(
    at::Tensor& ptr_A,
    at::Tensor& ptr_B,
    const c10::optional<at::Tensor>& ptr_bias,
    at::Tensor& ptr_D,
    at::Tensor& rows_per_expert,
    int64_t inter_size,
    int64_t K,
    int64_t num_experts,
  const c10::optional<at::Tensor>& active_expert_ids,
  const c10::optional<at::Tensor>& active_row_offsets,
  const c10::optional<at::Tensor>& active_expert_count,
    double clamp_limit) {
  auto& dpcpp_queue = at::xpu::getCurrentXPUStream(ptr_A.device().index()).queue();

  TORCH_CHECK(ptr_A.dtype() == at::kBFloat16, "BF16 activations are required");
  TORCH_CHECK(ptr_B.dtype() == at::kBFloat16, "BF16 weights are required");
  TORCH_CHECK(ptr_D.dtype() == at::kBFloat16, "BF16 outputs are required");

  TORCH_CHECK(ptr_A.dim() == 2, "ptr_A must be 2D [Total_M, K]");
  TORCH_CHECK(ptr_B.dim() == 3, "ptr_B must be 3D [num_experts, K, 2*inter_size]");
  TORCH_CHECK(ptr_D.dim() == 2, "ptr_D must be 2D [Total_M, inter_size]");
  TORCH_CHECK(rows_per_expert.dim() == 1, "rows_per_expert must be 1D");

  TORCH_CHECK(ptr_A.is_contiguous(), "ptr_A must be contiguous");
  TORCH_CHECK(ptr_B.is_contiguous(), "ptr_B must be contiguous");
  TORCH_CHECK(ptr_D.is_contiguous(), "ptr_D must be contiguous");
  TORCH_CHECK(rows_per_expert.is_contiguous(), "rows_per_expert must be contiguous");

  TORCH_CHECK(ptr_A.size(1) == K, "ptr_A.size(1) must match K");
  TORCH_CHECK(ptr_B.size(0) == num_experts, "ptr_B.size(0) must match num_experts");
  TORCH_CHECK(ptr_B.size(1) == K, "ptr_B.size(1) must match K");
  TORCH_CHECK(ptr_B.size(2) == inter_size * 2, "ptr_B.size(2) must match 2 * inter_size");
  TORCH_CHECK(ptr_D.size(0) == ptr_A.size(0), "ptr_D.size(0) must match ptr_A.size(0)");
  TORCH_CHECK(ptr_D.size(1) == inter_size, "ptr_D.size(1) must match inter_size");

  if (ptr_bias.has_value()) {
    TORCH_CHECK(ptr_bias->dtype() == at::kBFloat16, "ptr_bias must be BF16");
    TORCH_CHECK(ptr_bias->is_contiguous(), "ptr_bias must be contiguous");
    TORCH_CHECK(ptr_bias->dim() == 2, "ptr_bias must be 2D [num_experts, 2*inter_size]");
    TORCH_CHECK(ptr_bias->size(0) == num_experts, "ptr_bias.size(0) must match num_experts");
    TORCH_CHECK(ptr_bias->size(1) == inter_size * 2, "ptr_bias.size(1) must match 2 * inter_size");
  }

  TORCH_CHECK(
      active_expert_ids.has_value() == active_row_offsets.has_value() &&
          active_row_offsets.has_value() == active_expert_count.has_value(),
      "active expert metadata tensors must be provided together");
  if (active_expert_ids.has_value()) {
    TORCH_CHECK(active_expert_ids->dtype() == at::kInt, "active_expert_ids must be int32");
    TORCH_CHECK(active_expert_ids->dim() == 1, "active_expert_ids must be 1D");
    TORCH_CHECK(active_expert_ids->size(0) == num_experts, "active_expert_ids must be [num_experts]");
    TORCH_CHECK(active_expert_ids->is_contiguous(), "active_expert_ids must be contiguous");
    TORCH_CHECK(active_row_offsets->dtype() == at::kInt, "active_row_offsets must be int32");
    TORCH_CHECK(active_row_offsets->dim() == 1, "active_row_offsets must be 1D");
    TORCH_CHECK(active_row_offsets->size(0) == num_experts, "active_row_offsets must be [num_experts]");
    TORCH_CHECK(active_row_offsets->is_contiguous(), "active_row_offsets must be contiguous");
    TORCH_CHECK(active_expert_count->dtype() == at::kInt, "active_expert_count must be int32");
    TORCH_CHECK(active_expert_count->numel() == 1, "active_expert_count must contain 1 element");
  }

  if (ptr_A.size(0) == 0) {
    return ptr_D;
  }
  int avg_m = ptr_A.size(0) / std::max<int64_t>(num_experts, 1);
  at::Tensor atomic_buffer = at::empty({static_cast<long>(1)}, ptr_A.options().dtype(at::kInt));

#define FC1_SILU_LAUNCHER(policy)                                             \
  do {                                                                        \
    using scalar_t = bfloat16_t;                                              \
    MoEFc1SiLuLauncher<'R', 'R', policy>(                                     \
        dpcpp_queue,                                                          \
        reinterpret_cast<scalar_t*>(ptr_A.data_ptr()),                        \
        reinterpret_cast<scalar_t*>(ptr_B.data_ptr()),                        \
        ptr_bias.has_value() ? reinterpret_cast<scalar_t*>(ptr_bias->data_ptr()) \
                             : static_cast<scalar_t*>(nullptr),               \
        reinterpret_cast<scalar_t*>(ptr_D.data_ptr()),                        \
        static_cast<int>(inter_size),                                         \
        static_cast<int>(K),                                                  \
        reinterpret_cast<int*>(rows_per_expert.data_ptr()),                   \
        active_expert_ids.has_value()                                         \
          ? reinterpret_cast<int*>(active_expert_ids->data_ptr())           \
          : nullptr,                                                        \
        active_row_offsets.has_value()                                        \
          ? reinterpret_cast<int*>(active_row_offsets->data_ptr())          \
          : nullptr,                                                        \
        active_expert_count.has_value()                                       \
          ? reinterpret_cast<int*>(active_expert_count->data_ptr())         \
          : nullptr,                                                        \
        static_cast<int>(num_experts),                                        \
        static_cast<float>(clamp_limit),                                      \
        static_cast<int*>(atomic_buffer.data_ptr()));                         \
  } while (false)

  if (avg_m <= 8) {
    using policy = w16a16_policy_m_8;
    FC1_SILU_LAUNCHER(policy);
  } else if (avg_m <= 16) {
    using policy = w16a16_policy_m_16;
    FC1_SILU_LAUNCHER(policy);
  } else {
    using policy = w16a16_policy_m_32;
    FC1_SILU_LAUNCHER(policy);
  }

#undef FC1_SILU_LAUNCHER
  return ptr_D;
}

}  // namespace MoE
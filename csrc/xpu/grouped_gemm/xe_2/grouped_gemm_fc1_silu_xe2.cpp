#include <torch/all.h>

#include "grouped_gemm_fc1_silu_xe2.h"
#include "grouped_gemm_fc1_silu_xe2_interface.hpp"

torch::Tensor cutlass_grouped_gemm_fc1_silu_xe2(
    torch::Tensor ptr_A,
    torch::Tensor ptr_B,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor rows_per_expert,
    int64_t inter_size,
    int64_t K,
    int64_t num_experts,
    const c10::optional<at::Tensor>& active_expert_ids,
    const c10::optional<at::Tensor>& active_row_offsets,
    const c10::optional<at::Tensor>& active_expert_count,
    double clamp_limit) {
  return MoE::cutlass_grouped_gemm_fc1_silu_xe2_impl(
      ptr_A,
      ptr_B,
      ptr_bias,
      ptr_D,
      rows_per_expert,
      inter_size,
      K,
      num_experts,
      active_expert_ids,
      active_row_offsets,
      active_expert_count,
      clamp_limit);
}
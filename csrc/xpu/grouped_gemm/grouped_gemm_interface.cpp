#include "csrc/utils.h"
#include "grouped_gemm_interface.h"
#include <stdio.h>

#ifdef VLLM_XPU_ENABLE_XE2
  #include "xe_2/grouped_gemm_xe2.h"
  #include "xe_2/grouped_gemm_fc1_silu_xe2.h"
#endif
#ifdef VLLM_XPU_ENABLE_XE_DEFAULT
  #include "xe_default/grouped_gemm_xe_default.h"
#endif

torch::Tensor cutlass_grouped_gemm_interface(
    torch::Tensor ptr_A,
    torch::Tensor ptr_B,
    const c10::optional<at::Tensor>& ptr_scales,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    bool is_B_int4,
    bool is_B_mxfp4) {
  if (vllm::xpu::force_xe_default_kernel()) {
#ifdef VLLM_XPU_ENABLE_XE_DEFAULT
    int64_t groups = num_experts;
    return cutlass_grouped_gemm_xe_default(
        ptr_A, ptr_B, ptr_bias, ptr_D, rows_per_expert, N, K, groups);
#else
    TORCH_CHECK(
        false,
        "XE default cutlass kernel is not enabled in this build, force use XE "
        "default kernel failed.");
#endif
  } else if (vllm::xpu::is_xe2_arch()) {
#ifdef VLLM_XPU_ENABLE_XE2
    // Use XE2 cutlass kernel
    return cutlass_grouped_gemm_xe2(
        ptr_A,
        ptr_B,
        ptr_scales,
        ptr_bias,
        ptr_D,
        rows_per_expert,
        N,
        K,
        num_experts,
        is_B_int4,
        is_B_mxfp4);
#else
    TORCH_CHECK(false, "XE2 cutlass kernel is not enabled in this build.");
#endif
  } else {
#ifdef VLLM_XPU_ENABLE_XE_DEFAULT
    int64_t groups = num_experts;
    return cutlass_grouped_gemm_xe_default(
        ptr_A, ptr_B, ptr_bias, ptr_D, rows_per_expert, N, K, groups);
#else
    TORCH_CHECK(
        false, "XE default cutlass kernel is not enabled in this build.");
#endif
  }
}

torch::Tensor cutlass_moe_fc1_silu_interface(
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
  if (vllm::xpu::is_xe2_arch()) {
#ifdef VLLM_XPU_ENABLE_XE2
    return cutlass_grouped_gemm_fc1_silu_xe2(
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
#else
    TORCH_CHECK(false, "XE2 fused FC1+SiLU kernel is not enabled in this build.");
#endif
  }

  TORCH_CHECK(
      false,
      "cutlass_moe_fc1_silu_interface currently supports XE2 only.");
}

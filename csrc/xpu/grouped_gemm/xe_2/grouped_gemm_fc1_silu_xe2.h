#include <torch/all.h>

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
    double clamp_limit);
/***************************************************************************************************
 * Copyright 2025 Intel corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/platform/platform.h"
#include <cute/util/compat.hpp>
#include <sycl/sycl.hpp>

#include "grouped_gemm_xe2.hpp"

namespace MoE {
using namespace cute;

template <typename T>
CUTE_DEVICE auto make_gateup_tensor(T* ptr, int inter_size, int gemm_k, int full_n) {
  auto shape = make_shape(inter_size, gemm_k);
  return make_tensor(
      make_gmem_ptr(ptr), make_layout(shape, make_stride(_1{}, full_n)));
}

template <typename Acc>
CUTE_DEVICE Acc silu_mul_acc(Acc gate, Acc up, float clamp_limit) {
  float gate_f = static_cast<float>(gate);
  float up_f = static_cast<float>(up);
  if (clamp_limit > 0.0f) {
    gate_f = sycl::fmin(gate_f, clamp_limit);
    up_f = sycl::fmax(-clamp_limit, sycl::fmin(up_f, clamp_limit));
  }
  float silu_f = gate_f / (1.0f + sycl::exp(-gate_f));
  return static_cast<Acc>(silu_f * up_f);
}

template <
    class GmemTiledCopyA,
    class GmemTiledCopyB,
    class GmemTiledCopyD,
    class ATensor,
    class BGateTensor,
    class BUpTensor,
    class DTensor,
    class TiledMMA,
    typename ElementBI>
CUTE_DEVICE void xe_gemm_fc1_silu(
    ATensor const& A,
    BGateTensor const& B_gate,
    BUpTensor const& B_up,
    const ElementBI* Bias,
    DTensor& D,
    Coord<int, int, cute::Underscore, int> blk_coord,
    TiledMMA const& mma,
    int full_n,
    float clamp_limit) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_m = get<0>(blk_coord);
  auto wg_n = get<1>(blk_coord);
  int local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cBg = make_identity_tensor(B_gate.shape());
  Tensor cBu = make_identity_tensor(B_up.shape());
  Tensor cD = make_identity_tensor(D.shape());

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
  Tensor gBg = local_tile(cBg, select<1, 2>(wg_tile), make_coord(wg_n, _));
  Tensor gBu = local_tile(cBu, select<1, 2>(wg_tile), make_coord(wg_n, _));
  Tensor gD = local_tile(cD, wg_tile, wg_coord, Step<_1, _1, X>{});

  auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A);
  auto copy_bg = get_block_2d_copy_B<GmemTiledCopyB>(mma, B_gate);
  auto copy_bu = get_block_2d_copy_B<GmemTiledCopyB>(mma, B_up);
  auto copy_d = get_block_2d_copy_D<GmemTiledCopyD>(mma, D);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_bg = copy_bg.get_slice(local_id);
  auto thr_copy_bu = copy_bu.get_slice(local_id);
  auto thr_copy_d = copy_d.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrBg = thr_mma.partition_sg_fragment_B(gBg(_, _, 0));
  auto tCrBu = thr_mma.partition_sg_fragment_B(gBu(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrBg = thr_copy_bg.partition_sg_fragment_D(gBg(_, _, 0));
  auto tBrBu = thr_copy_bu.partition_sg_fragment_D(gBu(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgBg = thr_copy_bg.partition_S(gBg);
  Tensor tBgBu = thr_copy_bu.partition_S(gBu);

  auto tCrGate = thr_mma.partition_sg_fragment_C(gD);
  auto tCrUp = thr_mma.partition_sg_fragment_C(gD);
  auto tCrOut = thr_copy_d.partition_sg_fragment_S(gD);
  auto tCgD = thr_copy_d.partition_D(gD);

  constexpr int barrier_scope = 2;
  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));

  clear(tCrGate);
  clear(tCrUp);

  for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_bg, tBgBg(_, _, _, k_tile), tBrBg);
    copy(copy_bu, tBgBu(_, _, _, k_tile), tBrBu);

    reorder(tArA, tCrA);
    reorder(tBrBg, tCrBg);
    reorder(tBrBu, tCrBu);

    cute::gemm(mma, tCrA, tCrBg, tCrGate);
    cute::gemm(mma, tCrA, tCrBu, tCrUp);

    barrier_wait(barrier_scope);
  }

  static constexpr auto ATOM_M =
      get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N =
      get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto tile_m = get<0>(wg_tile);
  static constexpr auto tile_n = get<1>(wg_tile);
  static constexpr auto SG_M = tile_m / ATOM_M;
  static constexpr auto SG_N = tile_n / ATOM_N;

  auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;
  int sg_local_id = cutlass::get_sub_group_local_id();
  int n_tile_start = wg_n * tile_n;
  int n_sg_start = sg_local_n_coord * SG_N;
  static constexpr int sg_local_range = 16;
  int inter_size = full_n / 2;

  if (Bias != nullptr) {
    CUTLASS_PRAGMA_UNROLL
    for (int sn = 0; sn < SG_N / sg_local_range; ++sn) {
      int sg_local_n = sn * sg_local_range + sg_local_id;
      float gate_bias = static_cast<float>(Bias[n_tile_start + n_sg_start + sg_local_n]);
      float up_bias = static_cast<float>(Bias[inter_size + n_tile_start + n_sg_start + sg_local_n]);
      CUTLASS_PRAGMA_UNROLL
      for (int sm = 0; sm < SG_M; ++sm) {
        tCrGate(sn * SG_M + sm) += gate_bias;
        tCrUp(sn * SG_M + sm) += up_bias;
      }
    }
  }

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < tCrGate.size(); ++i) {
    tCrGate(i) = silu_mul_acc(tCrGate(i), tCrUp(i), clamp_limit);
  }

  reorder(tCrGate, tCrOut);
  copy(copy_d, tCrOut, tCgD);
}

template <
    class GmemTiledCopyA,
    class GmemTiledCopyB,
    class GmemTiledCopyD,
    char LayoutKindA,
    char LayoutKindB,
    char LayoutKindD,
    class TiledMMA,
    typename ElementA,
    typename ElementB,
    typename ElementBI,
    typename ElementD>
CUTE_DEVICE void MoEFc1SiLU(
    const ElementA* Activations,
    const ElementB* Weights,
    const ElementBI* Bias,
    ElementD* Outputs,
    TiledMMA const& mma,
    const int* rows_per_expert,
  const int* active_expert_ids,
  const int* active_row_offsets,
  const int* active_expert_count,
    const int32_t num_experts,
    const int32_t inter_size,
    const int32_t gemm_k,
    float clamp_limit,
    int32_t* atomic_buffer,
    const sycl::local_accessor<int32_t, 1>& slm_mem_const) {
  static_assert(LayoutKindB == 'R', "FC1 fused kernel expects row-major stored weights.");

  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_tile = mma.tile_mnk();
  auto wg_tile_m = get<0>(wg_tile);
  auto wg_tile_n = get<1>(wg_tile);

  int group_id = item.get_group_linear_id();
  int gemm_n_pad = (inter_size + wg_tile_n - 1) / wg_tile_n * wg_tile_n;
  int group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
  int group_range = item.get_group_range(1);
  int local_id = item.get_local_linear_id();

  if (group_id == 0 && local_id == 0) {
    auto atm = sycl::atomic_ref<
        int,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>(atomic_buffer[0]);
    atm.store(0);
  }

  int pre_tiles = 0;
  int32_t* slm_mem = static_cast<int32_t*>(
      slm_mem_const.template get_multi_ptr<sycl::access::decorated::no>().get());

  int full_n = inter_size * 2;

  int expert_loop_count = num_experts;
  if (active_expert_count != nullptr) {
    expert_loop_count = active_expert_count[0];
  }

  for (int i = 0; i < expert_loop_count; ++i) {
    int expert_id = active_expert_ids != nullptr ? active_expert_ids[i] : i;
    int pre_rows = active_row_offsets != nullptr ? active_row_offsets[i] : 0;
    int gemm_m = rows_per_expert[expert_id];
    if (gemm_m <= 0) {
      continue;
    }
    int cumsum_rows_for_experts = pre_rows + gemm_m;
    int cumsum_tiles_for_experts =
        (gemm_m + wg_tile_m - 1) / wg_tile_m + pre_tiles;

    if (group_m_id >= cumsum_tiles_for_experts) {
      pre_tiles = cumsum_tiles_for_experts;
      continue;
    }

    int64_t expert_offset = static_cast<int64_t>(expert_id) * static_cast<int64_t>(full_n) * static_cast<int64_t>(gemm_k);
    ElementA* ptr_A_curr_batch = const_cast<ElementA*>(Activations) + pre_rows * gemm_k;
    ElementB* ptr_B_curr_batch = const_cast<ElementB*>(Weights) + expert_offset;
    ElementD* ptr_D_curr_batch = Outputs + pre_rows * inter_size;
    ElementBI* ptr_Bias_curr_batch = nullptr;
    if (Bias != static_cast<ElementBI*>(nullptr)) {
      ptr_Bias_curr_batch = const_cast<ElementBI*>(Bias) + expert_id * full_n;
    }

    auto A_tensor = make_moe_tensor<ElementA, LayoutKindA>(ptr_A_curr_batch, gemm_m, gemm_k);
    auto B_gate_tensor = make_gateup_tensor(ptr_B_curr_batch, inter_size, gemm_k, full_n);
    auto B_up_tensor = make_gateup_tensor(ptr_B_curr_batch + inter_size, inter_size, gemm_k, full_n);
    auto D_tensor = make_moe_tensor<ElementD, LayoutKindD>(ptr_D_curr_batch, gemm_m, inter_size);

    while (group_m_id < cumsum_tiles_for_experts) {
      int n_coord = (group_id * wg_tile_n) % gemm_n_pad / wg_tile_n;
      int m_coord = (group_m_id - pre_tiles);
      auto tile_coord = make_coord(m_coord, n_coord, _, 0);

      xe_gemm_fc1_silu<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD>(
          A_tensor,
          B_gate_tensor,
          B_up_tensor,
          ptr_Bias_curr_batch,
          D_tensor,
          tile_coord,
          mma,
          full_n,
          clamp_limit);

      if (local_id == 0) {
        slm_mem[0] = cutlass::atomicAdd(atomic_buffer, 1);
      }
      item.barrier(sycl::access::fence_space::local_space);
      group_id = group_range + slm_mem[0];
      group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
    }

    pre_tiles = cumsum_tiles_for_experts;
  }
}

}  // namespace MoE
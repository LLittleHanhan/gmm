#pragma once

#include <torch/torch.h>
#include "ATen/core/TensorBody.h"

namespace hpc_ops_hopper {

// Hopper/SM90a grouped GEMM (forward only, path one).
//
//   A : [num_tokens, dim]            contiguous CUDA, dtype=fp32/bf16/fp16
//   B : [num_experts, dim, dim_out]  (or [num_experts, dim_out, dim] if transb)
//                                    contiguous CUDA, same dtype as A
//   batch_sizes : [num_experts]      CPU int64
//
// Returns:
//   D : [num_tokens, dim_out]        same dtype as A
//
// Constraints:
//   - trans_a is not supported in this forward-only build (always false).
//   - trans_b may be true (B layout [E, dim_out, dim]) for bf16/fp16.
//   - fp32 path always runs in (trans_a=false, trans_b=true) layout: B is
//     transposed internally to [E, dim_out, dim] contiguous.
torch::Tensor hopper_gmm_forward(torch::Tensor A, torch::Tensor B,
                                 torch::Tensor batch_sizes,
                                 bool transb = false);

}  // namespace hpc_ops_hopper

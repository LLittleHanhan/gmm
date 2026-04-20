// TORCH_LIBRARY bindings for the forward-only Hopper grouped GEMM.
//
// Registers one CUDA op:
//   hopper_gmm_fwd::gmm(Tensor A, Tensor B, Tensor batch_sizes, bool trans_b) -> Tensor
//
// Python side calls it via torch.ops.hopper_gmm_fwd.gmm(...).

#include <torch/library.h>
#include <torch/extension.h>

#include "hopper_gmm_forward.h"

TORCH_LIBRARY(hopper_gmm_fwd, m) {
  m.def("gmm(Tensor A, Tensor B, Tensor batch_sizes, bool trans_b) -> Tensor");
}

TORCH_LIBRARY_IMPL(hopper_gmm_fwd, CUDA, m) {
  m.impl("gmm", &hpc_ops_hopper::hopper_gmm_forward);
}

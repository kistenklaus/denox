#pragma once

#include "memory/hypergraph/ConstGraph.hpp"
#include "model/ComputeOp.hpp"
#include "model/ComputeTensor.hpp"

namespace denox::compiler {

memory::ConstGraph<ComputeTensor, ComputeOp>
fusion_pass(const memory::ConstGraph<ComputeTensor, ComputeOp> &graph);
}

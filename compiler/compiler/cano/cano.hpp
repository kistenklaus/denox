#pragma once

#include "compiler/ir/LinkedModel.hpp"
#include "memory/hypergraph/ConstGraph.hpp"
#include "memory/hypergraph/LinkedGraph.hpp"
#include "model/ComputeOp.hpp"
#include "model/ComputeTensor.hpp"
#include "model/Model.hpp"
namespace denox::compiler {

LinkedModel canonicalize(const Model &model);
} // namespace denox::compiler

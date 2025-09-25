#pragma once

#include "compiler/ir/CanoModel.hpp"
#include "memory/hypergraph/ConstGraph.hpp"
#include "memory/hypergraph/LinkedGraph.hpp"
#include "model/ComputeOp.hpp"
#include "model/ComputeTensor.hpp"
#include "model/Model.hpp"
namespace denox::compiler {

CanoModel canonicalize(const Model &model);
} // namespace denox::compiler

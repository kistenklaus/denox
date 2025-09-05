#pragma once

#include "vkcnn/common/compiler/specialize_tensors.hpp"
#include "vkcnn/common/hypergraph/ConstGraph.hpp"
namespace vkcnn::debug {

void print_const_graph(const hypergraph::ConstGraph<ComputeTensor, ComputeOp>& graph);

void print_const_graph(
    const hypergraph::ConstGraph<compiler::SpecializedTensor, ComputeOp>& graph);
}

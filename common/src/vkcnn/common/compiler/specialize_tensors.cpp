#include "./specialize_tensors.hpp"
#include "vkcnn/common/model/ComputeTensor.hpp"

namespace vkcnn::compiler {

hypergraph::ConstGraph<SpecializedTensor, ComputeOp>
specialize_tensors(const Model &model, const CompileOptions &options) {

  hypergraph::ConstGraph<ComputeTensor, ComputeOp> graph{model.graph()};

  hypergraph::AdjGraph<SpecializedTensor, ComputeOp> specializedGraph;

  // iterate over each node. add nodes to the specialized graph.

  return hypergraph::ConstGraph<SpecializedTensor, ComputeOp>{specializedGraph};
}

} // namespace vkcnn::compiler

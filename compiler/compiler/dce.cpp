#include "compiler/dce.hpp"
#include "memory/container/dynamic_bitset.hpp"
#include "memory/container/vector.hpp"
#include "memory/hypergraph/AdjGraph.hpp"
#include "memory/hypergraph/NodeId.hpp"
#include "model/ComputeOp.hpp"
#include <utility>

namespace denox::compiler {

AdjModel dce(const LinkedModel &model) {
  memory::AdjGraph<ComputeTensor, ComputeOp> adj;
  using LinkedGraph = LinkedModel::Graph;
  using NodeHandle = LinkedGraph::NodeHandle;



  return AdjModel{.graph = std::move(adj),
                  .input = memory::NodeId{0},
                  .output = memory::NodeId{0}};
}

} // namespace denox::compiler

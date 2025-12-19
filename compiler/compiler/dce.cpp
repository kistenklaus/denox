#include "compiler/dce.hpp"
#include "denox/diag/logging.hpp"
#include "denox/memory/container/dynamic_bitset.hpp"
#include "denox/memory/container/small_vector.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/hypergraph/AdjGraph.hpp"
#include "denox/memory/hypergraph/NodeId.hpp"
#include "denox/memory/hypergraph/NullWeight.hpp"
#include "model/ComputeOp.hpp"
#include <utility>

namespace denox::compiler {

OpModel dce(const SpecModel &model) {
  memory::AdjGraph<TensorInstance, ComputeOp> adj;
  using LinkedGraph = SpecModel::Graph;
  using NodeHandle = LinkedGraph::NodeHandle;

  memory::dynamic_bitset visited(model.graph.upperNodeCount());
  memory::dynamic_bitset exists(model.graph.upperNodeCount());
  memory::vector<NodeHandle> stack;
  stack.reserve(model.graph.upperNodeCount());

  memory::vector<memory::NodeId> adjNodes(model.graph.upperNodeCount());
  adjNodes[*model.output->id()] = adj.addNode(model.output->value());

  stack.push_back(model.output);

  while (!stack.empty()) {
    NodeHandle node = stack.back();
    stack.pop_back();
    memory::NodeId id = node->id();
    if (visited[*id]) {
      continue;
    }
    visited[*id] = true;

    for (const auto &edge : node->incoming()) {
      memory::small_vector<memory::NodeId, 2> srcs;
      for (auto &src : edge.srcs()) {
        if (exists[*src.id()]) {
          memory::NodeId adjSrc = adjNodes[*src.id()];
          srcs.push_back(adjSrc);
        } else {
          memory::NodeId adjSrc = adj.addNode(src.value());
          adjNodes[*src.id()] = adjSrc;
          exists[*src.id()] = true;
          srcs.push_back(adjSrc);
          stack.push_back(NodeHandle(src));
        }
      }
      adj.addEdge(std::span<const memory::NodeId>(srcs.begin(), srcs.end()),
                  adjNodes[*id], edge.value(), memory::NullWeight{});
    }
  }

  memory::ConstGraph<TensorInstance, ComputeOp> opGraph{adj};

  return OpModel{
    .graph = std::move(opGraph),
    .input = adjNodes[*model.input->id()],
      .output = adjNodes[*model.output->id()],
  };
}

} // namespace denox::compiler

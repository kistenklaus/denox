#include "denox/compiler/dce/dce.hpp"
#include "denox/diag/logging.hpp"
#include "denox/memory/container/dynamic_bitset.hpp"
#include "denox/memory/container/small_vector.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/hypergraph/AdjGraph.hpp"
#include "denox/memory/hypergraph/NodeId.hpp"
#include "denox/memory/hypergraph/NullWeight.hpp"
#include <utility>

namespace denox::compiler {

ConstModel dce(const SpecModel &model) {
  memory::AdjGraph<TensorInstance, ComputeOp> adj;
  using LinkedGraph = SpecModel::Graph;
  using NodeHandle = LinkedGraph::NodeHandle;

  memory::dynamic_bitset visited(model.graph.upperNodeCount());
  memory::dynamic_bitset exists(model.graph.upperNodeCount());
  memory::vector<NodeHandle> stack;
  stack.reserve(model.graph.upperNodeCount());

  memory::vector<memory::NodeId> adjNodes(model.graph.upperNodeCount());
  memory::vector<memory::NodeId> outputs;
  outputs.reserve(model.outputs.size());
  for (const auto &output : model.outputs) {
    memory::NodeId id = adj.addNode(output->value());
    adjNodes[*output->id()] = id;
    stack.push_back(output);
    outputs.push_back(id);
    exists[*output->id()] = true;
  }

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

  std::vector<memory::NodeId> inputs;
  inputs.reserve(model.inputs.size());
  for (const auto &input : model.inputs) {
    if (!exists[*input->id()]) {
      DENOX_WARN("Found dead input, implicitly pruned!");
      continue;
    }
    inputs.push_back(adjNodes[*input->id()]);
  }

  return ConstModel{
      .graph = std::move(opGraph),
      .inputs = std::move(inputs),
      .outputs = std::move(outputs),
  };
}

} // namespace denox::compiler

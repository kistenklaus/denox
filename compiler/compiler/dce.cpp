#include "compiler/dce.hpp"
#include "diag/logging.hpp"
#include "memory/container/dynamic_bitset.hpp"
#include "memory/container/small_vector.hpp"
#include "memory/container/vector.hpp"
#include "memory/hypergraph/AdjGraph.hpp"
#include "memory/hypergraph/NodeId.hpp"
#include "memory/hypergraph/NullWeight.hpp"
#include "model/ComputeOp.hpp"
#include <utility>

namespace denox::compiler {

AdjModel dce(const LinkedModel &model) {
  memory::AdjGraph<ComputeTensor, ComputeOp> adj;
  using LinkedGraph = LinkedModel::Graph;
  using NodeHandle = LinkedGraph::NodeHandle;

  // NOTE: Just for debugging to determine the exact node count!
  std::size_t nodeCount = 0;
  {
    memory::vector<NodeHandle> stack;
    stack.reserve(model.graph.upperNodeCount());
    memory::dynamic_bitset visited(model.graph.upperNodeCount());
    stack.push_back(model.input);
    while (!stack.empty()) {
      NodeHandle curr = stack.back();
      stack.pop_back();
      memory::NodeId id = curr->id();
      if (visited[id]) {
        continue;
      }
      visited[id] = true;
      nodeCount += 1;
      for (auto &outgoing : curr->outgoing()) {
        stack.push_back(NodeHandle(outgoing.dst()));
      }
    }
  }

  memory::dynamic_bitset visited(model.graph.upperNodeCount());
  memory::vector<NodeHandle> stack;
  stack.reserve(model.graph.upperNodeCount());

  memory::vector<memory::NodeId> adjNodes(model.graph.upperNodeCount());
  adjNodes[model.output->id()] = adj.addNode(model.output->value());

  stack.push_back(model.output);

  while (!stack.empty()) {
    NodeHandle node = stack.back();
    stack.pop_back();
    memory::NodeId id = node->id();
    if (visited[id]) {
      continue;
    }
    visited[id] = true;

    for (const auto &edge : node->incoming()) {
      memory::small_vector<memory::NodeId, 2> srcs;
      for (auto &src : edge.srcs()) {
        if (visited[src.id()]) {
          memory::NodeId adjSrc = adjNodes[src.id()];
          srcs.push_back(adjSrc);
        } else {
          memory::NodeId adjSrc = adj.addNode(src.value());
          adjNodes[src.id()] = adjSrc;
          srcs.push_back(adjSrc);
          stack.push_back(NodeHandle(src));
        }
      }
      adj.addEdge(std::span<const memory::NodeId>(srcs.begin(), srcs.end()),
                  adjNodes[id], edge.value(), memory::NullWeight{});
    }
  }

  std::int64_t pruned = static_cast<std::int64_t>(nodeCount) -
                        static_cast<std::int64_t>(adj.nodeCount());
  if (pruned < 0) {
    DENOX_WARN("Added {} intermediate tensor during dead code elimination."
               "This is wrong and most likely a bug on our side."
               "Please file a github issue! Everything else may still function "
               "correctly, but we are very close to UB at this point.");
  } else if (pruned > 0) {
    DENOX_INFO("Dead code elimination found {} intermediate tensors, which are "
               "not dependent on the input / output. This number is not exact "
               "there might be more intermediate tensors that where pruned!",
               pruned);
  }

  return AdjModel{
      .graph = std::move(adj),
      .input = adjNodes[model.input->id()],
      .output = adjNodes[model.output->id()],
  };
}

} // namespace denox::compiler

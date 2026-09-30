#include "denox/compiler/selection/cache_aware_dispatch_order.hpp"

#include <stdexcept>

namespace denox::compiler::selection {

memory::vector<memory::EdgeId> cache_aware_dispatch_order(
    const memory::ConstGraph<TensorId, SuperGraphEdge,
                             std::chrono::duration<float, std::milli>>
        &minCostSubgraph) {
  // NOTE: This is still far from optimal. 
  // Optimal, would consider written and read memory and order such that 
  // potential cache reuse is maximized.
  // Consider seperate branches
  // I -> A
  // I -> X -> Y -> Z
  // Optimal, would determine total access in both branches, and choose 
  // branch with less access first, to allow for potential reuse when accessing I again in 
  // XYZ branch.

  const std::size_t nodeCount = minCostSubgraph.nodeCount();
  const std::size_t edgeCount = minCostSubgraph.edgeCount();

  memory::vector<std::size_t> incomingEdges(nodeCount, 0);
  memory::vector<std::size_t> remainingSources(edgeCount, 0);
  for (std::size_t i = 0; i < edgeCount; ++i) {
    const memory::EdgeId edge{i};
    ++incomingEdges[*minCostSubgraph.dst(edge)];
    remainingSources[i] = minCostSubgraph.src(edge).size();
  }

  memory::vector<memory::NodeId> readyNodes;
  for (std::size_t i = 0; i < nodeCount; ++i) {
    if (incomingEdges[i] == 0) {
      readyNodes.push_back(memory::NodeId{i});
    }
  }

  memory::vector<memory::EdgeId> readyEdges;
  memory::vector<memory::EdgeId> order;
  order.reserve(edgeCount);

  memory::vector<std::size_t> lastWritten(nodeCount, 0);
  std::size_t writeClock = 0;

  auto releaseReadyEdges = [&]() {
    while (!readyNodes.empty()) {
      const memory::NodeId node = readyNodes.back();
      readyNodes.pop_back();
      for (const memory::EdgeId edge : minCostSubgraph.outgoing(node)) {
        if (--remainingSources[*edge] == 0) {
          readyEdges.push_back(edge);
        }
      }
    }
  };

  for (std::size_t i = 0; i < edgeCount; ++i) {
    if (remainingSources[i] != 0) {
      continue;
    }
    readyEdges.push_back(memory::EdgeId{i});
  }

  while (order.size() != edgeCount) {
    releaseReadyEdges();

    if (readyEdges.empty()) {
      break;
    }

    auto best = readyEdges.begin();
    auto score = [&](memory::EdgeId edge) {
      std::size_t newest = 0;
      for (const memory::NodeId source : minCostSubgraph.src(edge)) {
        newest = std::max(newest, lastWritten[*source]);
      }
      return newest;
    };
    for (auto candidate = best + 1; candidate != readyEdges.end();
         ++candidate) {
      if (score(*candidate) > score(*best) ||
          (score(*candidate) == score(*best) && **candidate < **best)) {
        best = candidate;
      }
    }

    const memory::EdgeId edge = *best;
    readyEdges.erase(best);
    order.push_back(edge);

    const memory::NodeId destination = minCostSubgraph.dst(edge);
    lastWritten[*destination] = ++writeClock;
    if (--incomingEdges[*destination] == 0) {
      readyNodes.push_back(destination);
    }
  }

  if (order.size() != edgeCount) {
    throw std::runtime_error(
        "cache_aware_dispatch_order: graph contains a cycle");
  }

  return order;
}


} // namespace denox::compiler::selection

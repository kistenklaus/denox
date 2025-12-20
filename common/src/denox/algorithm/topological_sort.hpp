#pragma once

#include "denox/memory/container/deque.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include "denox/memory/hypergraph/EdgeId.hpp"
#include "denox/memory/hypergraph/NodeId.hpp"

#include <stdexcept>

namespace denox::algorithm {

template <typename V, typename E>
memory::vector<memory::NodeId>
topologicalSort(const memory::ConstGraph<V, E> &hypergraph) {
  using EdgeId = memory::EdgeId;
  using NodeId = memory::NodeId;

  std::size_t nodeCount = hypergraph.nodeCount();
  std::size_t edgeCount = hypergraph.edgeCount();

  denox::memory::vector<unsigned int> edgeRemainingSrc(edgeCount, 0);
  denox::memory::vector<unsigned int> nodeUnsatisfiedIn(nodeCount, 0);

  for (std::size_t ei = 0; ei < edgeCount; ++ei) {
    EdgeId e{ei};
    auto srcs = hypergraph.src(e);
    edgeRemainingSrc[ei] = static_cast<unsigned int>(srcs.size());
    NodeId d = hypergraph.dst(e);
    nodeUnsatisfiedIn[d]++;
  }

  denox::memory::deque<NodeId> ready;

  for (std::size_t ni = 0; ni < nodeCount; ++ni) {
    if (nodeUnsatisfiedIn[ni] == 0) {
      ready.push_back(NodeId{ni});
    }
  }

  std::vector<NodeId> order;
  order.reserve(nodeCount);

  while (!ready.empty()) {
    NodeId n = ready.front();
    ready.pop_front();
    order.push_back(n);

    for (EdgeId e : hypergraph.outgoing(n)) {
      if (edgeRemainingSrc[e] > 0 && hypergraph.src(e).size() > 0) {

        if (--edgeRemainingSrc[e] == 0) {
          NodeId d = hypergraph.dst(e);
          if (--nodeUnsatisfiedIn[d] == 0) {
            ready.push_back(d);
          }
        }
      }
    }
  }
  if (order.size() != nodeCount) {
    throw std::runtime_error(
        "topologicialSort: graph contains a cycle or a dangling dependency");
  }

  return order;
}

} // namespace denox::algorithm

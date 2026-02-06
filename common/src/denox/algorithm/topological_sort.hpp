#pragma once

#include "denox/memory/container/deque.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include "denox/memory/hypergraph/EdgeId.hpp"
#include "denox/memory/hypergraph/NodeId.hpp"

#include <stdexcept>

namespace denox::algorithm {

template <typename V, typename E, typename W>
memory::vector<memory::NodeId>
topologicalSort(const memory::ConstGraph<V, E, W> &hypergraph) {
  const size_t N = hypergraph.nodeCount();
  const size_t M = hypergraph.edgeCount();

  memory::vector<uint32_t> indegree(N, 0);

  for (std::size_t ei = 0; ei < M; ++ei) {
    memory::EdgeId e{ei};
    memory::NodeId d = hypergraph.dst(e);
    indegree[*d] += static_cast<uint32_t>(hypergraph.src(e).size());
  }

  memory::deque<memory::NodeId> ready;
  for (std::size_t i = 0; i < N; ++i) {
    if (indegree[i] == 0) {
      ready.push_back(memory::NodeId{i});
    }
  }

  memory::vector<memory::NodeId> order;
  order.reserve(N);

  while (!ready.empty()) {
    memory::NodeId u = ready.front();
    ready.pop_front();
    order.push_back(u);

    for (memory::EdgeId e : hypergraph.outgoing(u)) {
      memory::NodeId d = hypergraph.dst(e);
      // u satisfies exactly one src-dependency of d
      if (--indegree[*d] == 0) {
        ready.push_back(d);
      }
    }
  }

  if (order.size() != N) {
    throw std::runtime_error(
        "topologicalSort: graph contains a cycle");
  }

  return order;
}

} // namespace denox::algorithm

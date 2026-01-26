#pragma once

#include "denox/algorithm/topological_sort.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include "denox/memory/hypergraph/EdgeId.hpp"

#include <algorithm>
#include <cstddef>

namespace denox::algorithm {

template <typename V, typename E, typename W>
memory::vector<memory::EdgeId>
topological_sort_edges(const memory::ConstGraph<V, E, W> &graph) {
  using NodeId = memory::NodeId;
  using EdgeId = memory::EdgeId;

  const std::size_t N = graph.nodeCount();
  const std::size_t M = graph.edgeCount();

  // Node topo order + rank map
  memory::vector<NodeId> topoNodes = algorithm::topologicalSort(graph);
  memory::vector<std::size_t> rank(N, 0);
  for (std::size_t i = 0; i < topoNodes.size(); ++i) {
    rank[*topoNodes[i]] = i;
  }

  // Precompute edge keys to avoid repeated scanning in comparator
  memory::vector<std::size_t> dstRank(M, 0);
  memory::vector<std::size_t> maxSrcRank(M, 0);

  for (std::size_t ei = 0; ei < M; ++ei) {
    EdgeId e{ei};
    NodeId d = graph.dst(e);
    dstRank[ei] = rank[*d];

    auto srcs = graph.src(e);
    if (srcs.empty()) {
      maxSrcRank[ei] = 0;
    } else {
      std::size_t mx = 0;
      bool first = true;
      for (NodeId u : srcs) {
        const std::size_t r = rank[*u];
        if (first) {
          mx = r;
          first = false;
        } else {
          mx = std::max(mx, r);
        }
      }
      maxSrcRank[ei] = mx;
    }
  }

  // Order edges by: when they become "ready" (max src rank), then dst rank, then id
  memory::vector<EdgeId> edges;
  edges.reserve(M);
  for (std::size_t ei = 0; ei < M; ++ei) {
    edges.push_back(EdgeId{ei});
  }

  std::sort(edges.begin(), edges.end(), [&](EdgeId a, EdgeId b) {
    const std::size_t ia = *a;
    const std::size_t ib = *b;
    if (maxSrcRank[ia] != maxSrcRank[ib]) return maxSrcRank[ia] < maxSrcRank[ib];
    if (dstRank[ia] != dstRank[ib]) return dstRank[ia] < dstRank[ib];
    return ia < ib;
  });

  return edges;
}

} // namespace denox::algorithm

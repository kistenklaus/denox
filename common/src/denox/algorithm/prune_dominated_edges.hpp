#pragma once

#include "denox/algorithm/hash_combine.hpp"
#include "denox/memory/container/hashmap.hpp"
#include "denox/memory/container/span.hpp"
#include "denox/memory/hypergraph/AdjGraph.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include "denox/memory/hypergraph/NodeId.hpp"
#include <algorithm>

namespace denox::algorithm {

template <typename V, typename E, typename W>
memory::AdjGraph<V, E, W>
prune_duplicate_edges(const memory::ConstGraph<V, E, W> &graph) {
  const size_t N = graph.nodeCount();
  const size_t M = graph.edgeCount();
  memory::AdjGraph<V, E, W> subgraph;

  for (uint64_t i = 0; i < N; ++i) {
    memory::NodeId nid{i};
    const V &n = graph.get(nid);
    memory::NodeId _nid = subgraph.addNode(n);
    assert(_nid == nid);
  }

  struct EdgeKey {
    memory::NodeId dst;
    memory::span<const memory::NodeId> srcs;
  };
  struct EdgeKeyHash {
    size_t operator()(const EdgeKey &e) const {
      uint64_t hash = *e.dst;
      hash = algorithm::hash_combine(hash, 0xD4101D490AAEF014);
      for (const memory::NodeId &src : e.srcs) {
        hash = algorithm::hash_combine(hash, *src);
      }
      return hash;
    }
  };
  struct EdgeKeyComp {
    bool operator()(const EdgeKey &lhs, const EdgeKey &rhs) const {
      if (lhs.dst != rhs.dst) {
        return false;
      }
      return std::ranges::equal(lhs.srcs, rhs.srcs);
    }
  };
  struct EdgeInfo {
    memory::EdgeId id;
    W minWeight;
  };
  memory::hash_map<EdgeKey, EdgeInfo, EdgeKeyHash, EdgeKeyComp> edgeMap;
  edgeMap.reserve(M);

  for (uint64_t i = 0; i < M; ++i) {
    memory::EdgeId eid{i};
    const W &weight = graph.weight(eid);
    EdgeKey key{
        .dst = graph.dst(eid),
        .srcs = graph.src(eid),
    };
    auto it = edgeMap.find(key);
    if (it != edgeMap.end()) {
      EdgeInfo &info = it->second;
      if (weight < info.minWeight) {
        info.minWeight = weight;
        info.id = eid;
      }
    } else {
      edgeMap.emplace(key, EdgeInfo{
                               .id = eid,
                               .minWeight = weight,
                           });
    }
  }
  for (const auto& [key, info] : edgeMap) {
    subgraph.addEdge(key.srcs, key.dst, graph.get(info.id),
        info.minWeight);
  }
  return subgraph;
}

} // namespace denox::algorithm

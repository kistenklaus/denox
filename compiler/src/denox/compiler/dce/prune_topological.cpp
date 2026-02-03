#include "denox/compiler/dce/prune_topological.hpp"
#include "denox/algorithm/minimum_const_subgraph.hpp"
#include "denox/algorithm/prune_dominated_edges.hpp"
#include "denox/memory/container/hashmap.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/hypergraph/AdjGraph.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include <exception>

static denox::memory::ConstGraph<denox::memory::NodeId,
                                 denox::memory::vector<denox::memory::EdgeId>,
                                 uint32_t>
construct_topological_graph(denox::compiler::SuperGraph &supergraph) {
  using namespace denox;
  const uint32_t N = static_cast<uint32_t>(supergraph.graph.nodeCount());
  const uint32_t M = static_cast<uint32_t>(supergraph.graph.edgeCount());

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
    memory::vector<memory::EdgeId> multi_edges;
    uint32_t minCost;
  };
  memory::hash_map<EdgeKey, EdgeInfo, EdgeKeyHash, EdgeKeyComp> edgeMap;
  edgeMap.reserve(M);

  for (uint32_t i = 0; i < M; ++i) {
    memory::EdgeId eid{i};
    const auto &edge = supergraph.graph.get(eid);
    EdgeKey key{.dst = supergraph.graph.dst(eid),
                .srcs = supergraph.graph.src(eid)};
    const uint32_t cost = static_cast<uint32_t>(edge.dispatches.size());
    auto it = edgeMap.find(key);
    if (it != edgeMap.end()) {
      EdgeInfo &info = it->second;
      if (cost < info.minCost) {
        info.multi_edges.clear();
        info.multi_edges.push_back(eid);
        info.minCost = cost;
      } else if (cost == info.minCost) {
        info.multi_edges.push_back(eid);
      } else {
        // ignore edge.
      }
    } else {
      edgeMap.emplace(key, EdgeInfo{
                               .multi_edges = {eid},
                               .minCost = cost,
                           });
    }
  }

  memory::AdjGraph<memory::NodeId, memory::vector<memory::EdgeId>, uint32_t>
      tgraph_builder;
  for (uint32_t i = 0; i < N; ++i) {
    memory::NodeId nid{i};
    [[maybe_unused]] memory::NodeId _nid = tgraph_builder.addNode(nid);
    assert(nid == _nid);
  }

  for (const auto &[key, info] : edgeMap) {
    tgraph_builder.addEdge(key.srcs, key.dst, info.multi_edges, info.minCost);
  }
  return memory::ConstGraph<memory::NodeId, memory::vector<memory::EdgeId>,
                            uint32_t>(std::move(tgraph_builder));
}

void denox::compiler::prune_topological(SuperGraph &supergraph) {
  fmt::println("supergraph = (N = {}, M = {})", supergraph.graph.nodeCount(),
               supergraph.graph.edgeCount());

  for (const auto i : supergraph.inputs) {
    fmt::println("input: {}", *i);
  }

  for (const auto o : supergraph.outputs) {
    fmt::println("output: {}", *o);
  }

  // weak reachability check
  for (uint32_t i = 0; i < supergraph.graph.nodeCount(); ++i) {
    memory::NodeId nid{i};
    if (std::ranges::find(supergraph.inputs, nid) == supergraph.inputs.end()) {
      if (supergraph.graph.incoming(nid).empty()) {
        fmt::println("UNREACHABLE NODE-ID: {}", i);
      }
      assert(!supergraph.graph.incoming(nid).empty());
    }
  }

  memory::ConstGraph<memory::NodeId, memory::vector<memory::EdgeId>, uint32_t>
      tgraph = construct_topological_graph(supergraph);

  fmt::println("tgraph = (N = {}, M = {})", tgraph.nodeCount(),
               tgraph.edgeCount());

  auto minCostSubgraph = algorithm::minimum_cost_subgraph(
      tgraph, supergraph.inputs, supergraph.outputs);
  uint32_t minCost = 0;
  for (uint32_t i = 0; i < minCostSubgraph.edgeCount(); ++i) {
    minCost += minCostSubgraph.weight(memory::EdgeId{i});
  }
  fmt::println("min-cost = {}", minCost);

  std::terminate();
}

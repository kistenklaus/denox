#include "denox/compiler/dce/prune_topological.hpp"
#include "denox/algorithm/minimum_const_subgraph.hpp"
#include "denox/algorithm/minimum_cost_subgraphs.hpp"
#include "denox/algorithm/prune_dominated_edges.hpp"
#include "denox/algorithm/topological_edge_sort.hpp"
#include "denox/algorithm/topological_sort.hpp"
#include "denox/memory/container/hashmap.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/hypergraph/AdjGraph.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include <absl/strings/internal/str_format/extension.h>
#include <exception>
#include <limits>

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

  memory::ConstGraph<memory::NodeId, memory::vector<memory::EdgeId>, uint32_t>
      tgraph = construct_topological_graph(supergraph);

  fmt::println("tgraph = (N={}, M={})", tgraph.nodeCount(), tgraph.edgeCount());

  auto minimum_cost_subgraph = algorithm::minimum_cost_subgraph(
      tgraph, supergraph.inputs, supergraph.outputs);

  // for (uint32_t i = 0; i < tgraph.nodeCount(); ++i) {
  //   memory::NodeId nid{i};
  //   memory::NodeId sid = tgraph.get(nid);
  //   fmt::println("Node: {}", sid);
  //   const auto& node = supergraph.tensors[supergraph.graph.get(sid).index];
  //   fmt::println("FORMAT: {} , {}", node.info.format, node.info.channels->constant());
  // }

  // for (uint32_t i = 0; i < tgraph.edgeCount(); ++i) {
  //   memory::EdgeId eid{i};
  //   auto multiedge = tgraph.get(eid);
  //   memory::EdgeId sid = multiedge.front();
  //   const auto &edge = supergraph.graph.get(sid);
  //   auto srcs = supergraph.graph.src(sid);
  //   auto dst = supergraph.graph.dst(sid);
  //   if (srcs.size() == 1) {
  //     fmt::println("EDGE: {} -> {}", srcs.front(), dst);
  //   } else {
  //     fmt::println("EDGE: {},{} -> {}", srcs[0], srcs[1], dst);
  //   }
  //   for (const auto& dispatch : edge.dispatches) {
  //     fmt::println("dispatch: {}", *dispatch.info.operation);
  //   }
  // }

  uint32_t minimum_cost = 0;
  for (uint32_t i = 0; i < minimum_cost_subgraph.edgeCount(); ++i) {
    memory::EdgeId eid{i};
    minimum_cost += minimum_cost_subgraph.weight(eid);
  }

  auto all_minimum_cost_subgraphs = algorithm::minimum_cost_subgraphs(
      tgraph, supergraph.inputs, supergraph.outputs, minimum_cost);

  fmt::println("N = {}, M = {}", all_minimum_cost_subgraphs.nodeCount(),
               all_minimum_cost_subgraphs.edgeCount());

  // fmt::println("SURVIVING-EDGES:");
  // for (uint32_t i = 0; i < tgraph.edgeCount(); ++i) {
  //   memory::EdgeId eid{i};
  //   auto multiedge = all_minimum_cost_subgraphs.get(eid);
  //   memory::EdgeId sid = multiedge.front();
  //   const auto &edge = supergraph.graph.get(sid);
  //   auto srcs = supergraph.graph.src(sid);
  //   auto dst = supergraph.graph.dst(sid);
  //   if (srcs.size() == 1) {
  //     fmt::println("EDGE: {} -> {}", srcs.front(), dst);
  //   } else {
  //     fmt::println("EDGE: {},{} -> {}", srcs[0], srcs[1], dst);
  //   }
  //   for (const auto& dispatch : edge.dispatches) {
  //     fmt::println("dispatch: {}", *dispatch.info.operation);
  //   }
  // }


  std::terminate();
}

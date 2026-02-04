#include "denox/compiler/dce/prune_topological.hpp"
#include "denox/algorithm/minimum_const_subgraph.hpp"
#include "denox/algorithm/minimum_cost_subgraphs.hpp"
#include "denox/algorithm/prune_dominated_edges.hpp"
#include "denox/algorithm/topological_edge_sort.hpp"
#include "denox/algorithm/topological_sort.hpp"
#include "denox/memory/container/dynamic_bitset.hpp"
#include "denox/memory/container/hashmap.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/hypergraph/AdjGraph.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include <absl/strings/internal/str_format/extension.h>
#include <exception>
#include <limits>

static denox::memory::ConstGraph<denox::memory::NodeId,
                                 denox::memory::vector<denox::memory::EdgeId>,
                                 uint32_t>
construct_topological_graph(const denox::compiler::SuperGraph &supergraph) {
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

  memory::ConstGraph<memory::NodeId, memory::vector<memory::EdgeId>, uint32_t>
      tgraph = construct_topological_graph(supergraph);

  auto all_minimum_cost_subgraphs = algorithm::minimum_cost_subgraphs(
      tgraph, supergraph.inputs, supergraph.outputs);

  if (all_minimum_cost_subgraphs.edgeCount() == 0) {
    // TODO: Proper error message (issue #92)
    throw std::runtime_error("Failed to implement model");
  }

  memory::vector<memory::optional<memory::NodeId>> nodeRemap(
      supergraph.graph.nodeCount(), memory::nullopt);

  // Reconstruct supergraph
  memory::AdjGraph<TensorId, SuperGraphEdge> subgraph;

  for (uint32_t e = 0; e < all_minimum_cost_subgraphs.edgeCount(); ++e) {
    memory::EdgeId eid{e};
    const auto &multiedge = all_minimum_cost_subgraphs.get(eid);
    assert(!multiedge.empty());
    const memory::span<const memory::NodeId> srcs =
        supergraph.graph.src(multiedge.front());
    const memory::NodeId dst = supergraph.graph.dst(multiedge.front());

    memory::small_vector<memory::NodeId, 2> new_srcs;
    for (memory::NodeId src : srcs) {
      if (!nodeRemap[*src]) {
        memory::NodeId new_nid = subgraph.addNode(supergraph.graph.get(src));
        nodeRemap[*src] = new_nid;
      }
      memory::NodeId new_nid = *nodeRemap[*src];
      new_srcs.push_back(new_nid);
    }
    if (!nodeRemap[*dst]) {
      memory::NodeId new_nid = subgraph.addNode(supergraph.graph.get(dst));
      nodeRemap[*dst] = new_nid;
    }
    memory::NodeId new_dst = *nodeRemap[*dst];

    for (memory::EdgeId seid : multiedge) {
      subgraph.addEdge(new_srcs, new_dst,
                       std::move(supergraph.graph.get_rvalue(seid)));
    }
  }

  supergraph.graph =
      memory::ConstGraph<TensorId, SuperGraphEdge>(std::move(subgraph));

  // remap inputs
  for (auto &nid : supergraph.inputs) {
    assert(nid != memory::NodeId{});
    assert(nodeRemap[*nid]);
    nid = *nodeRemap[*nid];
  }

  for (auto &nid : supergraph.outputs) {
    assert(nid != memory::NodeId{});
    assert(nodeRemap[*nid]);
    nid = *nodeRemap[*nid];
  }
}

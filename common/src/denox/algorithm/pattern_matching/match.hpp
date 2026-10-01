#pragma once

#include "denox/algorithm/pattern_matching/ConstGraphMatch.hpp"
#include "denox/algorithm/pattern_matching/EdgePattern.fwd.hpp"
#include "denox/algorithm/pattern_matching/GraphPattern.hpp"
#include "denox/algorithm/pattern_matching/NodePattern.fwd.hpp"
#include "denox/memory/coroutines/generator.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include "denox/memory/hypergraph/EdgeId.hpp"
#include <cassert>
#include <variant>
#include <vector>

namespace denox::algorithm {

namespace pattern_matching::details {

template <typename V, typename E, typename W>
memory::generator<ConstGraphMatch<V, E, W>>
match_all_rec(const NodePatternHandle<V, E, W> &nodePattern,
              const memory::ConstGraph<V, E, W> &graph, memory::NodeId nodeId);

template <typename V, typename E, typename W>
memory::generator<ConstGraphMatch<V, E, W>>
match_all_rec(const EdgePatternHandle<V, E, W> &edgePattern,
              const memory::ConstGraph<V, E, W> &graph, memory::EdgeId edgeId) {
  // Edge-level predicate (value/weight/rank)
  if (!(*edgePattern)(graph, edgeId)) {
    co_return;
  }

  const std::size_t nodeCount = edgePattern->details()->nextNodePatternId;
  const std::size_t edgeCount = edgePattern->details()->nextEdgePatternId;
  ConstGraphMatch<V, E, W> base{nodeCount, edgeCount};
  base.registerMatch(edgePattern, edgeId);

  // Compose constraints: dst (optional) and ordered sources (optional/sparse).
  std::vector<ConstGraphMatch<V, E, W>> partials;
  partials.reserve(4);
  partials.push_back(base);

  // 1) Dst constraint (if any)
  if (auto dstPat = edgePattern->getDst(); dstPat != nullptr) {
    const memory::NodeId dstId = graph.dst(edgeId);
    std::vector<ConstGraphMatch<V, E, W>> nextParts;
    for (const auto &p : partials) {
      for (const auto &dstMatch :
           match_all_rec<V, E, W>(dstPat, graph, dstId)) {
        ConstGraphMatch<V, E, W> merged = p;
        if (merged.mergeMatches(dstMatch)) {
          nextParts.push_back(std::move(merged));
        }
      }
    }
    if (nextParts.empty())
      co_return;
    partials.swap(nextParts);
  }

  // 2) Ordered sources constraints (sparse indices allowed)
  const auto srcReq =
      edgePattern->getSrcs(); // span of handles (may contain nullptrs)
  const auto srcSpan = graph.src(edgeId); // actual source nodes for this edge
  if (!srcReq.empty()) {
    // If any required index is out of bounds, fail early.
    for (std::size_t i = 0; i < srcReq.size(); ++i) {
      if (srcReq[i] != nullptr && i >= srcSpan.size())
        co_return;
    }

    for (std::size_t i = 0; i < srcReq.size(); ++i) {
      auto np = srcReq[i];
      if (np == nullptr)
        continue; // unconstrained at this index
      const memory::NodeId sid = srcSpan[i];
      std::vector<ConstGraphMatch<V, E, W>> nextParts;
      for (const auto &p : partials) {
        for (const auto &sMatch : match_all_rec<V, E, W>(np, graph, sid)) {
          ConstGraphMatch<V, E, W> merged = p;
          if (merged.mergeMatches(sMatch)) {
            nextParts.push_back(std::move(merged));
          }
        }
      }
      if (nextParts.empty())
        co_return;
      partials.swap(nextParts);
    }
  }

  // Yield all combined assignments for this edge.
  for (const auto &m : partials) {
    co_yield m;
  }
}

template <typename V, typename E, typename W>
memory::generator<ConstGraphMatch<V, E, W>>
match_all_rec(const NodePatternHandle<V, E, W> &nodePattern,
              const memory::ConstGraph<V, E, W> &graph, memory::NodeId nodeId) {
  if (!(*nodePattern)(graph, nodeId)) {
    co_return;
  }

  const std::size_t nodeCount = nodePattern->details()->nextNodePatternId;
  const std::size_t edgeCount = nodePattern->details()->nextEdgePatternId;
  ConstGraphMatch<V, E, W> base{nodeCount, edgeCount};
  base.registerMatch(nodePattern, nodeId);

  const auto incomingReq =
      nodePattern->getIncoming(); // span of EdgePatternHandle
  const auto outgoingReq =
      nodePattern->getOutgoing(); // span of EdgePatternHandle
  memory::span<const memory::EdgeId> incomingAdj =
      graph.incoming(nodeId); // span of EdgeId
  memory::span<const memory::EdgeId> outgoingAdj =
      graph.outgoing(nodeId); // span of EdgeId

  // Helper: satisfy a set of edge-requirements against an adjacency list,
  // with backtracking and "no edge reuse" within the set.
  auto satisfy_edge_set =
      [&](const std::span<const EdgePatternHandle<V, E, W>> &reqs,
          memory::span<const memory::EdgeId> adj,
          const ConstGraphMatch<V, E, W> &seed)
      -> std::vector<ConstGraphMatch<V, E, W>> {
    std::vector<ConstGraphMatch<V, E, W>> results;

    if (reqs.empty()) {
      results.push_back(seed);
      return results;
    }

    const std::size_t K = reqs.size();

    // Precompute candidate matches per requirement over adjacency edges.
    std::vector<std::size_t> start(K, 0), count(K, 0);
    std::vector<ConstGraphMatch<V, E, W>> pool;
    pool.reserve(32);
    std::vector<memory::EdgeId> pEdge;
    pEdge.reserve(32);

    std::size_t cursor = 0;
    for (std::size_t r = 0; r < K; ++r) {
      start[r] = cursor;
      const auto &ep = reqs[r];
      for (memory::EdgeId eid : adj) {
        for (const auto &em : match_all_rec<V, E, W>(ep, graph, eid)) {
          pool.push_back(em);
          pEdge.push_back(eid);
          ++count[r];
          ++cursor;
        }
      }
      if (count[r] == 0) {
        return results; // empty => unsatisfiable
      }
    }

    // Backtrack over requirement indices
    std::vector<std::size_t> idx(K, 0);
    std::vector<ConstGraphMatch<V, E, W>> partials;
    partials.reserve(K + 1);
    partials.push_back(seed);
    std::vector<memory::EdgeId> used;
    used.reserve(K);

    std::size_t d = 0;
    while (true) {
      if (d == K) {
        results.push_back(partials.back());
        if (d == 0)
          break;
        --d;
        if (!used.empty())
          used.pop_back();
        ++idx[d];
        partials.pop_back();
        continue;
      }

      const std::size_t begin = start[d];
      const std::size_t end = start[d] + count[d];
      bool advanced = false;

      while (begin + idx[d] < end) {
        const std::size_t pidx = begin + idx[d];
        const auto eid = pEdge[pidx];

        bool reuse = false;
        for (auto u : used) {
          if (u == eid) {
            reuse = true;
            break;
          }
        }
        if (!reuse) {
          ConstGraphMatch<V, E, W> merged = partials.back();
          if (merged.mergeMatches(pool[pidx])) {
            partials.push_back(std::move(merged));
            used.push_back(eid);
            ++d;
            if (d < K)
              idx[d] = 0;
            advanced = true;
            break;
          }
        }
        ++idx[d];
      }

      if (!advanced) {
        if (d == 0)
          break;
        idx[d] = 0;
        --d;
        if (!used.empty())
          used.pop_back();
        ++idx[d];
        if (partials.size() > d + 1)
          partials.pop_back();
      }
    }

    return results;
  };

  // First satisfy incoming, then outgoing, chaining the results.
  std::vector<ConstGraphMatch<V, E, W>> afterIncoming =
      satisfy_edge_set(incomingReq, incomingAdj, base);

  if (afterIncoming.empty()) {
    co_return;
  }

  for (const auto &mid : afterIncoming) {
    std::vector<ConstGraphMatch<V, E, W>> finals =
        satisfy_edge_set(outgoingReq, outgoingAdj, mid);
    if (finals.empty())
      continue;
    for (const auto &m : finals)
      co_yield m;
  }
}

} // namespace pattern_matching::details

template <typename V, typename E, typename W>
memory::generator<ConstGraphMatch<V, E, W>>
match_all(const GraphPattern<V, E, W> &pattern,
          const memory::ConstGraph<V, E, W> &graph) {
  const auto &root = pattern.root();
  if (std::holds_alternative<NodePatternHandle<V, E, W>>(root)) {
    const auto &nodePattern = std::get<NodePatternHandle<V, E, W>>(root);
    for (std::uint64_t n = 0; n < graph.nodeCount(); ++n) {
      memory::NodeId nid{n};
      for (const auto &match :
           pattern_matching::details::match_all_rec<V, E, W>(nodePattern, graph,
                                                             nid)) {
        co_yield match;
      }
    }
  } else if (std::holds_alternative<EdgePatternHandle<V, E, W>>(root)) {
    const auto &edgePattern = std::get<EdgePatternHandle<V, E, W>>(root);
    for (std::uint64_t e = 0; e < graph.edgeCount(); ++e) {
      memory::EdgeId eid{e};
      for (const auto &match :
           pattern_matching::details::match_all_rec<V, E, W>(edgePattern, graph,
                                                             eid)) {
        co_yield match;
      }
    }
  }
}

template <typename V, typename E, typename W>
std::optional<ConstGraphMatch<V, E, W>>
match_first(const GraphPattern<V, E, W> &pattern,
            const memory::ConstGraph<V, E, W> &graph) {
  for (const auto &m : match_all<V, E, W>(pattern, graph)) {
    return m; // first result wins
  }
  return std::nullopt;
}

} // namespace denox::algorithm

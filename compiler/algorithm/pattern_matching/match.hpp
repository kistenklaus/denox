#pragma once

#include "algorithm/pattern_matching/EdgePattern.fwd.hpp"
#include "algorithm/pattern_matching/LinkedGraphMatch.hpp"
#include "memory/container/dynamic_bitset.hpp"
#include "memory/container/vector.hpp"
#include "memory/coroutines/generator.hpp"

#include "algorithm/pattern_matching/ConstGraphMatch.hpp"
#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "algorithm/pattern_matching/NodePattern.fwd.hpp"
#include "memory/hypergraph/ConstGraph.hpp"
#include "memory/hypergraph/EdgeId.hpp"
#include "memory/hypergraph/LinkedGraph.hpp"
#include "memory/hypergraph/NullWeight.hpp"
#include <cassert>
#include <stdexcept>
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
  if (!(*edgePattern)(graph, edgeId)) {
    co_return;
  }

  const std::size_t nodeCount = edgePattern->details()->nextNodePatternId;
  const std::size_t edgeCount = edgePattern->details()->nextEdgePatternId;
  ConstGraphMatch<V, E, W> base{nodeCount, edgeCount};
  base.registerMatch(edgePattern, edgeId);

  if (auto dstPat = edgePattern->getDst(); dstPat != nullptr) {
    const memory::NodeId dstId = graph.dst(edgeId);
    for (const auto &dstMatch : match_all_rec<V, E, W>(dstPat, graph, dstId)) {
      ConstGraphMatch<V, E, W> merged = base;
      if (merged.mergeMatches(dstMatch)) {
        co_yield merged;
      }
    }
    co_return;
  }

  co_yield base;
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

  const auto &outgoingMatch = nodePattern->getOutgoing();
  const auto outgoingSpan = graph.outgoing(nodeId);

  if (outgoingMatch.empty()) {
    co_yield base;
    co_return;
  }

  const std::size_t K = outgoingMatch.size();
  std::vector<std::size_t> start(K, 0);
  std::vector<std::size_t> count(K, 0);
  std::vector<ConstGraphMatch<V, E, W>> pool;
  std::vector<memory::EdgeId> poolEdge;
  pool.reserve(32);
  poolEdge.reserve(32);

  std::size_t cursor = 0;
  for (std::size_t pi = 0; pi < K; ++pi) {
    start[pi] = cursor;
    const auto &ep = outgoingMatch[pi];

    for (memory::EdgeId eid : outgoingSpan) {
      for (const auto &cand : match_all_rec<V, E, W>(ep, graph, eid)) {
        pool.push_back(cand);
        poolEdge.push_back(eid);
        ++count[pi];
        ++cursor;
      }
    }
    if (count[pi] == 0) {
      co_return;
    }
  }

  std::vector<std::size_t> idx(K, 0);
  std::vector<ConstGraphMatch<V, E, W>> partials;
  partials.reserve(K + 1);
  partials.push_back(base);

  std::vector<memory::EdgeId> usedEdges;
  usedEdges.reserve(K);

  std::size_t d = 0;
  while (true) {
    if (d == K) {
      co_yield partials.back();
      if (d == 0)
        break;
      --d;
      if (!usedEdges.empty())
        usedEdges.pop_back();
      ++idx[d];
      partials.pop_back();
      continue;
    }

    const std::size_t begin = start[d];
    const std::size_t end = start[d] + count[d];

    bool advanced = false;
    while (begin + idx[d] < end) {
      const std::size_t pidx = begin + idx[d];
      const auto eid = poolEdge[pidx];

      bool alreadyUsed = false;
      for (auto ue : usedEdges) {
        if (ue == eid) {
          alreadyUsed = true;
          break;
        }
      }
      if (!alreadyUsed) {
        ConstGraphMatch<V, E, W> merged = partials.back();
        if (merged.mergeMatches(pool[pidx])) {
          partials.push_back(std::move(merged));
          usedEdges.push_back(eid);
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
      if (!usedEdges.empty())
        usedEdges.pop_back();
      ++idx[d];
      if (partials.size() > d + 1)
        partials.pop_back();
    }
  }
}

template <typename V, typename E, typename W, typename Allocator>
memory::generator<LinkedGraphMatch<V, E, W, Allocator>> mutable_match_all(
    const NodePatternHandle<V, E, W> &nodePattern,
    const typename memory::LinkedGraph<V, E, W, Allocator>::NodeHandle &node) {
  using LinkedGraph = memory::LinkedGraph<V, E, W, Allocator>;
  using NodeHandle = typename LinkedGraph::NodeHandle;
  using Edge = typename LinkedGraph::Edge;
  using EdgeIt = typename LinkedGraph::EdgeIt;
  using EdgeCtrl =
      pattern_matching::details::EdgeMatchControl<V, E, W, Allocator>;
  using EdgeMatchWrap = EdgeMatch<V, E, W, Allocator>;
  using Match = LinkedGraphMatch<V, E, W, Allocator>;

  if (!nodePattern->template mutable_predicate<Allocator>(node)) {
    co_return;
  }

  const std::size_t nodeCount = nodePattern->details()->nextNodePatternId;
  const std::size_t edgeCount = nodePattern->details()->nextEdgePatternId;

  Match base{nodeCount, edgeCount};
  base.registerMatch(nodePattern, node);

  const auto req = nodePattern->getOutgoing();
  const std::size_t K = req.size();

  if (K == 0) {
    co_yield base;
    co_return;
  }

  memory::vector<const Edge *> usedEdges;
  usedEdges.reserve(K);
  memory::vector<EdgeCtrl *> ctrls;
  ctrls.reserve(K);

  auto solve = [&](auto &self, std::size_t i,
                   const Match &accum) -> memory::generator<Match> {
    if (i == K) {
      for (EdgeCtrl *cb : ctrls) {
        if (cb && cb->dirty()) {
          co_return;
        }
      }
      co_yield accum;
      co_return;
    }

    auto epat = req[i];
    EdgeIt it = node->outgoing().begin();
    const EdgeIt end = node->outgoing().end();

    while (it != end) {
      EdgeIt curr = it;
      EdgeCtrl cb(node, curr);
      it = cb.nextIterator();

      Edge &edgeRef = *curr;
      const Edge *edgePtr = &edgeRef;

      bool alreadyUsed = false;
      for (const Edge *seen : usedEdges) {
        if (seen == edgePtr) {
          alreadyUsed = true;
          break;
        }
      }

      if (alreadyUsed)
        continue;

      if (!epat->template mutable_predicate<Allocator>(edgeRef))
        continue;
      Match next = accum;
      next.registerMatch(epat, EdgeMatchWrap{&cb});

      if (auto dstPat = epat->getDst(); dstPat != nullptr) {
        NodeHandle child = edgeRef.dst();
        for (const auto &childMatch :
             mutable_match_all<V, E, W, Allocator>(dstPat, child)) {
          Match merged = next;
          if (!merged.mergeMatches(childMatch))
            continue;

          usedEdges.push_back(edgePtr);
          ctrls.push_back(&cb);
          for (const auto &m2 : self(self, i + 1, merged)) {
            co_yield m2;
          }
          ctrls.pop_back();
          usedEdges.pop_back();
        }
      } else {
        usedEdges.push_back(edgePtr);
        ctrls.push_back(&cb);
        for (const auto &m2 : self(self, i + 1, next)) {
          co_yield m2;
        }
        ctrls.pop_back();
        usedEdges.pop_back();
      }
    }
  };

  for (const auto &m : solve(solve, 0, base)) {
    co_yield m;
  }
}

}; // namespace pattern_matching::details

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
           pattern_matching::details::match_all_rec(nodePattern, graph, nid)) {
        co_yield match;
      }
    }
  } else if (std::holds_alternative<EdgePatternHandle<V, E, W>>(root)) {
    const auto &edgePattern = std::get<EdgePatternHandle<V, E, W>>(root);
    for (std::uint64_t e = 0; e < graph.edgeCount(); ++e) {
      denox::memory::EdgeId eid{e};
      for (const auto &match :
           pattern_matching::details::match_all_rec(edgePattern, graph, eid)) {
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

/**
Pattern matching over LinkedGraph (with in-yield mutation)

This generator enumerates matches of a GraphPattern that are reachable from a
given root node in a LinkedGraph. Unlike the ConstGraph version, callers may
mutate the graph between yields. That flexibility comes with strict rules.

Scope and traversal

* Only the subgraph reachable from the supplied root node is explored.
* The relative order of yielded matches is unspecified and may change if you
  mutate the graph between yields.
* The implementation assumes an acyclic graph (LinkedGraph is intended for
  DAGs).

Lifetime and usage rules (very important)

1. Treat each yielded LinkedGraphMatch as an temporary view into the matcher’s
   internal state. Do not persist a LinkedGraphMatch, any EdgeMatch obtained
   from it, or any iterator returned by an EdgeMatch, after you advance the
   generator (i.e., after the next co_yield / resume). Using them afterwards is
   undefined behavior.

2. If you need to identify a particular edge during the yield, use
   EdgeMatch::ptr() as a transient identity only. Do not dereference or store
   it beyond the current yield.

3. Nodes actively on the matcher’s stack are pinned via NodeHandle. Downstream
   removals that would otherwise collect them are deferred until they’re no
   longer referenced by the algorithm.

Permitted mutations between yields

* Erasing the currently matched edge for an edge-pattern via EdgeMatch::erase().
  The matcher will resume from the “next” position defined by
  OutgoingList::erase().
as UB.
* Inserting new outgoing edges using insert_after(...). Newly inserted edges
  may be considered later at the same depth; their ordering relative to
  existing edges is unspecified. If you want to maximize the chance the new
  edge is considered soon at the current depth, prefer inserting after the
  iterator returned by EdgeMatch::nextOutgoingIterator().
* Erasing edges unrelated to the currently matched one. The matcher may
  backtrack to the earliest affected depth and continue.
* Erasing nodes that are not currently pinned by the matcher’s internal
  NodeHandles.

Prohibited or unspecified mutations

1. Do not create/add nodes while this generator is active. The matcher uses
   NodeId for visitation; adding nodes can cause an ABA problem and results in
   UB.

2. Do not mutate node or edge payloads (values or weights) of elements that the
   current invocation has visited or may visit. Predicate results are not
   required to be revalidated after selection; changing payloads can cause the
   matcher to yield inconsistent results.

3. Do not erase a matched edge via any API other than EdgeMatch::erase() during
   the yield in which it is exposed. Doing so bypasses the matcher’s cursor
   and is UB.

Postconditions per yield

* Every satisfied NodePattern in the pattern is bound to a NodeHandle accessible
  via the match.
* Every satisfied EdgePattern is bound to an EdgeMatch accessible via
  the match.

**/
template <typename V, typename E, typename W = memory::NullWeight,
          typename Allocator = memory::mallocator>
memory::generator<LinkedGraphMatch<V, E, W, Allocator>>
match_all(const GraphPattern<V, E, W> &pattern,
          const typename memory::LinkedGraph<V, E, W, Allocator>::NodeHandle
              &rootNode) {

  using LinkedGraph = memory::LinkedGraph<V, E, W, Allocator>;
  using NodeHandle = typename LinkedGraph::NodeHandle;

  if (std::holds_alternative<std::monostate>(pattern.root())) {
    co_return;
  }
  if (std::holds_alternative<EdgePatternHandle<V, E, W>>(pattern.root())) {
    throw std::runtime_error(
        "Matching a LinkedGraph with an edge root is currently not supported!");
  }

  NodePatternHandle<V, E, W> rootpattern =
      std::get<NodePatternHandle<V, E, W>>(pattern.root());

  memory::vector<NodeHandle> stack;
  stack.reserve(rootNode.upperNodeCount());
  stack.push_back(rootNode);

  memory::dynamic_bitset visited(rootNode.upperNodeCount() + 1);

  while (!stack.empty()) {
    NodeHandle node = stack.back();
    stack.pop_back();

    memory::NodeId nid = node->id();
    if (visited[nid]) {
      continue;
    }
    visited[nid] = true;

    for (const auto &match :
         pattern_matching::details::mutable_match_all<V, E, W, Allocator>(
             rootpattern, node)) {
      co_yield match;
    }

    for (const auto &e : node->outgoing()) {
      stack.push_back(NodeHandle(e.dst())); // pin dst
    }
  }
}

} // namespace denox::algorithm

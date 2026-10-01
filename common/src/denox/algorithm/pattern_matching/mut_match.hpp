#pragma once

#include "denox/algorithm/pattern_matching/GraphPattern.hpp"
#include "denox/algorithm/pattern_matching/LinkedGraphMatch.hpp"
#include "denox/memory/container/dynamic_bitset.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/coroutines/generator.hpp"
#include "denox/memory/hypergraph/LinkedGraph.hpp"
#include "denox/memory/hypergraph/NullWeight.hpp"
#include <stdexcept>
#include <variant>

namespace denox::algorithm {
namespace pattern_matching::details {

template <typename V, typename E, typename W, typename Allocator>
memory::generator<LinkedGraphMatch<V, E, W, Allocator>> mutable_match_all(
    const NodePatternHandle<V, E, W> &nodePattern,
    const typename memory::LinkedGraph<V, E, W, Allocator>::NodeHandle &node,
    typename EdgeMatchControl<V, E, W, Allocator>::Context &context) {
  using LinkedGraph = memory::LinkedGraph<V, E, W, Allocator>;
  using NodeHandle = typename LinkedGraph::NodeHandle;
  using Edge = typename LinkedGraph::Edge;
  using EdgeIt = typename LinkedGraph::EdgeIt;
  using EdgeCtrl =
      pattern_matching::details::EdgeMatchControl<V, E, W, Allocator>;
  using EdgeMatchWrap = EdgeMatch<V, E, W, Allocator>;
  using Match = LinkedGraphMatch<V, E, W, Allocator>;

  if (!nodePattern->template mutable_predicate<Allocator>(node))
    co_return;
  Match base{nodePattern->details()->nextNodePatternId,
             nodePattern->details()->nextEdgePatternId};
  base.registerMatch(nodePattern, node);
  std::vector<std::pair<EdgePatternHandle<V, E, W>, MatchDirection>> req;
  for (auto edge : nodePattern->getIncoming())
    req.emplace_back(edge, MatchDirection::Incoming);
  for (auto edge : nodePattern->getOutgoing())
    req.emplace_back(edge, MatchDirection::Outgoing);
  const std::size_t K = req.size();
  if (K == 0) {
    co_yield base;
    co_return;
  }

  memory::vector<const Edge *> usedEdges;
  memory::vector<EdgeCtrl *> ctrls;
  usedEdges.reserve(K);
  ctrls.reserve(K);
  auto solve = [&](auto &self, std::size_t i,
                   const Match &accum) -> memory::generator<Match> {
    if (!accum.valid())
      co_return;
    if (i == K) {
      for (EdgeCtrl *cb : ctrls)
        if (cb && cb->dirty())
          co_return;
      co_yield accum;
      co_return;
    }
    auto [epat, direction] = req[i];
    EdgeIt it = direction == MatchDirection::Outgoing
                    ? node->outgoing().begin()
                    : node->incoming().begin();
    const EdgeIt end = direction == MatchDirection::Outgoing
                           ? node->outgoing().end()
                           : node->incoming().end();
    while (it != end) {
      if (!accum.valid())
        co_return;
      EdgeIt curr = it;
      EdgeCtrl cb(node, curr, direction, &context);
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
      if (alreadyUsed || !epat->template mutable_predicate<Allocator>(edgeRef))
        continue;
      Match next = accum;
      next.registerMatch(epat, EdgeMatchWrap{&cb});
      // Pin constrained endpoints before yielding. Never retain source-list
      // iterators across mutations that may erase the edge.
      std::vector<std::pair<NodePatternHandle<V, E, W>, NodeHandle>> endpoints;
      if (auto dstPat = epat->getDst(); dstPat != nullptr) {
        endpoints.emplace_back(dstPat, edgeRef.dst());
      }
      const auto sources = epat->getSrcs();
      auto src = edgeRef.srcs().begin();
      const auto srcEnd = edgeRef.srcs().end();
      bool missingSource = false;
      for (auto sourcePattern : sources) {
        if (sourcePattern != nullptr) {
          if (src == srcEnd) {
            missingSource = true;
            break;
          }
          endpoints.emplace_back(sourcePattern, NodeHandle(*src));
        }
        if (src != srcEnd)
          ++src;
      }
      if (missingSource)
        continue;

      auto matchEndpoints =
          [&](auto &recurse, size_t endpoint,
              const Match &partial) -> memory::generator<Match> {
        if (!partial.valid())
          co_return;
        if (endpoint == endpoints.size()) {
          for (const auto &result : self(self, i + 1, partial)) {
            co_yield result;
            if (!partial.valid())
              co_return;
          }
          co_return;
        }
        const auto &[pattern, child] = endpoints[endpoint];
        for (const auto &childMatch :
             mutable_match_all<V, E, W, Allocator>(pattern, child, context)) {
          Match merged = partial;
          if (!merged.mergeMatches(childMatch))
            continue;
          for (const auto &result : recurse(recurse, endpoint + 1, merged)) {
            co_yield result;
            if (!merged.valid())
              break;
          }
          if (!partial.valid())
            co_return;
        }
      };
      usedEdges.push_back(edgePtr);
      ctrls.push_back(&cb);
      for (const auto &result : matchEndpoints(matchEndpoints, 0, next))
        co_yield result;
      ctrls.pop_back();
      usedEdges.pop_back();
      // Mutations during the yield may insert a successor even when the
      // pre-yield cursor was end(). erase() records the updated successor.
      it = cb.nextIterator();
    }
  };
  for (const auto &m : solve(solve, 0, base))
    co_yield m;
}

} // namespace pattern_matching::details

// Candidate roots are visited once through outgoing graph reachability.
// Constraints may follow incoming/outgoing edges and ordered
// sources/destination. Requirements are enumerated incoming first, then
// outgoing, in pattern order. Matches and edge controls are valid only during
// the current yield. Erase matched edges through EdgeMatch::erase(); active
// assignments using an erased edge are abandoned. Insertion at
// nextOutgoingIterator() before erasure is supported (including repeated
// SliceSlice fusion in one traversal). Do not create nodes, mutate payloads, or
// erase edges outside EdgeMatch while iterating. Newly inserted edges behind
// the continuation are not revisited.
template <typename V, typename E, typename W = memory::NullWeight,
          typename Allocator = memory::mallocator>
memory::generator<LinkedGraphMatch<V, E, W, Allocator>>
match_all(const GraphPattern<V, E, W> &pattern,
          const typename memory::LinkedGraph<V, E, W, Allocator>::NodeHandle
              &rootNode) {
  using LinkedGraph = memory::LinkedGraph<V, E, W, Allocator>;
  using NodeHandle = typename LinkedGraph::NodeHandle;
  if (std::holds_alternative<std::monostate>(pattern.root()))
    co_return;
  if (std::holds_alternative<EdgePatternHandle<V, E, W>>(pattern.root()))
    throw std::runtime_error(
        "Matching a LinkedGraph with an edge root is currently not supported!");
  NodePatternHandle<V, E, W> rootpattern =
      std::get<NodePatternHandle<V, E, W>>(pattern.root());
  memory::vector<NodeHandle> stack;
  stack.reserve(rootNode.upperNodeCount());
  stack.push_back(rootNode);
  memory::dynamic_bitset visited(rootNode.upperNodeCount() + 1);
  typename pattern_matching::details::EdgeMatchControl<
      V, E, W, Allocator>::Context context;
  while (!stack.empty()) {
    NodeHandle node = stack.back();
    stack.pop_back();
    memory::NodeId nid = node->id();
    if (visited[*nid])
      continue;
    visited[*nid] = true;
    for (const auto &match :
         pattern_matching::details::mutable_match_all<V, E, W, Allocator>(
             rootpattern, node, context))
      co_yield match;
    for (const auto &e : node->outgoing())
      stack.push_back(NodeHandle(e.dst()));
  }
}

} // namespace denox::algorithm

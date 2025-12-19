#pragma once

#include "denox/memory/hypergraph/ConstGraph.hpp"
#include <type_traits>

namespace denox::algorithm {

template <typename V, typename E, typename W>
  requires(std::is_arithmetic_v<W>)
memory::optional<memory::vector<memory::EdgeId>>
shortest_dag_hyperpath(const memory::ConstGraph<V, E, W> &graph,
                       memory::span<const memory::NodeId> starts,
                       memory::span<const memory::NodeId> ends) {
  const std::size_t Vn = graph.nodeCount();
  const std::size_t En = graph.edgeCount();

  if (ends.size() == 0) {
    memory::vector<memory::EdgeId> path;
    return path;
  }

  if (Vn == 0 || En == 0 || ends.size() == 0)
    return {};

  // Distances and best incoming edge per node
  memory::vector<memory::optional<W>> dist(Vn, memory::optional<W>{});
  memory::vector<memory::optional<memory::EdgeId>> best_in(
      Vn, memory::optional<memory::EdgeId>{});

  // Kahn-like topological sweep over tails
  memory::vector<std::uint32_t> indeg_by_tail(Vn, 0);
  memory::vector<std::uint16_t> rem(En, 0);
  memory::vector<memory::optional<W>> partial(En, W(0));

  for (std::size_t vi = 0; vi < Vn; ++vi) {
    memory::NodeId v{vi};
    for (memory::EdgeId e : graph.incoming(v)) {
      memory::span<const memory::NodeId> tails = graph.src(e);
      std::uint16_t k = static_cast<std::uint16_t>(tails.size());
      rem[*e] = k;
      indeg_by_tail[*v] += k;
    }
  }

  for (memory::NodeId s : starts)
    dist[*s] = W(0);

  memory::vector<memory::NodeId> q;
  q.reserve(Vn);
  for (std::size_t vi = 0; vi < Vn; ++vi) {
    if (indeg_by_tail[vi] == 0)
      q.push_back(memory::NodeId{vi});
  }

  std::size_t qh = 0;
  while (qh < q.size()) {
    memory::NodeId u = q[qh++];

    for (memory::EdgeId e : graph.outgoing(u)) {
      if (!partial[*e] || !dist[*u]) {
        partial[*e] = memory::optional<W>{};
      } else {
        partial[*e] = W(*partial[*e] + *dist[*u]);
      }
      std::uint16_t r = --rem[*e];
      memory::NodeId v = graph.dst(e);
      if (r == 0 && partial[*e]) {
        W cand = W(*partial[*e] + graph.weight(e));
        if (!dist[*v] || cand < *dist[*v]) {
          dist[*v] = cand;
          best_in[*v] = e;
        }
      }
      if (--indeg_by_tail[*v] == 0)
        q.push_back(v);
    }
  }

  // Require all requested ends to be reachable (synthetic sink semantics)
  for (memory::NodeId t : ends) {
    if (!dist[*t])
      return memory::nullopt;
  }

  // Reconstruct union of shortest paths to all ends
  memory::vector<unsigned char> emitted(En, 0);
  memory::vector<memory::EdgeId> order;
  order.reserve(En);

  struct Frame {
    memory::NodeId v;
    unsigned char stage;
  };
  memory::vector<Frame> stack;
  stack.reserve(ends.size() * 2);
  for (memory::NodeId t : ends)
    stack.push_back(Frame{t, 0});

  while (!stack.empty()) {
    Frame fr = stack.back();
    stack.pop_back();

    memory::NodeId v = fr.v;
    if (!best_in[*v])
      continue;

    memory::EdgeId e = *best_in[*v];
    if (fr.stage == 0) {
      if (emitted[*e])
        continue;
      stack.push_back(Frame{v, 1});
      memory::span<const memory::NodeId> tails = graph.src(e);
      for (std::size_t i = 0; i < tails.size(); ++i) {
        stack.push_back(Frame{tails[i], 0});
      }
    } else {
      if (!emitted[*e]) {
        emitted[*e] = 1;
        order.push_back(e);
      }
    }
  }

  return memory::optional<memory::vector<memory::EdgeId>>(std::move(order));
}

} // namespace denox::algorithm

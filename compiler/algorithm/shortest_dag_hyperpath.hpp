#pragma once

#include "memory/hypergraph/ConstGraph.hpp"
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
  if (Vn == 0 || En == 0 || ends.size() == 0)
    return {};

  memory::vector<memory::optional<W>> dist(Vn, memory::optional<W>{});
  memory::vector<memory::optional<memory::EdgeId>> best_in(
      Vn, memory::optional<memory::EdgeId>{});

  memory::vector<std::uint32_t> indeg_by_tail(Vn, 0);
  memory::vector<std::uint16_t> rem(En, 0);
  memory::vector<memory::optional<W>> partial(En, W(0));

  for (std::size_t vi = 0; vi < Vn; ++vi) {
    const auto v = memory::NodeId{vi};
    for (auto e : graph.incoming(v)) {
      const auto tails = graph.src(e);
      const auto k = static_cast<std::uint16_t>(tails.size());
      rem[e] = k;
      indeg_by_tail[v] += k;
    }
  }

  for (auto s : starts) {
    dist[s] = W(0);
  }

  memory::vector<memory::NodeId> q;
  q.reserve(Vn);
  for (std::size_t vi = 0; vi < Vn; ++vi) {
    if (indeg_by_tail[vi] == 0) {
      q.push_back(memory::NodeId{vi});
    }
  }

  std::size_t qh = 0;
  while (qh < q.size()) {
    const auto u = q[qh++];

    for (auto e : graph.outgoing(u)) {
      if (!partial[e] || !dist[u]) {
        partial[e] = memory::optional<W>{};
      } else {
        partial[e] = W(*partial[e] + *dist[u]);
      }
      const auto r = --rem[e];
      const auto v = graph.dst(e);
      if (r == 0 && partial[e]) {
        const W cand = W(*partial[e] + graph.weight(e));
        if (!dist[v] || cand < *dist[v]) {
          dist[v] = cand;
          best_in[v] = e;
        }
      }
      if (--indeg_by_tail[v] == 0) {
        q.push_back(v);
      }
    }
  }

  memory::optional<memory::NodeId> best_t;
  memory::optional<W> best_cost;
  for (auto t : ends) {
    if (dist[t] && (!best_cost || *dist[t] < *best_cost)) {
      best_cost = dist[t];
      best_t = t;
    }
  }
  if (!best_t) {
    return memory::nullopt; // no reachable target
  }

  memory::vector<unsigned char> emitted(En, 0);
  memory::vector<memory::EdgeId> order;
  order.reserve(En);

  struct Frame {
    memory::NodeId v;
    unsigned char stage;
  };
  memory::vector<Frame> stack;
  stack.push_back(Frame{*best_t, 0});

  while (!stack.empty()) {
    const Frame fr = stack.back();
    stack.pop_back();

    const auto v = fr.v;
    if (!best_in[v]) {
      continue;
    }

    const auto e = *best_in[v];
    if (fr.stage == 0) {
      if (emitted[e])
        continue;
      stack.push_back(Frame{v, 1});
      for (auto u : graph.src(e)) {
        stack.push_back(Frame{u, 0});
      }
    } else {
      if (!emitted[e]) {
        emitted[e] = 1;
        order.push_back(e);
      }
    }
  }
  return memory::optional<memory::vector<memory::EdgeId>>(std::move(order));
}

} // namespace denox::algorithm

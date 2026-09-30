#pragma once

#include "denox/algorithm/countr_zero.hpp"
#include "denox/algorithm/popcount.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/memory/container/small_dynamic_bitset.hpp"
#include "denox/memory/hypergraph/AdjGraph.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <tracy/Tracy.hpp>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace denox::algorithm {

template <typename V, typename E, typename W>
memory::AdjGraph<V, E, W>
minimum_cost_subgraph(const memory::ConstGraph<V, E, W> &graph,
                      memory::span<const memory::NodeId> inputs,
                      memory::span<const memory::NodeId> outputs,
                      size_t max_states = std::numeric_limits<size_t>::max(),
                      bool *truncated = nullptr) {
  ZoneScopedN("algorithm::minimum_cost_subgraph2");
  if (truncated) {
    *truncated = false;
  }

  const size_t n = graph.nodeCount();
  memory::small_dynamic_bitset<1> isInput(n, false);
  memory::small_dynamic_bitset<1> isOutput(n, false);
  for (auto v : inputs) {
    isInput.set(*v);
  }
  for (auto v : outputs) {
    isOutput.set(*v);
  }

  // Unique dependency vertices, independent of parallel implementation edges.
  // Inputs need no producer, so their incoming edges impose no DP constraints.
  std::vector<std::vector<size_t>> dependencies(n);
  std::vector<size_t> consumers(n, 0);
  for (size_t v = 0; v < n; ++v) {
    if (isInput[v]) {
      continue;
    }
    auto &sources = dependencies[v];
    for (auto e : graph.incoming(memory::NodeId{v})) {
      for (auto u : graph.src(e)) {
        if (!isInput[*u]) {
          sources.push_back(*u);
        }
      }
    }
    std::sort(sources.begin(), sources.end());
    sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
    for (auto u : sources) {
      ++consumers[u];
    }
  }

  std::vector<size_t> ready;
  for (size_t v = 0; v < n; ++v) {
    if (consumers[v] == 0) {
      ready.push_back(v);
    }
  }
  memory::small_dynamic_bitset<1> boundary(n, false);

  std::vector<memory::NodeId> order;
  size_t peak_frontier = 0;
  {
    // Modified Kahn topological sort, which additionally
    // greedily minimized peek frontier size.
    // time complexity: O(NM)
    order.reserve(n);
    size_t frontier_size = 0;
    while (!ready.empty()) {
      auto score = [&](size_t v) {
        // change in frontier size. (primary sorting criterion)
        ptrdiff_t delta = boundary[v] ? -1 : 0;
        ptrdiff_t released = 0;
        for (auto u : dependencies[v]) {
          if (!boundary[u]) {
            ++delta;
          }
          if (consumers[u] == 1) {
            ++released;
          }
        }
        return std::tuple{delta, -released, v};
      };

      // NOTE: could use bucket or radix heap, as keys are integer, but probably
      // doesn't actually matter.
      auto best =
          std::min_element(ready.begin(), ready.end(), [&](size_t a, size_t b) {
            return score(a) < score(b);
          });

      const size_t v = *best;
      ready.erase(best);
      order.emplace_back(v);
      if (boundary[v]) {
        boundary.reset(v);
        --frontier_size;
      }
      for (auto u : dependencies[v]) {
        if (!boundary[u]) {
          boundary.set(u);
          frontier_size++;
        }
        if (--consumers[u] == 0) {
          ready.push_back(u);
        }
      }
      peak_frontier = std::max(peak_frontier, frontier_size);
    }
  }

  using frontier_t = memory::uint128;

  if (order.size() != n) {
    diag::invalid_state("minimum_cost_subgraph: graph contains a cycle");
  }
  if (peak_frontier > sizeof(frontier_t) * 8) {
    // NOTE: If this fails before throwing the implementation away consider
    // replacing frontier_t with larger bitset like uint128_t, for some
    // networks, large frontier may be acceptable, but it's unlikely.
    diag::invalid_state(
        fmt::format("minimum_cost_subgraph: frontier exceeds {} slots",
                    sizeof(frontier_t) * 8));
  }

  // h[v] is a lower bound on the cost of deriving v from the inputs.
  std::vector<std::optional<W>> h(n);
  std::vector<memory::EdgeId> feasible_producer(n);
  std::vector<std::optional<W>> remaining_output_cost(order.size() + 1, W{});
  {
    // Precompute remaining output cost as
    // remaining_output_cost[i] = max(h[v] for required outputs v in order[i ...
    // end])
    for (auto v : inputs) {
      h[*v] = W{};
    }
    // The consumer-first processing order reversed is dependency-first.
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      const auto v = *it;
      if (isInput[*v]) {
        continue;
      }
      for (auto edge : graph.incoming(v)) {
        W source_cost{};
        bool reachable = true;
        for (auto u : graph.src(edge)) {
          if (!h[*u]) {
            reachable = false;
            break;
          }
          source_cost = std::max(source_cost, *h[*u]);
        }
        if (!reachable) {
          continue;
        }
        const W candidate = graph.weight(edge) + source_cost;
        if (!h[*v] || candidate < *h[*v]) {
          h[*v] = candidate;
          feasible_producer[*v] = edge;
        }
      }
    }
    for (size_t i = order.size(); i-- > 0;) {
      remaining_output_cost[i] = remaining_output_cost[i + 1];
      const auto v = order[i];
      if (!isOutput[*v]) {
        continue;
      }
      if (!h[*v] || !remaining_output_cost[i]) {
        remaining_output_cost[i] = std::nullopt;
      } else {
        remaining_output_cost[i] = std::max(*remaining_output_cost[i], *h[*v]);
      }
    }
  }

  memory::AdjGraph<V, E, W> result;
  for (size_t i = 0; i < n; ++i) {
    result.addNode(graph.get(memory::NodeId{i}));
  }

  if (!remaining_output_cost[0]) {
    return result;
  }

  // Slots live from first processed consumer through the tensor's own step.
  // Clear the retiring tensor's bit before adding sources that reuse its slot.
  const size_t no_slot = static_cast<size_t>(-1);
  std::vector<size_t> slots(n, no_slot), free_slots;
  size_t slot_count = 0;
  for (auto v : order) {
    if (slots[*v] != no_slot) {
      free_slots.push_back(slots[*v]);
    }
    for (auto u : dependencies[*v]) {
      if (slots[u] != no_slot) {
        continue;
      }
      if (free_slots.empty()) {
        slots[u] = slot_count++;
      } else {
        slots[u] = free_slots.back();
        free_slots.pop_back();
      }
    }
  }
  assert(slot_count == peak_frontier);

  struct Transition {
    size_t predecessor;
    memory::EdgeId edge; // Invalid for a skipped vertex.
  };

  struct State {
    W cost;
    Transition predecessor;
  };
  // Frontier keys are only needed for the current/next layer. Older layers
  // retain costs and transition indices, not copies of their frontier sets.
  using Index = std::unordered_map<frontier_t, size_t>;

  Index current;

  current.emplace(0, 0);
  std::vector<std::vector<State>> layers(1);
  layers.back().push_back(State{W{}, {0, memory::EdgeId{}}});

  frontier_t feasible_frontier = 0;
  std::vector<std::optional<W>> slot_cost(peak_frontier);
  size_t layer_index = 0;

  // DP: in step n, consider all possible frontiers
  // after processing the first n verticies from order.
  // Time complexity: O(2^w), where w is the frontier width.
  // NOTE: This can absolutely blow up!
  // Worst case is a graph like this:
  //                     O
  //                     ^
  //                     |
  //               {C1, C2, C3}    <--- rank-3 hyperedge
  //                /   |   \
  //               /    |    \
  //             C1    C2    C3
  //            /  \   /  \   /  \
  //           A1  B1 A2  B2 A3  B3
  //
  for (memory::NodeId v : order) {
    // Track slot ownership at the *next* boundary. A source can reuse v's slot.
    if (slots[*v] != no_slot) {
      slot_cost[slots[*v]].reset();
    }
    for (auto u : dependencies[*v]) {
      slot_cost[slots[u]] = h[u];
    }
    // Precompute source masks once per layer, rather than per transition.
    std::vector<frontier_t> source_masks;
    if (!isInput[*v]) {
      source_masks.reserve(graph.incoming(v).size());
      for (auto edge : graph.incoming(v)) {
        frontier_t mask = 0;
        for (auto u : graph.src(edge)) {
          if (!isInput[*u]) {
            mask |= (1ull << slots[*u]);
          }
        }
        source_masks.push_back(std::move(mask));
      }
    }
    // Follow one fixed feasible derivation, reserving its mask in every beam.
    // A cheaper prefix reaching that same mask is an equally valid fallback.
    const bool fallback_required =
        slots[*v] != no_slot &&
        (feasible_frontier & (frontier_t{1} << slots[*v]));
    if (!isInput[*v] && (fallback_required || isOutput[*v])) {
      if (fallback_required) {
        feasible_frontier &= ~(frontier_t{1} << slots[*v]);
      }
      assert(feasible_producer[*v]);
      for (auto u : graph.src(feasible_producer[*v])) {
        if (!isInput[*u]) {
          feasible_frontier |= frontier_t{1} << slots[*u];
        }
      }
    }
    Index next;
    std::vector<State> states;
    // The protected mask sorts first. Others rank by g + lower bound, then
    // fewer requirements and mask value. Streaming eviction keeps storage <= K.
    using Rank = std::tuple<bool, W, unsigned, frontier_t>;
    std::set<Rank> ranked;
    bool beam_active = false;
    auto rank = [&](frontier_t mask, W cost) {
      W bound = *remaining_output_cost[layer_index + 1];
      for (auto bits = mask; bits; bits &= bits - 1) {
        const auto &value = slot_cost[algorithm::countr_zero(bits)];
        assert(value);
        bound = std::max(bound, *value);
      }
      return Rank{mask != feasible_frontier, cost + bound,
                  static_cast<unsigned>(algorithm::popcount(mask)), mask};
    };
    auto emit = [&](frontier_t frontier, W cost, size_t predecessor,
                    memory::EdgeId edge) {
      auto it = next.find(frontier);
      if (it != next.end()) {
        auto &state = states[it->second];
        if (cost < state.cost) {
          if (beam_active) {
            ranked.erase(rank(frontier, state.cost));
            ranked.insert(rank(frontier, cost));
          }
          state.cost = cost;
          state.predecessor = {predecessor, edge};
        }
        return;
      }
      size_t index = states.size();
      if (max_states && next.size() == max_states) {
        // Exact layers pay no ranking cost until the first actual overflow.
        if (!beam_active) {
          for (const auto &[mask, slot] : next) {
            ranked.insert(rank(mask, states[slot].cost));
          }
          beam_active = true;
        }
        // Discarding any distinct state conservatively loses the exactness
        // guarantee.
        if (truncated) {
          *truncated = true;
        }
        auto worst = std::prev(ranked.end());
        if (!(rank(frontier, cost) < *worst)) {
          return;
        }
        const auto discarded = std::get<3>(*worst);
        index = next.at(discarded);
        next.erase(discarded);
        ranked.erase(worst);
        states[index] = State{cost, {predecessor, edge}};
      } else {
        states.push_back(State{cost, {predecessor, edge}});
      }
      next.emplace(frontier, index);
      if (beam_active) {
        ranked.insert(rank(frontier, cost));
      }
    };
    // Stable enumeration makes approximate results independent of hash-table
    // iteration order. State indices are already dense and deterministic.
    std::vector<frontier_t> masks(layers.back().size());
    for (const auto &[mask, index] : current) {
      masks[index] = mask;
    }
    for (size_t index = 0; index < masks.size(); ++index) {
      const auto frontier = masks[index];
      const bool required = [&]() {
        return slots[*v] != no_slot && ((frontier & (1ull << slots[*v])) != 0);
      }();
      const W cost = layers.back()[index].cost;
      if (isInput[*v] || (!required && !isOutput[*v])) {
        assert(!required);
        emit(frontier, cost, index, memory::EdgeId{});
        continue;
      }
      frontier_t remaining = frontier;
      if (required) {
        remaining &= ~(1ull << slots[*v]);
      }
      size_t mask_index = 0;
      for (auto edge : graph.incoming(v)) {
        const size_t this_mask = mask_index++;
        if (std::any_of(graph.src(edge).begin(), graph.src(edge).end(),
                        [&](auto u) { return !h[*u]; }))
          continue;
        assert(!(graph.weight(edge) < W{}));
        frontier_t dependencies = remaining;
        dependencies |= source_masks[this_mask];
        emit(std::move(dependencies), cost + graph.weight(edge), index, edge);
      }
    }
    if (next.empty()) {
      return result; // At least one output is not derivable.
    }
    layers.push_back(std::move(states));
    current = std::move(next);
    ++layer_index;
  }

  const auto terminal = current.find(0);
  if (terminal == current.end()) {
    return result;
  }
  memory::small_dynamic_bitset<1> selected(graph.edgeCount(), false);

  size_t index = terminal->second;
  for (size_t layer = layers.size() - 1; layer > 0; --layer) {
    const auto &transition = layers[layer][index].predecessor;
    if (transition.edge) {
      selected.set(*transition.edge);
    }
    index = transition.predecessor;
  }
  for (size_t i = 0; i < selected.size(); ++i) {
    if (!selected[i]) {
      continue;
    }
    memory::EdgeId edge{i};
    result.addEdge(graph.src(edge), graph.dst(edge), graph.get(edge),
                   graph.weight(edge));
  }
  return result;
}

} // namespace denox::algorithm

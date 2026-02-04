#pragma once

#include "denox/algorithm/hash_combine.hpp"
#include "denox/algorithm/topological_sort.hpp"
#include "denox/memory/container/dynamic_bitset.hpp"
#include "denox/memory/container/hashmap.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/small_dynamic_bitset.hpp"
#include "denox/memory/container/small_vector.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/hypergraph/AdjGraph.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include "denox/memory/hypergraph/NodeId.hpp"
#include <algorithm>
#include <unordered_set>

namespace denox::algorithm {

template <typename V, typename E, typename W>
memory::AdjGraph<V, E, W>
minimum_cost_subgraphs(const memory::ConstGraph<V, E, W> &graph,
                       memory::span<const memory::NodeId> inputs,
                       memory::span<const memory::NodeId> outputs,
                       const W eps = {}) {
  // SVO optimization tuning parameters generally don't matter at all.
  static constexpr size_t SMALL_N = 512;
  static constexpr size_t SMALL_FRONTIER = 4;
  static constexpr size_t SMALL_INDEG = 8;
  static constexpr size_t SMALL_SRC = 2;

  const size_t N = graph.nodeCount();
  const size_t M = graph.edgeCount();

  memory::vector<memory::NodeId> topologicalOrder =
      algorithm::topologicalSort(graph);

  memory::dynamic_bitset isInput(N, false);
  for (memory::NodeId nid : inputs) {
    isInput[*nid] = true;
  }

  // ======= PREPROCESSING ===========

  // 1. Precompute min derive cost heuristic:
  //    h(v) = min{w(e) + max{h(u) | u \in src(e)} | e \in incomming(v)}
  //  NOTE: h(v) is a lower bound of the optimal cost to derive v.
  memory::vector<memory::optional<W>> h(N, memory::nullopt);
  {
    for (memory::NodeId v : topologicalOrder) {
      if (isInput[*v]) {
        h[*v] = W{};
        continue;
      }
      const auto incoming = graph.incoming(v);
      memory::optional<W> best = memory::nullopt;

      for (memory::EdgeId e : incoming) {
        memory::optional<W> max_src = memory::nullopt;
        bool ok = true;
        for (memory::NodeId u : graph.src(e)) {
          if (!h[*u]) {
            ok = false;
            break;
          }
          if (!max_src || *h[*u] > *max_src) {
            max_src = *h[*u];
          }
        }
        if (ok && max_src) {
          W candidate = graph.weight(e) + *max_src;
          if (!best || candidate < *best) {
            best = candidate;
          }
        }
      }
      h[*v] = best;
    }
  }
  // 2. Precompute edge cost heuristic:
  //   similarly we compute the edge cost heuristic based on the min cost
  //   of all of it's sources.
  memory::vector<memory::optional<W>> he(M, memory::nullopt);
  {
    for (uint64_t i = 0; i < M; ++i) {
      memory::EdgeId eid{i};
      memory::optional<W> max_src = memory::nullopt;
      bool ok = true;
      for (memory::NodeId u : graph.src(eid)) {
        if (!h[*u]) {
          ok = false;
          break;
        }
        if (!max_src || *h[*u] > *max_src) {
          max_src = *h[*u];
        }
      }
      if (ok && max_src) {
        he[i] = graph.weight(eid) + *max_src;
      }
    }
  }
  // 3. Precompute ancestors bitsets:
  //   For all nodes that are ancestors, we simple set a bit.
  memory::vector<memory::small_dynamic_bitset<SMALL_N>> ancestors(
      N, memory::small_dynamic_bitset<SMALL_N>{N, false});
  {
    // DP in topo order
    for (memory::NodeId v : topologicalOrder) {
      auto incoming = graph.incoming(v);
      if (incoming.empty()) {
        continue;
      }
      for (memory::EdgeId e : incoming) {
        for (memory::NodeId u : graph.src(e)) {
          ancestors[*v].set(*u, true);
          ancestors[*v] |= ancestors[*u];
        }
      }
      assert(ancestors[*v][*v] == false &&
             "Cycle?? graph has to be a Hyper-DAG!");
    }
  }

  // 4. Memoization Cache.
  //   Keyed on dfs state, so done and open, but we additionally
  //   made the observation that we only have to consider
  //   ancestors of open in the done key.
  //   That's why we simply bitand the done and
  //   ancestors of the done set together to get our key,
  //   improved memo hitrates by at least a factor of 2x.
  struct MemoKey {
    memory::small_dynamic_bitset<SMALL_N> done;
    memory::small_vector<memory::NodeId, SMALL_FRONTIER> open;
  };
  struct MemoHash {
    size_t operator()(const MemoKey &key) const {
      assert(!key.open.empty());
      uint64_t hash = *key.open.front();
      for (size_t i = 1; i < key.open.size(); ++i) {
        hash = algorithm::hash_combine(hash, *key.open[i]);
      }
      hash = algorithm::hash_combine(0xE6C6DAE8F56DA2A9, hash);
      for (size_t i = 0; i < key.done.word_count(); ++i) {
        hash = algorithm::hash_combine(hash, key.done.words()[i]);
      }
      return hash;
    }
  };
  struct MemoComp {
    bool operator()(const MemoKey &lhs, const MemoKey &rhs) const {
      return std::ranges::equal(lhs.open, rhs.open) && lhs.done == rhs.done;
    }
  };
  memory::hash_map<MemoKey, memory::optional<W>, MemoHash, MemoComp> memo;

  // Done set: Nodes that have already been computed,
  //   including them again in the partial subgraph is free.
  //   Only required because we have hyperedges!
  memory::small_dynamic_bitset<SMALL_N> done(N, false);
  for (memory::NodeId nid : inputs) {
    done.set(*nid, true);
  }

  // Open set: Nodes that are required to be included within the
  //    current subtree, because previously picked nodes depend on them
  //    beeing available!
  memory::small_vector<memory::NodeId, SMALL_FRONTIER> open(outputs.begin(),
                                                            outputs.end());
  // We maintain the invariants that open is a sorted unique set!
  std::sort(open.begin(), open.end());
  open.erase(std::unique(open.begin(), open.end()), open.end());
  // Inputs don't belong here, because they are always available.
  open.erase(std::remove_if(open.begin(), open.end(),
                            [&](memory::NodeId v) { return isInput[*v]; }),
             open.end());

  // ============ Minimum-Cost-Subgraph (Actual algorithm) ==============

  // NOTE: Basically the recursive_solve essentially just iterates over
  //   all possible valid subgraphs which connect inputs to outputs.
  //   That alone is a hard NP problem so we only consider partial subgraphs
  //   further if the current cost is less than the best cost found so far.
  //   Because we pick the traversal order based on a very tight heuristic we
  //   often find the best solution very quickly and can therefor prune a lot of
  //   the search space.
  //   Additionally we do some memoization over the solver state, which
  //   makes the problem solvable even for large subproblems, but for the
  //   usecases that we have right now it's probably overkill.
  //
  const auto h_comp = [&](memory::NodeId lhs, memory::NodeId rhs) {
    auto const &a = h[*lhs];
    auto const &b = h[*rhs];

    if (a && b) {
      if (*a < *b)
        return true;
      if (*b < *a)
        return false;
      return *lhs < *rhs;
    }
    if (a && !b)
      return true;
    if (!a && b)
      return false;
    return *lhs < *rhs;
  };

  const auto he_comp = [&](const memory::EdgeId &lhs,
                           const memory::EdgeId &rhs) {
    const auto &a = he[*lhs];
    const auto &b = he[*rhs];

    if (a && b) {
      if (*a < *b)
        return true;
      if (*b < *a)
        return false;
      return *lhs < *rhs;
    }
    if (a && !b)
      return true;
    if (!a && b)
      return false;
    return *lhs < *rhs;
  };

  auto recusive_solve = [&](auto &&self) -> memory::optional<W> {
    // Termination condition: if open is empty
    // all upstream dependencies have been resolved and we now that
    // we have found a valid subgraph!
    if (open.empty()) {
      return W{};
    }

    // Memoization!
    memory::small_dynamic_bitset<SMALL_N> openAnc(N);
    for (memory::NodeId nid : open) {
      openAnc |= ancestors[*nid];
    }
    MemoKey memoKey{
        .done = done & openAnc, // <- only consider ancestors of the open set.
        .open = open,
    };
    // Memo lookup
    if (auto it = memo.find(memoKey); it != memo.end()) {
      return it->second;
    }

    memory::NodeId v;
    { // Pick next upstream!
      // NOTE: We want to pick fast implementations first,
      //  the h heurstic is very close to the true optimum.
      //  So in 99% of cases, we only ever pick a single upstream
      //  and all other upstreams are pruned.
      const auto it =
          std::ranges::min_element(open.begin(), open.end(), h_comp);
      assert(it != open.end());
      v = *it;
    }

    if (!h[*v]) {
      memo.emplace(std::move(memoKey), memory::nullopt);
      return memory::nullopt;
    }

    assert(!done[*v] && "done may never contain node, which is in open!");

    if (graph.incoming(v).empty()) {
      memo.emplace(std::move(memoKey), memory::nullopt);
      return memory::nullopt;
    }

    // NOTE: Instead of copying done and open around we do some bookkeeping
    //   before the recursion and afterwards recover our old state, this saves
    //   on a lot of allocations improved perf by around a factor of 2x.
    done.set(*v, true);
    // Remove v from open once for all branches (later added back in)
    auto itv = std::lower_bound(open.begin(), open.end(), v);
    assert(itv != open.end() && *itv == v);
    const std::size_t v_pos = static_cast<std::size_t>(itv - open.begin());
    open.erase(itv);

    memory::small_vector<memory::EdgeId, SMALL_INDEG> inc(
        graph.incoming(v).begin(), graph.incoming(v).end());
    assert(std::ranges::count(inputs, v) == 0);
    std::sort(inc.begin(), inc.end(), he_comp);

    memory::optional<W> best;

    for (memory::EdgeId e : inc) {
      bool edge_ok = true;
      for (memory::NodeId u : graph.src(e)) {
        if (!h[*u]) {
          edge_ok = false;
          break;
        }
      }
      if (!edge_ok) {
        continue;
      }

      // Record inserted positions for this edge choice (to rollback)
      memory::small_vector<std::size_t, SMALL_SRC> inserted_positions;

      for (memory::NodeId u : graph.src(e)) {
        if (done[*u]) {
          continue;
        }

        // sorted insert (only if unique)
        auto ins = std::lower_bound(open.begin(), open.end(), u);
        if (ins == open.end() || *ins != u) {
          const std::size_t pos = static_cast<std::size_t>(ins - open.begin());
          open.insert(ins, u);
          inserted_positions.push_back(pos);
        }
      }

      memory::optional<W> w = self(self);
      if (w) {
        W cand = *w + graph.weight(e);
        if (!best || cand < *best) {
          best = cand;
        }
      }

      // rollback inserts in reverse order (positions remain valid)
      for (std::size_t i = inserted_positions.size(); i-- > 0;) {
        const std::size_t pos = inserted_positions[i];
        open.erase(open.begin() + static_cast<std::ptrdiff_t>(pos));
      }
    }
    // Restore v into open at original position
    open.insert(open.begin() + static_cast<std::ptrdiff_t>(v_pos), v);
    done.set(*v, false);
    memo.emplace(std::move(memoKey), best);
    return best;
  };

  memory::AdjGraph<V, E, W> subgraph;
  for (uint64_t i = 0; i < N; ++i) {
    memory::NodeId nid{i};
    memory::NodeId _nid = subgraph.addNode(graph.get(nid));
    assert(_nid == nid);
  }

  // Where the fun stuff is happening, the actual NP algorithm!
  // NOTE: recursion depth is max the size of the minimum-cost-subgraph,
  //   so generally we should be safe from StackOverflows here.
  memory::optional<W> root_cost = recusive_solve(recusive_solve);
  if (!root_cost) {
    // There exist no subgraph which contains inputs and output!
    // Return empty graph!
    return subgraph;
  }

  done = memory::small_dynamic_bitset<SMALL_N>(N, false);
  for (memory::NodeId nid : inputs) {
    done.set(*nid, true);
  }

  open.clear();
  open.assign(outputs.begin(), outputs.end());
  // We maintain the invariants that open is a sorted unique set!
  std::sort(open.begin(), open.end(),
            [](memory::NodeId a, memory::NodeId b) { return *a < *b; });
  open.erase(
      std::unique(open.begin(), open.end(),
                  [](memory::NodeId a, memory::NodeId b) { return *a == *b; }),
      open.end());
  // Inputs don't belong here, because they are always available.
  open.erase(std::remove_if(open.begin(), open.end(),
                            [&](memory::NodeId v) {
                              return done[*v]; // <- isInput
                            }),
             open.end());

  memory::dynamic_bitset optimal_edge(M, false);
  std::unordered_set<MemoKey, MemoHash, MemoComp> visited;
  visited.reserve(memo.size());

  auto select_optimal_edges = [&](auto &&self) -> void {
    if (open.empty()) {
      return;
    }
    memory::small_dynamic_bitset<SMALL_N> openAnc(N);
    for (memory::NodeId nid : open) {
      openAnc |= ancestors[*nid];
    }
    assert(!open.empty());
    MemoKey key{.done = done & openAnc, .open = open};

    if (!visited.insert(key).second)
      return;

    // current f(state)
    auto it_cur = memo.find(key);
    if (it_cur == memo.end() || !it_cur->second) {
      // state not optimal or infeasible.
      return;
    }
    const W f_cur = *it_cur->second;

    memory::NodeId v;
    {
      const auto it =
          std::ranges::min_element(open.begin(), open.end(), h_comp);
      assert(it != open.end());
      v = *it;
    }
    if (!h[*v]) {
      return;
    }

    done.set(*v, true);
    auto itv = std::lower_bound(open.begin(), open.end(), v);
    assert(itv != open.end() && *itv == v);
    const std::size_t v_pos = static_cast<std::size_t>(itv - open.begin());
    open.erase(itv);

    memory::small_vector<memory::EdgeId, SMALL_INDEG> inc(
        graph.incoming(v).begin(), graph.incoming(v).end());
    std::sort(inc.begin(), inc.end(), he_comp);

    for (memory::EdgeId e : inc) {

      bool edge_ok = true;
      for (memory::NodeId u : graph.src(e)) {
        if (!h[*u]) {
          edge_ok = false;
          break;
        }
      }
      if (!edge_ok) {
        continue;
      }

      memory::small_vector<std::size_t, SMALL_SRC> inserted_positions;
      // add missing sources to open
      for (memory::NodeId u : graph.src(e)) {
        if (done[*u]) {
          continue;
        }

        auto ins = std::lower_bound(open.begin(), open.end(), u);
        if (ins == open.end() || *ins != u) {
          const std::size_t pos = static_cast<std::size_t>(ins - open.begin());
          open.insert(ins, u);
          inserted_positions.push_back(pos);
        }
      }

      // compute f(next)
      memory::optional<W> f_next_opt;
      if (open.empty()) {
        f_next_opt = W{};
      } else {
        memory::small_dynamic_bitset<SMALL_N> openAnc2(N);
        for (memory::NodeId nid : open) {
          openAnc2 |= ancestors[*nid];
        }
        MemoKey key2{.done = done & openAnc2, .open = open};
        auto it2 = memo.find(key2);
        if (it2 != memo.end()) {
          f_next_opt = it2->second;
        }
      }
      if (f_next_opt) {
        W f_next = *f_next_opt;
        // optimal transition test
        const W lhs = graph.weight(e) + f_next;
        if (lhs - eps <= f_cur && lhs + eps >= f_cur) {
          optimal_edge[*e] = true;
          self(self);
        }
      }

      // rollback
      for (std::size_t i = inserted_positions.size(); i-- > 0;) {
        const std::size_t pos = inserted_positions[i];
        open.erase(open.begin() + static_cast<std::ptrdiff_t>(pos));
      }
    }
    open.insert(open.begin() + static_cast<std::ptrdiff_t>(v_pos), v);
    done.set(*v, false);
  };

  select_optimal_edges(select_optimal_edges);

  // Reconstruct subgraph from bestEdges.
  // NOTE: subgraph contains all nodes of the original graph,
  //   we add all of them simply because otherwise we would have to
  //   remap NodeIds everywhere.

  for (uint32_t i = 0; i < M; ++i) {
    if (optimal_edge[i]) {
      memory::EdgeId eid{i};
      subgraph.addEdge(graph.src(eid), graph.dst(eid), graph.get(eid),
                       graph.weight(eid));
    }
  }

  return subgraph;
}

} // namespace denox::algorithm

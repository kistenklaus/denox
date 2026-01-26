#pragma once

#include "denox/algorithm/hash_combine.hpp"
#include "denox/algorithm/topological_sort.hpp"
#include "denox/memory/container/hashmap.hpp"
#include "denox/memory/container/small_dynamic_bitset.hpp"
#include "denox/memory/container/small_vector.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/hypergraph/AdjGraph.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include "denox/memory/hypergraph/NodeId.hpp"
#include <algorithm>

using namespace std::chrono_literals;

namespace denox::algorithm {

// computes a subgraph, with the minimum total cost, which
// contains all input and output nodes.
//
// NOTE: This problem is generally NP-hard.
//  However there is a fixed-parameter tracable lower bound O(V^B),
//  which is only exponential in the frontier size B,
//  which is the amount of unresolved requirements.
//  To give an example a hyperedge with 2 source nodes splits,
//  creates a new requirement for each source so 2.
//
//  The key insight is that the order of graph traveral
//  (starting from the outputs) effects the frontier size strongly.
//  In the worst case with only 4 skip-connections (i.e. concats) we could end
//  up with B=2^4=16 (or worse with fusion), which is essentially uncomputable,
//  for large graphs. However for U-Nets what works well is to try to first
//  traverse the skip-connections and then remaining source, this way the
//  frontier stays below 2. The general rule is that if we first include cheap
//  to compute nodes based on a heuristic we keep the frontier small. (Only
//  ofcause only holds for simple U-Nets).
//
//  Another advantage of a heuristic is that by traversing good solutions first,
//  we can prune complete subgraphs if their total cost is greater than our
//  current best.
//
//  We additionally memoized the solver states, because although global
//  decisions have to be made independently we often end up in identical states.
//  This memoization is further improved by only considering the remaining
//  subproblem when looking up cached results, which improved hitrates a lot.
//
template <typename V, typename E, typename W>
memory::AdjGraph<V, E, W>
minimum_cost_subgraph(const memory::ConstGraph<V, E, W> &graph,
                      memory::span<const memory::NodeId> inputs,
                      memory::span<const memory::NodeId> outputs) {
  // SVO optimization tuning parameters generally don't matter at all.
  static constexpr size_t SMALL_N = 512;
  static constexpr size_t SMALL_FRONTIER = 4;
  static constexpr size_t SMALL_INDEG = 8;
  static constexpr size_t SMALL_SRC = 2;

  const size_t N = graph.nodeCount();
  const size_t M = graph.edgeCount();

  memory::vector<memory::NodeId> topologicalOrder =
      algorithm::topologicalSort(graph);

  // ======= PREPROCESSING ===========

  // 1. Precompute min derive cost heuristic:
  //    h(v) = min{w(e) + max{h(u) | u \in src(e)} | e \in incomming(v)}
  //  NOTE: h(v) is a lower bound of the optimal cost to derive v.
  memory::vector<W> h(N); // default constructed, but not yet meaningful
  {
    for (memory::NodeId v : topologicalOrder) {
      const auto incoming = graph.incoming(v);
      if (incoming.empty()) {
        h[*v] = W{};
        continue;
      }
      bool first = true;
      W best;
      for (memory::EdgeId e : incoming) {
        W max_src = W{};
        bool first_src = true;
        for (memory::NodeId u : graph.src(e)) {
          if (first_src) {
            max_src = h[*u];
            first_src = false;
          } else {
            max_src = std::max(max_src, h[*u]);
          }
        }
        W candidate = graph.weight(e) + max_src;
        if (first || candidate < best) {
          best = candidate;
          first = false;
        }
      }
      h[*v] = best;
    }
  }
  // 2. Precompute edge cost heuristic:
  //   similarly we compute the edge cost heuristic based on the min cost
  //   of all of it's sources.
  memory::vector<W> he(M);
  {
    for (uint64_t i = 0; i < M; ++i) {
      memory::EdgeId eid{i};
      W max_src = W{};
      bool first_src = true;
      for (memory::NodeId u : graph.src(eid)) {
        if (first_src) {
          max_src = h[*u];
          first_src = false;
        } else {
          max_src = std::max(max_src, h[*u]);
        }
      }
      he[i] = graph.weight(eid) + max_src;
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
  memory::hash_map<MemoKey, W, MemoHash, MemoComp> memo;

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

  std::optional<W> bestCost;
  memory::vector<memory::EdgeId> bestEdges;
  memory::vector<memory::EdgeId> curEdges;
  curEdges.reserve(64);
  auto recusive_solve = [&](auto &&self, W cost) {
    if (bestCost) {
      if (cost >= *bestCost) {
        return;
      }
      // NOTE: Conceptually we could prune based on the heuristic
      // here, because it is a true lower bound, but it already
      // defines our traversal order and would actually never prune anything.
    }

    // Termination condition: if open is empty
    // all upstream dependencies have been resolved and we now that
    // we have found a valid subgraph!
    if (open.empty()) {
      if (!bestCost || cost < *bestCost) {
        bestCost = cost;
        bestEdges.assign(curEdges.begin(), curEdges.end());
      }
      return;
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
    {
      auto it = memo.find(memoKey);
      if (it != memo.end()) {
        const W &memoCost = it->second;
        if (memoCost <= cost) {
          return; // prune, we already found a better edge.
        } else {
          it->second = cost; // update memoized cost and continue.
        }
      } else {
        memo.emplace(std::move(memoKey), cost);
      }
    }
    memory::NodeId v;
    { // Pick next upstream!
      // NOTE: We want to pick fast implementations first,
      //  the h heurstic is very close to the true optimum.
      //  So in 99% of cases, we only ever pick a single upstream
      //  and all other upstreams are pruned.
      const auto it = std::ranges::min_element(
          open.begin(), open.end(),
          [&](const memory::NodeId &lhs, const memory::NodeId &rhs) {
            return h[*lhs] < h[*rhs];
          });
      assert(it != open.end());
      v = *it;
    }

    assert(!done[*v] && "done may never contain node, which is in open!");

    // NOTE: Instead of copying done and open around we do some bookkeeping
    //   before the recursion and afterwards recover our old state, this saves
    //   on a lot of allocations improved perf by around a factor of 2x.
    done.set(*v, true);
    // Remove v from open once for all branches (later added back in)
    auto it = std::lower_bound(
        open.begin(), open.end(), v,
        [](memory::NodeId a, memory::NodeId b) { return *a < *b; });
    assert(it != open.end() && *it == v);
    const std::size_t v_pos = static_cast<std::size_t>(it - open.begin());
    open.erase(it);

    memory::small_vector<memory::EdgeId, SMALL_INDEG> inc(
        graph.incoming(v).begin(), graph.incoming(v).end());

    assert(std::ranges::count(inputs, v) == 0);
    assert(!graph.incoming(v).empty());
    std::sort(inc.begin(), inc.end(), [&](memory::EdgeId a, memory::EdgeId b) {
      return he[*a] < he[*b];
    });

    for (memory::EdgeId e : inc) {
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

      curEdges.push_back(e);
      self(self, cost + graph.weight(e));
      curEdges.pop_back();

      // rollback inserts in reverse order (positions remain valid)
      for (std::size_t i = inserted_positions.size(); i-- > 0;) {
        const std::size_t pos = inserted_positions[i];
        open.erase(open.begin() + static_cast<std::ptrdiff_t>(pos));
      }
    }
    // Restore v into open at original position
    open.insert(open.begin() + static_cast<std::ptrdiff_t>(v_pos), v);
    done.set(*v, false);
  };

  // Where the fun stuff is happening, the actual NP algorithm!
  // NOTE: recursion depth is max the size of the minimum-cost-subgraph,
  //   so generally we should be safe from StackOverflows here.
  recusive_solve(recusive_solve, W{});

  // Reconstruct subgraph from bestEdges.
  // NOTE: subgraph contains all nodes of the original graph,
  //   we add all of them simply because otherwise we would have to 
  //   remap NodeIds everywhere.
  memory::AdjGraph<V, E, W> subgraph;
  for (uint64_t i = 0; i < N; ++i) {
    memory::NodeId nid{i};
    memory::NodeId _nid = subgraph.addNode(graph.get(nid));
    assert(_nid == nid);
  }

  for (memory::EdgeId e : bestEdges) {
    subgraph.addEdge(graph.src(e), graph.dst(e), graph.get(e), graph.weight(e));
  }

  return subgraph;
}

} // namespace denox::algorithm

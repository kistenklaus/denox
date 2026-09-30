#include "denox/algorithm/all_minimum_cost_subgraphs.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

namespace {
using denox::memory::EdgeId;
using denox::memory::NodeId;
using Graph = denox::memory::ConstGraph<unsigned, unsigned, uint32_t>;
using Builder = denox::memory::AdjGraph<unsigned, unsigned, uint32_t>;

struct Edge {
  std::vector<unsigned> sources;
  unsigned destination;
  uint32_t cost;
};

std::vector<unsigned> solve(unsigned nodes, const std::vector<Edge> &edges,
                            const std::vector<unsigned> &inputs,
                            const std::vector<unsigned> &outputs,
                            size_t max_states = 4096,
                            bool *truncated = nullptr) {
  Builder builder;
  for (unsigned i = 0; i < nodes; ++i)
    builder.addNode(i);
  for (unsigned i = 0; i < edges.size(); ++i) {
    std::vector<NodeId> sources;
    for (auto source : edges[i].sources)
      sources.emplace_back(source);
    builder.addEdge(sources, NodeId{edges[i].destination}, i, edges[i].cost);
  }
  std::vector<NodeId> in, out;
  for (auto v : inputs)
    in.emplace_back(v);
  for (auto v : outputs)
    out.emplace_back(v);
  Graph graph(builder);
  Graph result(denox::algorithm::all_minimum_cost_subgraphs(
      graph, in, out, max_states, truncated));

  EXPECT_EQ(result.nodeCount(), nodes);
  for (unsigned i = 0; i < nodes; ++i)
    EXPECT_EQ(result.get(NodeId{i}), i);
  std::vector<unsigned> kept;
  for (unsigned i = 0; i < result.edgeCount(); ++i) {
    EdgeId id{i};
    unsigned original = result.get(id);
    kept.push_back(original);
    EXPECT_EQ(result.weight(id), edges[original].cost);
    EXPECT_EQ(*result.dst(id), edges[original].destination);
    std::vector<unsigned> sources;
    for (auto v : result.src(id))
      sources.push_back(*v);
    EXPECT_EQ(sources, edges[original].sources);
  }
  std::sort(kept.begin(), kept.end());
  return kept;
}

// Independent oracle: enumerate edge subsets, then compute forward
// reachability. Random cases use positive weights so irrelevant edges cannot be
// optimal.
std::vector<unsigned> exhaustive(unsigned nodes, const std::vector<Edge> &edges,
                                 const std::vector<unsigned> &inputs,
                                 const std::vector<unsigned> &outputs) {
  uint64_t best = std::numeric_limits<uint64_t>::max(), unionMask = 0;
  for (uint64_t mask = 0; mask < (uint64_t{1} << edges.size()); ++mask) {
    uint64_t cost = 0;
    for (unsigned e = 0; e < edges.size(); ++e)
      if (mask & (uint64_t{1} << e))
        cost += edges[e].cost;
    if (cost > best)
      continue;
    std::vector<bool> available(nodes, false);
    for (auto v : inputs)
      available[v] = true;
    bool changed;
    do {
      changed = false;
      for (unsigned e = 0; e < edges.size(); ++e) {
        const auto &edge = edges[e];
        if (!(mask & (uint64_t{1} << e)) || available[edge.destination])
          continue;
        if (std::all_of(edge.sources.begin(), edge.sources.end(),
                        [&](unsigned v) { return available[v]; })) {
          available[edge.destination] = true;
          changed = true;
        }
      }
    } while (changed);
    if (!std::all_of(outputs.begin(), outputs.end(),
                     [&](unsigned v) { return available[v]; }))
      continue;
    if (cost < best) {
      best = cost;
      unionMask = mask;
    } else
      unionMask |= mask;
  }
  std::vector<unsigned> kept;
  for (unsigned e = 0; e < edges.size(); ++e)
    if (unionMask & (uint64_t{1} << e))
      kept.push_back(e);
  return kept;
}

TEST(AllMinimumCostSubgraphs, Chain) {
  EXPECT_EQ(solve(3, {{{0}, 1, 2}, {{1}, 2, 3}}, {0}, {2}),
            (std::vector<unsigned>{0, 1}));
}

TEST(AllMinimumCostSubgraphs, RejectsExpensiveAndIrrelevantEdges) {
  EXPECT_EQ(
      solve(4, {{{0}, 1, 1}, {{1}, 2, 1}, {{0}, 2, 3}, {{0}, 3, 1}}, {0}, {2}),
      (std::vector<unsigned>{0, 1}));
}

TEST(AllMinimumCostSubgraphs, KeepsAllTiedPathsAndParallelEdges) {
  EXPECT_EQ(solve(4,
                  {{{0}, 1, 1},
                   {{1}, 3, 1},
                   {{0}, 2, 1},
                   {{2}, 3, 1},
                   {{0}, 3, 2},
                   {{0}, 3, 2}},
                  {0}, {3}),
            (std::vector<unsigned>{0, 1, 2, 3, 4, 5}));
}

TEST(AllMinimumCostSubgraphs, HyperedgeRequiresEverySource) {
  EXPECT_EQ(solve(4, {{{0}, 1, 1}, {{1, 2}, 3, 1}, {{0}, 3, 5}}, {0}, {3}),
            (std::vector<unsigned>{2}));
  EXPECT_EQ(solve(4, {{{0}, 1, 1}, {{1, 2}, 3, 1}}, {0, 2}, {3}),
            (std::vector<unsigned>{0, 1}));
}

TEST(AllMinimumCostSubgraphs, SharesWorkAcrossOutputs) {
  EXPECT_EQ(
      solve(4,
            {{{0}, 1, 3}, {{1}, 2, 1}, {{1}, 3, 1}, {{0}, 2, 3}, {{0}, 3, 3}},
            {0}, {2, 3}),
      (std::vector<unsigned>{0, 1, 2}));
}

TEST(AllMinimumCostSubgraphs, SharedAncestorOfHyperedgeSourcesIsPaidOnce) {
  EXPECT_EQ(
      solve(
          5,
          {{{0}, 1, 3}, {{1}, 2, 1}, {{1}, 3, 1}, {{2, 3}, 4, 1}, {{0}, 4, 7}},
          {0}, {4}),
      (std::vector<unsigned>{0, 1, 2, 3}));
}

TEST(AllMinimumCostSubgraphs, HandlesDuplicateAndAlreadyAvailableOutputs) {
  EXPECT_EQ(solve(3, {{{0}, 1, 1}, {{1}, 2, 1}}, {0, 0}, {2, 1, 2, 0}),
            (std::vector<unsigned>{0, 1}));
  EXPECT_TRUE(solve(1, {}, {0}, {0}).empty());
  EXPECT_TRUE(solve(1, {}, {0}, {}).empty());
}

TEST(AllMinimumCostSubgraphs, UnreachableOutputMakesWholeRequestInfeasible) {
  EXPECT_TRUE(solve(3, {{{0}, 1, 1}}, {0}, {1, 2}).empty());
  EXPECT_TRUE(solve(2, {{{0}, 1, 1}}, {}, {1}).empty());
}

TEST(AllMinimumCostSubgraphs, ZeroCostRequiredEdgesAndTies) {
  EXPECT_EQ(solve(3, {{{0}, 1, 0}, {{1}, 2, 1}, {{0}, 2, 1}}, {0}, {2}),
            (std::vector<unsigned>{0, 1, 2}));
  EXPECT_EQ(solve(3, {{{0}, 1, 0}, {{1}, 2, 0}}, {0}, {2}),
            (std::vector<unsigned>{0, 1}));
}

TEST(AllMinimumCostSubgraphs, ExcludesUnusedZeroCostEdges) {
  EXPECT_EQ(solve(4, {{{0}, 1, 0}, {{0}, 2, 0}, {{1}, 3, 1}}, {0}, {3}),
            (std::vector<unsigned>{0, 2}));
}

TEST(AllMinimumCostSubgraphs, MatchesExhaustiveSubsetsOnRandomHyperDags) {
  std::mt19937 rng(0xD3A0);
  for (unsigned trial = 0; trial < 500; ++trial) {
    unsigned nodes = 2 + rng() % 6;
    std::vector<unsigned> order(nodes);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<Edge> edges;
    unsigned count = rng() % 11;
    for (unsigned e = 0; e < count; ++e) {
      unsigned dst = 1 + rng() % (nodes - 1);
      std::vector<unsigned> sources;
      for (unsigned s = 0; s < dst; ++s)
        if (rng() % 2)
          sources.push_back(order[s]);
      if (sources.empty())
        sources.push_back(order[rng() % dst]);
      std::shuffle(sources.begin(), sources.end(), rng);
      edges.push_back({sources, order[dst], 1 + uint32_t(rng() % 5)});
    }
    std::vector<unsigned> inputs{order[0]}, outputs{order.back()};
    if (nodes > 2 && rng() % 2)
      inputs.push_back(order[1]);
    if (rng() % 2)
      outputs.push_back(order[rng() % nodes]);
    std::ostringstream description;
    description << "seed=0xD3A0 trial=" << trial << " inputs:";
    for (auto v : inputs)
      description << ' ' << v;
    description << " outputs:";
    for (auto v : outputs)
      description << ' ' << v;
    for (unsigned e = 0; e < edges.size(); ++e) {
      description << "\nedge " << e << ": {";
      for (auto v : edges[e].sources)
        description << v << ',';
      description << "} -> " << edges[e].destination
                  << " cost=" << edges[e].cost;
    }
    SCOPED_TRACE(description.str());
    const auto expected = exhaustive(nodes, edges, inputs, outputs);
    ASSERT_EQ(solve(nodes, edges, inputs, outputs), expected);
    ASSERT_EQ(solve(nodes, edges, inputs, outputs), expected);
  }
}

TEST(AllMinimumCostSubgraphs, EmptyGraphAndDisconnectedNodes) {
  EXPECT_TRUE(solve(0, {}, {}, {}).empty());
  EXPECT_TRUE(solve(65, {}, {0}, {64}).empty());
  EXPECT_EQ(solve(65, {{{63}, 64, 1}}, {63}, {64}), (std::vector<unsigned>{0}));
}

TEST(AllMinimumCostSubgraphs, AvailableInputDoesNotNeedItsProducer) {
  EXPECT_EQ(solve(4, {{{0}, 1, 1}, {{1}, 2, 1}, {{2}, 3, 1}}, {0, 2}, {3}),
            (std::vector<unsigned>{2}));
}

TEST(AllMinimumCostSubgraphs, GloballyTiedSharedAndIndependentImplementations) {
  // Shared implementation costs 2 + 1 + 1; independent costs 2 + 2.
  // Both must survive even though each individual output prefers its direct
  // edge.
  const std::vector<Edge> edges = {
      {{0}, 1, 2}, {{1}, 2, 1}, {{1}, 3, 1}, {{0}, 2, 2}, {{0}, 3, 2}};
  EXPECT_EQ(solve(4, edges, {0}, {2, 3}),
            (std::vector<unsigned>{0, 1, 2, 3, 4}));
  EXPECT_EQ(solve(4, edges, {0}, {3, 2}), exhaustive(4, edges, {0}, {2, 3}));
}

TEST(AllMinimumCostSubgraphs, LargeRepresentableCostsAreComparedExactly) {
  // Stay below uint32_t overflow while exceeding float's exact integer range.
  constexpr uint32_t cost = 1'000'000'000;
  EXPECT_EQ(solve(3, {{{0}, 1, cost}, {{1}, 2, cost}, {{0}, 2, 2 * cost + 1}},
                  {0}, {2}),
            (std::vector<unsigned>{0, 1}));
}

TEST(AllMinimumCostSubgraphs, ChainsAcrossBitsetStorageBoundaries) {
  for (unsigned nodes : {63u, 64u, 65u, 511u, 512u, 513u, 1025u}) {
    SCOPED_TRACE(nodes);
    std::vector<Edge> edges;
    std::vector<unsigned> expected;
    // Reverse node IDs so numerical order cannot substitute for topology.
    for (unsigned v = nodes - 1; v > 0; --v) {
      expected.push_back(edges.size());
      edges.push_back({{v}, v - 1, 1});
    }
    edges.push_back({{nodes - 1}, 0, nodes}); // One costlier shortcut.
    EXPECT_EQ(solve(nodes, edges, {nodes - 1}, {0}), expected);
  }
}

TEST(AllMinimumCostSubgraphs, ManyIncomingAlternativesKeepEveryMinimum) {
  std::vector<Edge> edges;
  std::vector<unsigned> expected;
  for (unsigned i = 0; i < 40; ++i) {
    const uint32_t cost = i % 3 == 0 ? 2 : 3;
    if (cost == 2)
      expected.push_back(i);
    edges.push_back({{0}, 1, cost});
  }
  EXPECT_EQ(solve(2, edges, {0}, {1}), expected);
}

TEST(AllMinimumCostSubgraphs,
     WideHyperedgesRestoreFrontierBetweenAlternatives) {
  std::vector<Edge> edges;
  std::vector<unsigned> expected;
  std::vector<unsigned> sources;
  for (unsigned v = 1; v <= 16; ++v) {
    expected.push_back(edges.size());
    edges.push_back({{0}, v, 1});
    sources.push_back(v);
  }
  // Unsorted sources exercise insertion and rollback at changing positions.
  std::reverse(sources.begin(), sources.end());
  expected.push_back(edges.size());
  edges.push_back({sources, 17, 1});
  std::rotate(sources.begin(), sources.begin() + 5, sources.end());
  expected.push_back(edges.size());
  edges.push_back({sources, 17, 1});
  edges.push_back({{0}, 17, 18});
  EXPECT_EQ(solve(18, edges, {0}, {17}), expected);
}

TEST(AllMinimumCostSubgraphs, ManyOutputsShareOneExpensiveIntermediate) {
  std::vector<Edge> edges{{{0}, 1, 20}};
  std::vector<unsigned> expected{0}, outputs;
  for (unsigned v = 2; v < 34; ++v) {
    outputs.push_back(v);
    expected.push_back(edges.size());
    edges.push_back({{1}, v, 1});
    edges.push_back({{0}, v, 3});
  }
  EXPECT_EQ(solve(34, edges, {0}, outputs), expected);
}

TEST(AllMinimumCostSubgraphs,
     SerialDiamondsPreserveExponentiallyManyOptimalPaths) {
  // 2^180 optimal paths, but only 720 edges in their union.
  // This also crosses the inline bitset boundary with a nontrivial DAG.
  constexpr unsigned stages = 180;
  std::vector<Edge> edges;
  std::vector<unsigned> expected;
  for (unsigned stage = 0; stage < stages; ++stage) {
    unsigned start = 3 * stage;
    for (Edge edge :
         {Edge{{start}, start + 1, 1}, Edge{{start}, start + 2, 1},
          Edge{{start + 1}, start + 3, 1}, Edge{{start + 2}, start + 3, 1}}) {
      expected.push_back(edges.size());
      edges.push_back(edge);
    }
    edges.push_back({{start}, start + 3, 3});
  }
  EXPECT_EQ(solve(3 * stages + 1, edges, {0}, {3 * stages}), expected);
}
// A layer of independently computed intermediates feeds many outputs. Each
// output has three competing implementations with overlapping source sets.
// Unlike serial diamonds, choices leave different subsets of shared work open.
struct BranchyCase {
  unsigned intermediates;
  std::vector<Edge> edges;
  std::vector<unsigned> outputs;
};

BranchyCase makeBranchyCase(unsigned width, uint32_t seed, bool tied) {
  BranchyCase result{width, {}, {}};
  std::mt19937 rng(seed);
  for (unsigned v = 1; v <= width; ++v)
    result.edges.push_back({{0}, v, tied ? 2u : 2u + uint32_t(rng() % 5)});
  for (unsigned i = 0; i < width; ++i) {
    unsigned output = width + 1 + i;
    result.outputs.push_back(output);
    for (unsigned option = 0; option < 3; ++option) {
      std::vector<unsigned> sources{1 + (i + option) % width};
      // Include both ordinary and multi-source alternatives. Cyclic overlap
      // is within the intermediate layer's consumers; the graph is a DAG.
      if (option != 0) {
        unsigned other = 1 + (i + option + 1) % width;
        if (option == 2) {
          // Distant overlap prevents the search from decomposing into a chain
          // of local choices as easily as the ring alone would.
          other = 1 + rng() % width;
          if (other == sources.front())
            other = 1 + other % width;
        }
        sources.push_back(other);
      }
      result.edges.push_back(
          {sources, output, tied ? 1u : 1u + uint32_t(rng() % 3)});
    }
  }
  return result;
}

// Independent exact oracle for this two-layer family: enumerate available
// intermediate subsets, then minimize each output independently. This avoids
// enumerating 2^(4*width) edge subsets and shares no backward-search machinery.
std::vector<unsigned> exhaustiveIntermediateSubsets(const BranchyCase &graph) {
  uint64_t best = std::numeric_limits<uint64_t>::max();
  std::vector<bool> unionEdges(graph.edges.size(), false);
  for (uint64_t mask = 0; mask < (uint64_t{1} << graph.intermediates); ++mask) {
    uint64_t cost = 0;
    std::vector<unsigned> selected;
    for (unsigned i = 0; i < graph.intermediates; ++i) {
      if (mask & (uint64_t{1} << i)) {
        cost += graph.edges[i].cost;
        selected.push_back(i);
      }
    }
    bool feasible = true;
    for (unsigned output : graph.outputs) {
      uint32_t minimum = std::numeric_limits<uint32_t>::max();
      std::vector<unsigned> choices;
      for (unsigned e = graph.intermediates; e < graph.edges.size(); ++e) {
        const auto &edge = graph.edges[e];
        if (edge.destination != output ||
            !std::all_of(
                edge.sources.begin(), edge.sources.end(),
                [&](unsigned v) { return mask & (uint64_t{1} << (v - 1)); }))
          continue;
        if (edge.cost < minimum) {
          minimum = edge.cost;
          choices.clear();
        }
        if (edge.cost == minimum)
          choices.push_back(e);
      }
      if (choices.empty()) {
        feasible = false;
        break;
      }
      cost += minimum;
      selected.insert(selected.end(), choices.begin(), choices.end());
    }
    if (!feasible || cost > best)
      continue;
    if (cost < best) {
      best = cost;
      std::fill(unionEdges.begin(), unionEdges.end(), false);
    }
    for (auto e : selected)
      unionEdges[e] = true;
  }
  std::vector<unsigned> result;
  for (unsigned e = 0; e < unionEdges.size(); ++e)
    if (unionEdges[e])
      result.push_back(e);
  return result;
}

TEST(AllMinimumCostSubgraphs, BranchyOracleAgreesWithFullEdgeEnumeration) {
  for (bool tied : {false, true}) {
    auto graph = makeBranchyCase(3, 0xB12A, tied);
    EXPECT_EQ(exhaustiveIntermediateSubsets(graph),
              exhaustive(7, graph.edges, {0}, graph.outputs));
  }
}

TEST(AllMinimumCostSubgraphs, BranchySmallGraphsMatchIndependentOracle) {
  for (unsigned width : {5u, 6u, 7u}) {
    for (uint32_t seed : {0xB12Au, 0x53A9u, 0xD3A0u}) {
      for (bool tied : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "width=" << width << " seed="
                                          << seed << " tied=" << tied);
        auto graph = makeBranchyCase(width, seed, tied);
        const auto expected = exhaustiveIntermediateSubsets(graph);
        EXPECT_EQ(solve(2 * width + 1, graph.edges, {0}, graph.outputs),
                  expected);
        // Reverse insertion order and source/output order without changing the
        // problem; payload IDs are remapped to compare the same original edges.
        std::reverse(graph.edges.begin(), graph.edges.end());
        for (auto &edge : graph.edges)
          std::reverse(edge.sources.begin(), edge.sources.end());
        std::reverse(graph.outputs.begin(), graph.outputs.end());
        auto actual = solve(2 * width + 1, graph.edges, {0}, graph.outputs);
        for (auto &e : actual)
          e = graph.edges.size() - 1 - e;
        std::sort(actual.begin(), actual.end());
        EXPECT_EQ(actual, expected);
      }
    }
  }
}

TEST(AllMinimumCostSubgraphs, BranchyWidthsMatchExactEdgeUnion) {
  for (unsigned width : {5u, 7u, 10u}) {
    auto graph = makeBranchyCase(width, 0xB12A, false);
    EXPECT_EQ(solve(2 * width + 1, graph.edges, {0}, graph.outputs),
              exhaustiveIntermediateSubsets(graph));
  }
}

TEST(AllMinimumCostSubgraphs, ReusesSlotsAcrossWideLayers) {
  // Cross 64 slots, then reuse them in another layer with large node IDs.
  using namespace denox::algorithm;
  for (unsigned width : {32u, 63u}) {
    std::vector<Edge> edges;
    std::vector<unsigned> sources, expected;
    for (unsigned v = 1; v <= width; ++v) {
      edges.push_back({{0}, v, 1});
      sources.push_back(v);
    }
    edges.push_back({sources, width + 1, 1});
    sources.clear();
    for (unsigned v = width + 2; v <= 2 * width + 1; ++v) {
      edges.push_back({{width + 1}, v, 1});
      sources.push_back(v);
    }
    edges.push_back({sources, 2 * width + 2, 1});
    expected.resize(edges.size());
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(solve(2 * width + 3, edges, {0}, {2 * width + 2}), expected);
  }
}

TEST(AllMinimumCostSubgraphs, BeamReportsApproximationAndPreservesFallback) {
  // Individually cheap direct producers cost 6 together; sharing costs 5.
  const std::vector<Edge> edges{
      {{0}, 1, 3}, {{1}, 2, 1}, {{1}, 3, 1}, {{0}, 2, 3}, {{0}, 3, 3}};
  bool truncated = true;
  EXPECT_EQ(solve(4, edges, {0}, {2, 3}, 0, &truncated),
            (std::vector<unsigned>{0, 1, 2}));
  EXPECT_FALSE(truncated);
  EXPECT_EQ(solve(4, edges, {0}, {2, 3}, 32, &truncated),
            (std::vector<unsigned>{0, 1, 2}));
  EXPECT_FALSE(truncated);
  EXPECT_EQ(solve(4, edges, {0}, {2, 3}, 1, &truncated),
            (std::vector<unsigned>{3, 4}));
  EXPECT_TRUE(truncated);
}

TEST(AllMinimumCostSubgraphs, BeamPreservesMergedTiesButReportsDiscardedTies) {
  bool truncated = true;
  EXPECT_EQ(solve(2, {{{0}, 1, 1}, {{0}, 1, 1}}, {0}, {1}, 1, &truncated),
            (std::vector<unsigned>{0, 1}));
  EXPECT_FALSE(truncated); // Two optimal histories, only one distinct state.
  const std::vector<Edge> diamond{
      {{0}, 1, 1}, {{0}, 2, 1}, {{1}, 3, 1}, {{2}, 3, 1}};
  const auto selected = solve(4, diamond, {0}, {3}, 1, &truncated);
  EXPECT_TRUE(truncated);
  EXPECT_EQ(selected.size(), 2u);
  EXPECT_EQ(selected, solve(4, diamond, {0}, {3}, 1));
}

TEST(AllMinimumCostSubgraphs, BeamFiltersUnreachableAlternatives) {
  bool truncated = true;
  EXPECT_EQ(solve(4, {{{0}, 1, 0}, {{2}, 3, 0}, {{1}, 3, 2}}, {0}, {3}, 1,
                  &truncated),
            (std::vector<unsigned>{0, 2}));
  EXPECT_FALSE(truncated);
  EXPECT_TRUE(solve(3, {{{0}, 1, 1}}, {0}, {1, 2}, 1, &truncated).empty());
  EXPECT_FALSE(truncated);
}

TEST(AllMinimumCostSubgraphs, TightBeamsAlwaysCompleteBranchyGraphs) {
  for (unsigned seed = 0; seed < 20; ++seed) {
    const auto graph = makeBranchyCase(7, seed, seed % 2 == 0);
    for (size_t limit : {1u, 2u, 4u, 8u}) {
      SCOPED_TRACE(::testing::Message()
                   << "seed=" << seed << " limit=" << limit);
      bool truncated = false;
      const auto kept =
          solve(15, graph.edges, {0}, graph.outputs, limit, &truncated);
      std::vector<bool> available(15, false);
      available[0] = true;
      bool changed;
      do {
        changed = false;
        for (auto e : kept) {
          const auto &edge = graph.edges[e];
          if (!available[edge.destination] &&
              std::all_of(edge.sources.begin(), edge.sources.end(),
                          [&](unsigned v) { return available[v]; })) {
            available[edge.destination] = true;
            changed = true;
          }
        }
      } while (changed);
      for (auto v : graph.outputs)
        EXPECT_TRUE(available[v]);
      if (!truncated)
        EXPECT_EQ(kept, exhaustiveIntermediateSubsets(graph));
    }
  }
}

} // namespace

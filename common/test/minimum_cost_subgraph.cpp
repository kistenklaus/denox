#include "denox/algorithm/minimum_cost_subgraph.hpp"
#include <gtest/gtest.h>
#include <numeric>
#include <optional>
#include <random>

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

struct Case {
  unsigned nodes;
  std::vector<Edge> edges;
  std::vector<unsigned> inputs, outputs;
  Graph graph() const {
    Builder builder;
    for (unsigned v = 0; v < nodes; ++v)
      builder.addNode(v);
    for (unsigned e = 0; e < edges.size(); ++e) {
      std::vector<NodeId> sources;
      for (auto v : edges[e].sources)
        sources.emplace_back(v);
      builder.addEdge(sources, NodeId{edges[e].destination}, e, edges[e].cost);
    }
    return Graph(builder);
  }
};

std::vector<NodeId> ids(const std::vector<unsigned> &values) {
  std::vector<NodeId> result;
  for (auto v : values)
    result.emplace_back(v);
  return result;
}

bool feasible(const Case &c, const std::vector<bool> &chosen) {
  std::vector<bool> available(c.nodes, false);
  for (auto v : c.inputs)
    available[v] = true;
  bool changed;
  do {
    changed = false;
    for (unsigned e = 0; e < c.edges.size(); ++e) {
      const auto &edge = c.edges[e];
      if (!chosen[e] || available[edge.destination])
        continue;
      if (std::all_of(edge.sources.begin(), edge.sources.end(),
                      [&](unsigned v) { return available[v]; })) {
        available[edge.destination] = true;
        changed = true;
      }
    }
  } while (changed);
  return std::all_of(c.outputs.begin(), c.outputs.end(),
                     [&](unsigned v) { return available[v]; });
}

// Enumerate all edge subsets independently of either solver's search order.
std::optional<uint64_t> oracle(const Case &c) {
  std::optional<uint64_t> best;
  for (uint64_t mask = 0; mask < (uint64_t{1} << c.edges.size()); ++mask) {
    uint64_t cost = 0;
    std::vector<bool> chosen(c.edges.size());
    for (unsigned e = 0; e < c.edges.size(); ++e) {
      chosen[e] = (mask & (uint64_t{1} << e)) != 0;
      if (chosen[e])
        cost += c.edges[e].cost;
    }
    if ((!best || cost < *best) && feasible(c, chosen))
      best = cost;
  }
  return best;
}

void validate(const Case &c, const Graph &result, uint64_t expected) {
  ASSERT_EQ(result.nodeCount(), c.nodes);
  for (unsigned v = 0; v < c.nodes; ++v)
    EXPECT_EQ(result.get(NodeId{v}), v);
  std::vector<bool> chosen(c.edges.size(), false);
  std::vector<unsigned> producers(c.nodes, 0);
  uint64_t cost = 0;
  for (unsigned e = 0; e < result.edgeCount(); ++e) {
    EdgeId id{e};
    unsigned original = result.get(id);
    ASSERT_LT(original, c.edges.size());
    EXPECT_FALSE(chosen[original]);
    chosen[original] = true;
    const auto &edge = c.edges[original];
    EXPECT_EQ(result.weight(id), edge.cost);
    EXPECT_EQ(*result.dst(id), edge.destination);
    EXPECT_EQ(++producers[edge.destination], 1u);
    std::vector<unsigned> sources;
    for (auto v : result.src(id))
      sources.push_back(*v);
    EXPECT_EQ(sources, edge.sources);
    cost += result.weight(id);
  }
  EXPECT_EQ(cost, expected);
  EXPECT_TRUE(feasible(c, chosen));
}

void checkMinimum(const Case &c, uint64_t expected) {
  auto graph = c.graph();
  auto in = ids(c.inputs), out = ids(c.outputs);
  validate(c, Graph(denox::algorithm::minimum_cost_subgraph(graph, in, out)),
           expected);
}

TEST(MinimumCostSubgraph, ChainsAndTiedAlternatives) {
  checkMinimum({4,
                {{{0}, 1, 1},
                 {{1}, 3, 1},
                 {{0}, 2, 1},
                 {{2}, 3, 1},
                 {{0}, 3, 2},
                 {{0}, 3, 3}},
                {0},
                {3}},
               2);
}

TEST(MinimumCostSubgraph, SharedIntermediateAcrossOutputs) {
  checkMinimum(
      {4,
       {{{0}, 1, 3}, {{1}, 2, 1}, {{1}, 3, 1}, {{0}, 2, 3}, {{0}, 3, 3}},
       {0},
       {2, 3}},
      5);
}

TEST(MinimumCostSubgraph, HyperedgeSharesAncestor) {
  checkMinimum(
      {5,
       {{{0}, 1, 3}, {{1}, 2, 1}, {{1}, 3, 1}, {{3, 2}, 4, 1}, {{0}, 4, 7}},
       {0},
       {4}},
      6);
}

TEST(MinimumCostSubgraph, SourceFreeEdges) {
  checkMinimum(
      {3, {{{}, 0, 2}, {{0}, 1, 1}, {{0, 1}, 2, 1}, {{}, 2, 5}}, {}, {2}}, 4);
  checkMinimum({1, {{{}, 0, 0}}, {}, {0}}, 0);
}

TEST(MinimumCostSubgraph, InputsOutputsAndZeroCosts) {
  checkMinimum({0, {}, {}, {}}, 0);
  checkMinimum({1, {}, {0}, {0, 0}}, 0);
  checkMinimum({1, {}, {}, {}}, 0);
  checkMinimum(
      {3, {{{0}, 1, 0}, {{1}, 2, 0}, {{0}, 2, 1}}, {0, 0}, {0, 1, 2, 2}}, 0);
  checkMinimum({3, {{{0}, 1, 20}, {{1}, 2, 2}}, {0, 1}, {2}}, 2);
}

TEST(MinimumCostSubgraph, UnreachableOutputReturnsNoEdges) {
  Case c{3, {{{0}, 1, 1}}, {0}, {1, 2}};
  auto graph = c.graph();
  Graph result(denox::algorithm::minimum_cost_subgraph(graph, ids(c.inputs),
                                                       ids(c.outputs)));
  EXPECT_EQ(result.nodeCount(), 3u);
  EXPECT_EQ(result.edgeCount(), 0u);
}

TEST(MinimumCostSubgraph, RandomDerivableDagsMatchExhaustiveMinimum) {
  std::mt19937 rng(0xC057);
  for (unsigned trial = 0; trial < 200; ++trial) {
    SCOPED_TRACE(trial);
    const unsigned n = 3 + rng() % 6;
    std::vector<unsigned> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    Case c{n, {}, {order[0]}, {order.back(), order[n - 2]}};
    for (unsigned v = 1; v < n; ++v)
      c.edges.push_back({{order[v - 1]}, order[v], uint32_t(rng() % 5)});
    for (unsigned e = 0; e < 5; ++e) {
      unsigned dst = 1 + rng() % (n - 1);
      std::vector<unsigned> sources;
      for (unsigned u = 0; u < dst; ++u)
        if (rng() % 2)
          sources.push_back(order[u]);
      std::shuffle(sources.begin(), sources.end(), rng);
      c.edges.push_back({sources, order[dst], uint32_t(rng() % 5)});
    }
    const auto expected = oracle(c);
    ASSERT_TRUE(expected);
    checkMinimum(c, *expected);
  }
}

TEST(MinimumCostSubgraph, LongChainAndSlotReuse) {
  Case c{1025, {}, {1024}, {0}};
  for (unsigned v = 1024; v > 0; --v)
    c.edges.push_back({{v}, v - 1, 1});
  checkMinimum(c, 1024);
}

TEST(MinimumCostSubgraph, SupportsExactly64SlotsRejects65) {
  for (unsigned width : {63u, 64u, 65u}) {
    Case c{width + 2, {}, {0}, {width + 1}};
    std::vector<unsigned> sources;
    for (unsigned v = 1; v <= width; ++v) {
      c.edges.push_back({{0}, v, 1});
      sources.push_back(v);
    }
    c.edges.push_back({sources, width + 1, 1});
    auto graph = c.graph();
    if (width <= 64)
      checkMinimum(c, width + 1);
    else
      EXPECT_THROW(denox::algorithm::minimum_cost_subgraph(graph, ids(c.inputs),
                                                           ids(c.outputs)),
                   std::runtime_error);
  }
}

TEST(MinimumCostSubgraph, FloatingPointWeights) {
  denox::memory::AdjGraph<unsigned, unsigned, double> builder;
  for (unsigned v = 0; v < 3; ++v)
    builder.addNode(v);
  builder.addEdge(NodeId{0}, NodeId{1}, 0, 0.25);
  builder.addEdge(NodeId{1}, NodeId{2}, 1, 0.5);
  builder.addEdge(NodeId{0}, NodeId{2}, 2, 1.0);
  denox::memory::ConstGraph<unsigned, unsigned, double> graph(builder);
  const auto in = ids({0}), out = ids({2});
  const auto result = denox::algorithm::minimum_cost_subgraph(graph, in, out);
  EXPECT_EQ(result.edgeCount(), 2u);
}
TEST(MinimumCostSubgraph, BeamLimitAndTruncationFlag) {
  const Case c{
      4,
      {{{0}, 1, 3}, {{1}, 2, 1}, {{1}, 3, 1}, {{0}, 2, 3}, {{0}, 3, 3}},
      {0},
      {2, 3}};
  const auto graph = c.graph();
  bool truncated = true;
  for (size_t limit : {0u, 32u}) {
    Graph result(denox::algorithm::minimum_cost_subgraph(
        graph, ids(c.inputs), ids(c.outputs), limit, &truncated));
    validate(c, result, 5);
    EXPECT_FALSE(truncated);
  }
  Graph limited(denox::algorithm::minimum_cost_subgraph(
      graph, ids(c.inputs), ids(c.outputs), 1, &truncated));
  validate(c, limited,
           6); // Protected direct derivation, not the global optimum.
  EXPECT_TRUE(truncated);
}

TEST(MinimumCostSubgraph,
     BeamHandlesSourceFreeEdgesAndUnreachableAlternatives) {
  const Case c{4, {{{}, 0, 0}, {{0}, 1, 1}, {{2}, 3, 0}, {{1}, 3, 1}}, {}, {3}};
  bool truncated = true;
  Graph result(denox::algorithm::minimum_cost_subgraph(
      c.graph(), ids(c.inputs), ids(c.outputs), 1, &truncated));
  validate(c, result, 2);
  EXPECT_FALSE(truncated);
  const Case impossible{2, {}, {0}, {1}};
  Graph empty(denox::algorithm::minimum_cost_subgraph(
      impossible.graph(), ids(impossible.inputs), ids(impossible.outputs), 1,
      &truncated));
  EXPECT_EQ(empty.nodeCount(), 2u);
  EXPECT_EQ(empty.edgeCount(), 0u);
  EXPECT_FALSE(truncated);
}

TEST(MinimumCostSubgraph, BeamKeepsOneProducerForMergedTies) {
  const Case c{2, {{{0}, 1, 0}, {{0}, 1, 0}}, {0}, {1}};
  bool truncated = true;
  Graph result(denox::algorithm::minimum_cost_subgraph(
      c.graph(), ids(c.inputs), ids(c.outputs), 1, &truncated));
  validate(c, result, 0);
  EXPECT_EQ(result.edgeCount(), 1u);
  EXPECT_FALSE(truncated);
}

TEST(MinimumCostSubgraph, TightBeamsReturnFeasibleDeterministicSolutions) {
  std::mt19937 rng(0xBEA1);
  for (unsigned trial = 0; trial < 80; ++trial) {
    SCOPED_TRACE(trial);
    Case c{8, {}, {0}, {5, 6, 7}};
    for (unsigned v = 1; v < 8; ++v) {
      c.edges.push_back({{v - 1}, v, uint32_t(rng() % 5)});
      std::vector<unsigned> sources;
      for (unsigned u = 0; u < v; ++u)
        if (rng() % 2)
          sources.push_back(u);
      c.edges.push_back({sources, v, uint32_t(rng() % 5)});
    }
    const auto expected = oracle(c);
    ASSERT_TRUE(expected);
    const auto graph = c.graph();
    for (size_t limit : {1u, 2u, 4u}) {
      bool truncated = false;
      Graph result(denox::algorithm::minimum_cost_subgraph(
          graph, ids(c.inputs), ids(c.outputs), limit, &truncated));
      uint64_t cost = 0;
      for (size_t e = 0; e < result.edgeCount(); ++e)
        cost += result.weight(EdgeId{e});
      validate(c, result,
               cost); // Checks reachability, payloads, and single producers.
      EXPECT_GE(cost, *expected);
      if (!truncated)
        EXPECT_EQ(cost, *expected);
      Graph again(denox::algorithm::minimum_cost_subgraph(
          graph, ids(c.inputs), ids(c.outputs), limit));
      ASSERT_EQ(result.edgeCount(), again.edgeCount());
      for (size_t e = 0; e < result.edgeCount(); ++e)
        EXPECT_EQ(result.get(EdgeId{e}), again.get(EdgeId{e}));
    }
  }
}
} // namespace

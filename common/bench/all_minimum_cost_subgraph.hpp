#pragma once

#include "denox/algorithm/all_minimum_cost_subgraphs.hpp"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace denox_bench {
using denox::memory::EdgeId;
using denox::memory::NodeId;
using Graph = denox::memory::ConstGraph<unsigned, unsigned, uint32_t>;
using Builder = denox::memory::AdjGraph<unsigned, unsigned, uint32_t>;
struct Edge {
  std::vector<unsigned> sources;
  unsigned destination;
  uint32_t cost;
};

// Generalizes the solver comment's {C1, C2, ...} -> O example. Every C has
// eight distinct rank-3 alternatives over a shared pool of intermediates.
// The benchmark adds the final all-C join after constructing this fixture.
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
    std::vector<std::vector<unsigned>> alternatives;
    while (alternatives.size() < 8) {
      std::vector<unsigned> sources;
      while (sources.size() < 3) {
        const unsigned source = 1 + rng() % width;
        if (std::find(sources.begin(), sources.end(), source) == sources.end())
          sources.push_back(source);
      }
      std::sort(sources.begin(), sources.end());
      if (std::find(alternatives.begin(), alternatives.end(), sources) !=
          alternatives.end())
        continue;
      alternatives.push_back(sources);
      result.edges.push_back(
          {sources, output, tied ? 1u : 1u + uint32_t(rng() % 3)});
    }
  }
  return result;
}

enum class Solver {
  Exact,
  Beam64,
};

static void minimumCostSubgraphs(benchmark::State &state, Solver solver) {
  const unsigned width = static_cast<unsigned>(state.range(0));
  const auto fixture = makeBranchyCase(width, 0xB12A, false);
  Builder builder;
  for (unsigned v = 0; v < 2 * width + 1; ++v)
    builder.addNode(v);
  for (unsigned e = 0; e < fixture.edges.size(); ++e) {
    const auto &edge = fixture.edges[e];
    std::vector<NodeId> sources;
    for (auto v : edge.sources)
      sources.emplace_back(v);
    builder.addEdge(sources, NodeId{edge.destination}, e, edge.cost);
  }
  const std::vector<NodeId> inputs{NodeId{0}};
  std::vector<NodeId> branches;
  for (auto v : fixture.outputs)
    branches.emplace_back(v);
  const NodeId output = builder.addNode(2 * width + 1);
  builder.addEdge(branches, output, fixture.edges.size(), 1);
  const std::vector<NodeId> outputs{output};
  const Graph graph(builder);

  auto run = [&]() {
    return denox::algorithm::all_minimum_cost_subgraphs(
        graph, inputs, outputs, solver == Solver::Beam64 ? 64 : 0);
  };

  for (auto _ : state) {
    auto result = run();
    benchmark::DoNotOptimize(result);
    benchmark::ClobberMemory();
  }
  state.counters["nodes"] = graph.nodeCount();
  state.counters["edges"] = graph.edgeCount();
  state.counters["outputs"] = outputs.size();
  state.counters["alternatives"] = 8;
  state.counters["alternative_rank"] = 3;
  state.counters["join_rank"] = width;
}

BENCHMARK_CAPTURE(minimumCostSubgraphs, DenseHyperedgeExact, Solver::Exact)
    ->Arg(10)
    ->Arg(14)
    ->Arg(18)
    ->Arg(22)
    ->Arg(24)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(minimumCostSubgraphs, DenseHyperedgeBeam64, Solver::Beam64)
    ->Arg(10)
    ->Arg(14)
    ->Arg(18)
    ->Arg(22)
    ->Arg(24)
    ->Arg(30)
    ->Arg(36)
    ->Arg(42)
    ->Arg(48)
    ->Unit(benchmark::kMillisecond);
} // namespace denox_bench

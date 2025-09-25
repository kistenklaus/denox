#include "algorithm/shortest_dag_hyperpath.hpp"
#include "memory/hypergraph/AdjGraph.hpp"
#include "memory/hypergraph/ConstGraph.hpp"
#include <gtest/gtest.h>

using namespace denox::memory;
using namespace denox::algorithm;

#include "algorithm/shortest_dag_hyperpath.hpp"
#include "memory/hypergraph/AdjGraph.hpp"
#include "memory/hypergraph/ConstGraph.hpp"
#include <gtest/gtest.h>

using namespace denox::memory;
using namespace denox::algorithm;

TEST(algorithm_shortest_dag_hyperpath, chain_prefers_two_step_over_direct) {
  AdjGraph<int, int, float> G;
  const NodeId s = G.addNode(1);
  const NodeId b = G.addNode(2);
  const NodeId t = G.addNode(3);

  (void)G.addEdge(s, b, 101, 2.0f); // s->b (2)
  (void)G.addEdge(b, t, 102, 2.0f); // b->t (2)  total 4
  (void)G.addEdge(s, t, 103, 5.0f); // s->t (5)  should lose

  ConstGraph<int, int, float> CG{G};

  vector<NodeId> starts_vec{s};
  vector<NodeId> ends_vec{t};
  span<const NodeId> starts(starts_vec.begin(), starts_vec.end());
  span<const NodeId> ends(ends_vec.begin(), ends_vec.end());

  auto res = shortest_dag_hyperpath<int, int, float>(CG, starts, ends);
  ASSERT_TRUE(res.has_value());
  const auto &order = *res;

  // Firing validity & reachability
  vector<unsigned char> avail(CG.nodeCount(), 0);
  for (auto s0 : starts_vec)
    avail[s0] = 1;
  for (auto e : order) {
    for (auto u : CG.src(e))
      EXPECT_TRUE(avail[u]);
    avail[CG.dst(e)] = 1;
  }
  bool reached = false;
  for (auto z : ends_vec)
    reached |= (avail[z] != 0);
  EXPECT_TRUE(reached);

  // Cost and structure
  float total = 0.f;
  for (auto e : order)
    total += CG.weight(e);
  EXPECT_EQ(order.size(), 2u);
  EXPECT_NEAR(total, 4.0f, 1e-6f);
  EXPECT_EQ(CG.dst(order.back()), t);
}

TEST(algorithm_shortest_dag_hyperpath, branching_two_tail_edge_beats_direct) {
  AdjGraph<int, int, float> G;
  const NodeId s = G.addNode(1);
  const NodeId x = G.addNode(2);
  const NodeId y = G.addNode(3);
  const NodeId t = G.addNode(4);

  (void)G.addEdge(s, x, 201, 0.5f);    // s->x (0.5)
  (void)G.addEdge(s, y, 202, 0.75f);   // s->y (0.75)
  (void)G.addEdge(x, y, t, 203, 1.0f); // {x,y}->t (1.0) total 2.25
  (void)G.addEdge(s, t, 204, 3.0f);    // s->t (3.0) should lose

  ConstGraph<int, int, float> CG{G};

  vector<NodeId> starts_vec{s};
  vector<NodeId> ends_vec{t};
  span<const NodeId> starts(starts_vec.begin(), starts_vec.end());
  span<const NodeId> ends(ends_vec.begin(), ends_vec.end());

  auto res = shortest_dag_hyperpath<int, int, float>(CG, starts, ends);
  ASSERT_TRUE(res.has_value());
  const auto &order = *res;

  vector<unsigned char> avail(CG.nodeCount(), 0);
  for (auto s0 : starts_vec)
    avail[s0] = 1;
  for (auto e : order) {
    for (auto u : CG.src(e))
      EXPECT_TRUE(avail[u]);
    avail[CG.dst(e)] = 1;
  }
  bool reached = false;
  for (auto z : ends_vec)
    reached |= (avail[z] != 0);
  EXPECT_TRUE(reached);

  float total = 0.f;
  for (auto e : order)
    total += CG.weight(e);
  EXPECT_EQ(order.size(), 3u);
  EXPECT_NEAR(total, 2.25f, 1e-6f);
  EXPECT_EQ(CG.dst(order.back()), t);
}

TEST(algorithm_shortest_dag_hyperpath, multi_source_two_tail_edge_single_step) {
  AdjGraph<int, int, float> G;
  const NodeId s1 = G.addNode(1);
  const NodeId s2 = G.addNode(2);
  const NodeId t = G.addNode(3);

  (void)G.addEdge(s1, s2, t, 301, 2.5f); // {s1,s2}->t (2.5)

  ConstGraph<int, int, float> CG{G};

  vector<NodeId> starts_vec{s1, s2};
  vector<NodeId> ends_vec{t};
  span<const NodeId> starts(starts_vec.begin(), starts_vec.end());
  span<const NodeId> ends(ends_vec.begin(), ends_vec.end());

  auto res = shortest_dag_hyperpath<int, int, float>(CG, starts, ends);
  ASSERT_TRUE(res.has_value());
  const auto &order = *res;

  vector<unsigned char> avail(CG.nodeCount(), 0);
  for (auto s0 : starts_vec)
    avail[s0] = 1;
  for (auto e : order) {
    for (auto u : CG.src(e))
      EXPECT_TRUE(avail[u]);
    avail[CG.dst(e)] = 1;
  }
  bool reached = false;
  for (auto z : ends_vec)
    reached |= (avail[z] != 0);
  EXPECT_TRUE(reached);

  ASSERT_EQ(order.size(), 1u);
  EXPECT_EQ(CG.dst(order[0]), t);
  EXPECT_NEAR(CG.weight(order[0]), 2.5f, 1e-6f);
}

TEST(algorithm_shortest_dag_hyperpath, unreachable_returns_nullopt) {
  AdjGraph<int, int, float> G;
  const NodeId s = G.addNode(1);
  const NodeId t = G.addNode(2);
  (void)s;
  (void)t; // no edges

  ConstGraph<int, int, float> CG{G};

  vector<NodeId> starts_vec{s};
  vector<NodeId> ends_vec{t};
  span<const NodeId> starts(starts_vec.begin(), starts_vec.end());
  span<const NodeId> ends(ends_vec.begin(), ends_vec.end());

  auto res = shortest_dag_hyperpath<int, int, float>(CG, starts, ends);
  EXPECT_FALSE(res.has_value());
}

TEST(algorithm_shortest_dag_hyperpath, chooses_best_among_multiple_targets) {
  AdjGraph<int, int, float> G;
  const NodeId s = G.addNode(1);
  const NodeId a = G.addNode(2);
  const NodeId t1 = G.addNode(3);
  const NodeId t2 = G.addNode(4);

  (void)G.addEdge(s, a, 401, 1.0f);  // s->a (1)
  (void)G.addEdge(a, t1, 402, 5.0f); // a->t1 (5) total 6
  (void)G.addEdge(s, t2, 403, 3.0f); // s->t2 (3) should win

  ConstGraph<int, int, float> CG{G};

  vector<NodeId> starts_vec{s};
  vector<NodeId> ends_vec{t1, t2};
  span<const NodeId> starts(starts_vec.begin(), starts_vec.end());
  span<const NodeId> ends(ends_vec.begin(), ends_vec.end());

  auto res = shortest_dag_hyperpath<int, int, float>(CG, starts, ends);
  ASSERT_TRUE(res.has_value());
  const auto &order = *res;

  vector<unsigned char> avail(CG.nodeCount(), 0);
  for (auto s0 : starts_vec)
    avail[s0] = 1;
  for (auto e : order) {
    for (auto u : CG.src(e))
      EXPECT_TRUE(avail[u]);
    avail[CG.dst(e)] = 1;
  }
  // It should reach t2 specifically
  EXPECT_TRUE(avail[t2]);

  ASSERT_EQ(order.size(), 1u);
  EXPECT_EQ(CG.dst(order[0]), t2);
  EXPECT_NEAR(CG.weight(order[0]), 3.0f, 1e-6f);
}

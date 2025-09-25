#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "algorithm/pattern_matching/match.hpp"
#include "memory/hypergraph/AdjGraph.hpp"
#include "memory/hypergraph/ConstGraph.hpp"
#include "memory/hypergraph/LinkedGraph.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <random>

using namespace denox::memory;
using namespace denox::algorithm;

TEST(algorithm_match_all, singular_node_match) {

  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  [[maybe_unused]] NodeId B = adj.addNode(2);
  NodeId C = adj.addNode(1);
  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](const int &v) { return v == 1; });

  std::vector<ConstGraphMatch<int, int>> matches;
  for (const auto &match : match_all(pattern, graph)) {
    matches.push_back(match);
  }

  ASSERT_EQ(matches.size(), 2);

  auto A_it = std::find_if(matches.begin(), matches.end(),
                           [&](const auto &match) { return A == match[m_X]; });
  ASSERT_TRUE(A_it != matches.end());
  matches.erase(A_it);
  ASSERT_EQ(matches.size(), 1);

  auto C_it = std::find_if(matches.begin(), matches.end(),
                           [&](const auto &match) { return C == match[m_X]; });
  ASSERT_TRUE(C_it != matches.end());

  matches.erase(C_it);

  EXPECT_TRUE(matches.empty());
}

TEST(algorithm_match_all, singular_edge_match) {
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  NodeId C = adj.addNode(1);

  EdgeId AB = adj.addEdge(A, B, 42);
  EdgeId BC = adj.addEdge(B, C, 42);

  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchEdge();
  m_X->matchValue([](const int &v) { return v == 42; });

  std::vector<ConstGraphMatch<int, int>> matches;
  for (const auto &match : match_all(pattern, graph)) {
    matches.push_back(match);
  }

  ASSERT_EQ(matches.size(), 2);

  auto AB_it =
      std::find_if(matches.begin(), matches.end(),
                   [&](const auto &match) { return AB == match[m_X]; });
  ASSERT_TRUE(AB_it != matches.end());
  matches.erase(AB_it);
  ASSERT_EQ(matches.size(), 1);

  auto BC_it =
      std::find_if(matches.begin(), matches.end(),
                   [&](const auto &match) { return BC == match[m_X]; });
  ASSERT_TRUE(BC_it != matches.end());

  matches.erase(BC_it);

  EXPECT_TRUE(matches.empty());
}

TEST(algorithm_match_all, multiple_matches_per_node) {
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(1);
  NodeId C = adj.addNode(1);
  EdgeId AB = adj.addEdge(A, B, 1);
  EdgeId AC = adj.addEdge(A, C, 1);

  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](const int &v) { return v == 1; });
  auto e = m_X->matchOutgoing();
  e->matchRank(1);
  auto m_Y = e->matchDst();

  std::vector<ConstGraphMatch<int, int>> matches;
  for (const auto &match : match_all(pattern, graph)) {
    matches.push_back(match);
  }

  // Only node A has two outgoing edges with rank 1; X can also be B or C but
  // they have no outgoing edges -> only the two (A->B) and (A->C) matches
  ASSERT_EQ(matches.size(), 2u);

  auto hasAB = std::any_of(matches.begin(), matches.end(), [&](const auto &m) {
    return m[m_X] == A && m[e] == AB && m[m_Y] == B;
  });
  auto hasAC = std::any_of(matches.begin(), matches.end(), [&](const auto &m) {
    return m[m_X] == A && m[e] == AC && m[m_Y] == C;
  });
  EXPECT_TRUE(hasAB);
  EXPECT_TRUE(hasAC);
}

TEST(algorithm_match_all, two_required_outgoing_edges_from_same_node) {
  // Expect exactly the two permutations (AB,AC) and (AC,AB), no edge reuse.
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  NodeId C = adj.addNode(3);
  EdgeId AB = adj.addEdge(A, B, 1);
  EdgeId AC = adj.addEdge(A, C, 1);
  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](const int &v) { return v == 1; });

  auto e1 = m_X->matchOutgoing();
  e1->matchRank(1);
  auto m_Y1 = e1->matchDst();

  auto e2 = m_X->matchOutgoing();
  e2->matchRank(1);
  auto m_Y2 = e2->matchDst();

  std::vector<ConstGraphMatch<int, int>> matches;
  for (const auto &m : match_all(pattern, graph)) {
    matches.push_back(m);
  }

  ASSERT_EQ(matches.size(), 2u);

  auto it_case1 =
      std::find_if(matches.begin(), matches.end(), [&](const auto &m) {
        return m[m_X] == A && m[e1] == AB && m[m_Y1] == B && m[e2] == AC &&
               m[m_Y2] == C;
      });
  auto it_case2 =
      std::find_if(matches.begin(), matches.end(), [&](const auto &m) {
        return m[m_X] == A && m[e1] == AC && m[m_Y1] == C && m[e2] == AB &&
               m[m_Y2] == B;
      });
  EXPECT_TRUE(it_case1 != matches.end() || it_case2 != matches.end());
}

TEST(algorithm_match_all, two_required_outgoing_but_only_one_edge_available) {
  // X requires two distinct outgoing edges of rank 1, but only one exists -> 0
  // matches
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  [[maybe_unused]] EdgeId AB = adj.addEdge(A, B, 1);
  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](const int &v) { return v == 1; });

  auto e1 = m_X->matchOutgoing();
  e1->matchRank(1);
  auto m_Y1 = e1->matchDst();
  auto e2 = m_X->matchOutgoing();
  e2->matchRank(1);
  auto m_Y2 = e2->matchDst();

  std::size_t count = 0;
  for (const auto &m : match_all(pattern, graph)) {
    (void)m;
    ++count;
  }
  EXPECT_EQ(count, 0u);
}

TEST(algorithm_match_all, three_required_outgoing_edges_permutations) {
  // 3 distinct rank-1 edges from A -> permutations = 3! = 6 matches
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  NodeId C = adj.addNode(3);
  NodeId D = adj.addNode(4);
  [[maybe_unused]] EdgeId AB = adj.addEdge(A, B, 1);
  [[maybe_unused]] EdgeId AC = adj.addEdge(A, C, 1);
  [[maybe_unused]] EdgeId AD = adj.addEdge(A, D, 1);
  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](const int &v) { return v == 1; });

  auto e1 = m_X->matchOutgoing();
  e1->matchRank(1);
  auto m_Y1 = e1->matchDst();
  auto e2 = m_X->matchOutgoing();
  e2->matchRank(1);
  auto m_Y2 = e2->matchDst();
  auto e3 = m_X->matchOutgoing();
  e3->matchRank(1);
  auto m_Y3 = e3->matchDst();

  std::size_t count = 0;
  for (const auto &m : match_all(pattern, graph)) {
    (void)m;
    ++count;
  }
  EXPECT_EQ(count, 6u);
}

TEST(algorithm_match_all, edge_root_with_dst) {
  // Root is an edge; enumerate both edges whose value==42, and bind dst nodes.
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  NodeId C = adj.addNode(3);
  EdgeId AB = adj.addEdge(A, B, 42);
  EdgeId AC = adj.addEdge(A, C, 42);
  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto e = pattern.matchEdge();
  e->matchValue([](const int &v) { return v == 42; });
  auto m_Y = e->matchDst();

  std::vector<ConstGraphMatch<int, int>> matches;
  for (const auto &m : match_all(pattern, graph)) {
    matches.push_back(m);
  }

  ASSERT_EQ(matches.size(), 2u);
  EXPECT_TRUE(std::any_of(matches.begin(), matches.end(), [&](const auto &m) {
    return m[e] == AB && m[m_Y] == B;
  }));
  EXPECT_TRUE(std::any_of(matches.begin(), matches.end(), [&](const auto &m) {
    return m[e] == AC && m[m_Y] == C;
  }));
}

TEST(algorithm_match_all, two_required_outgoing_permutations_nPk_3P2) {
  // A has 3 distinct outgoing edges of rank 1. Pattern requires two -> 3P2 = 6
  // matches.
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  NodeId C = adj.addNode(3);
  NodeId D = adj.addNode(4);
  [[maybe_unused]] EdgeId AB = adj.addEdge(A, B, 1);
  [[maybe_unused]] EdgeId AC = adj.addEdge(A, C, 1);
  [[maybe_unused]] EdgeId AD = adj.addEdge(A, D, 1);

  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](const int &v) { return v == 1; });

  auto e1 = m_X->matchOutgoing();
  e1->matchRank(1);
  auto m_Y1 = e1->matchDst();
  auto e2 = m_X->matchOutgoing();
  e2->matchRank(1);
  auto m_Y2 = e2->matchDst();

  std::size_t count = 0;
  for (const auto &m : match_all(pattern, graph)) {
    (void)m;
    ++count;
  }
  EXPECT_EQ(count,
            6u); // permutations: AB/AC, AC/AB, AB/AD, AD/AB, AC/AD, AD/AC
}

TEST(algorithm_match_all, two_required_outgoing_parallel_edges_same_dst) {
  // Two parallel edges A->B (rank 1). Requiring two distinct outgoing edges ->
  // 2! = 2 matches.
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  EdgeId AB1 = adj.addEdge(A, B, 1);
  EdgeId AB2 = adj.addEdge(A, B, 1);

  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](const int &v) { return v == 1; });

  auto e1 = m_X->matchOutgoing();
  e1->matchRank(1);
  auto m_Y1 = e1->matchDst();
  auto e2 = m_X->matchOutgoing();
  e2->matchRank(1);
  auto m_Y2 = e2->matchDst();

  std::vector<ConstGraphMatch<int, int>> matches;
  for (const auto &m : match_all(pattern, graph))
    matches.push_back(m);

  ASSERT_EQ(matches.size(), 2u);
  // Accept both permutations (AB1,AB2) or (AB2,AB1)
  auto has12 = std::any_of(matches.begin(), matches.end(), [&](const auto &m) {
    return m[m_X] == A && m[e1] == AB1 && m[e2] == AB2 && m[m_Y1] == B &&
           m[m_Y2] == B;
  });
  auto has21 = std::any_of(matches.begin(), matches.end(), [&](const auto &m) {
    return m[m_X] == A && m[e1] == AB2 && m[e2] == AB1 && m[m_Y1] == B &&
           m[m_Y2] == B;
  });
  EXPECT_TRUE(has12 || has21);
}

TEST(algorithm_match_all, nested_two_step_paths_from_root_node) {
  // A has edges to B and C; B has edges to D and E; C has edge to F.
  // Pattern: X -> Y (rank 1), then Y -> Z (rank 1).
  // Expected matches: A->B->D, A->B->E, A->C->F  => 3 total.
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  NodeId C = adj.addNode(3);
  NodeId D = adj.addNode(4);
  NodeId E = adj.addNode(5);
  NodeId F = adj.addNode(6);
  EdgeId AB = adj.addEdge(A, B, 1);
  EdgeId AC = adj.addEdge(A, C, 1);
  EdgeId BD = adj.addEdge(B, D, 1);
  EdgeId BE = adj.addEdge(B, E, 1);
  EdgeId CF = adj.addEdge(C, F, 1);

  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](const int &v) { return v == 1; });

  auto e1 = m_X->matchOutgoing();
  e1->matchRank(1);
  auto m_Y = e1->matchDst();

  auto e2 = m_Y->matchOutgoing();
  e2->matchRank(1);
  auto m_Z = e2->matchDst();

  std::vector<ConstGraphMatch<int, int>> matches;
  for (const auto &m : match_all(pattern, graph))
    matches.push_back(m);

  ASSERT_EQ(matches.size(), 3u);

  // more explicit checks:
  bool abd = false, abe = false, acf = false;
  for (const auto &m : matches) {
    if (m[e1] == AB && m[e2] == BD && m[m_X] == A && m[m_Y] == B && m[m_Z] == D)
      abd = true;
    if (m[e1] == AB && m[e2] == BE && m[m_X] == A && m[m_Y] == B && m[m_Z] == E)
      abe = true;
    if (m[e1] == AC && m[e2] == CF && m[m_X] == A && m[m_Y] == C && m[m_Z] == F)
      acf = true;
  }
  EXPECT_TRUE(abd && abe && acf);
}

TEST(algorithm_match_all, edge_root_then_two_required_outgoing_from_dst) {
  // Root is edge A->B (value 9). B has two outgoing of rank 1 (to C and D).
  // Pattern requires both outgoing (distinct) from B -> expect 2! = 2 matches.
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  NodeId C = adj.addNode(3);
  NodeId D = adj.addNode(4);
  EdgeId AB = adj.addEdge(A, B, 9);
  EdgeId BC = adj.addEdge(B, C, 1);
  EdgeId BD = adj.addEdge(B, D, 1);

  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto eRoot = pattern.matchEdge();
  eRoot->matchValue([](const int &v) { return v == 9; });
  auto m_B = eRoot->matchDst();

  auto e1 = m_B->matchOutgoing();
  e1->matchRank(1);
  auto m_Y1 = e1->matchDst();
  auto e2 = m_B->matchOutgoing();
  e2->matchRank(1);
  auto m_Y2 = e2->matchDst();

  std::vector<ConstGraphMatch<int, int>> matches;
  for (const auto &m : match_all(pattern, graph))
    matches.push_back(m);

  ASSERT_EQ(matches.size(), 2u);
  auto hasCD = std::any_of(matches.begin(), matches.end(), [&](const auto &m) {
    return m[eRoot] == AB &&
           ((m[e1] == BC && m[e2] == BD && m[m_Y1] == C && m[m_Y2] == D) ||
            (m[e1] == BD && m[e2] == BC && m[m_Y1] == D && m[m_Y2] == C));
  });
  EXPECT_TRUE(hasCD);
}

TEST(algorithm_match_all, three_required_outgoing_more_than_enough_edges) {
  // A has 4 outgoing edges of rank 1. Require 3 distinct -> permutations 4P3 =
  // 24 matches.
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  NodeId C = adj.addNode(3);
  NodeId D = adj.addNode(4);
  NodeId E = adj.addNode(5);
  [[maybe_unused]] EdgeId AB = adj.addEdge(A, B, 1);
  [[maybe_unused]] EdgeId AC = adj.addEdge(A, C, 1);
  [[maybe_unused]] EdgeId AD = adj.addEdge(A, D, 1);
  [[maybe_unused]] EdgeId AE = adj.addEdge(A, E, 1);

  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](const int &v) { return v == 1; });

  auto e1 = m_X->matchOutgoing();
  e1->matchRank(1);
  auto m_Y1 = e1->matchDst();
  auto e2 = m_X->matchOutgoing();
  e2->matchRank(1);
  auto m_Y2 = e2->matchDst();
  auto e3 = m_X->matchOutgoing();
  e3->matchRank(1);
  auto m_Y3 = e3->matchDst();

  std::size_t count = 0;
  for (const auto &m : match_all(pattern, graph)) {
    (void)m;
    ++count;
  }
  EXPECT_EQ(count, 24u); // 4P3 = 24
}

TEST(algorithm_match_all, cycle_two_step_paths_start_any_node) {
  // 2-node cycle A<->B with rank-1 edges. Pattern: X->Y->Z (rank 1 both).
  // Starting at A or B yields exactly two matches: A->B->A and B->A->B.
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(1);
  EdgeId AB = adj.addEdge(A, B, 1);
  EdgeId BA = adj.addEdge(B, A, 1);

  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](const int &v) { return v == 1; });

  auto e1 = m_X->matchOutgoing();
  e1->matchRank(1);
  auto m_Y = e1->matchDst();

  auto e2 = m_Y->matchOutgoing();
  e2->matchRank(1);
  auto m_Z = e2->matchDst();

  std::vector<ConstGraphMatch<int, int>> matches;
  for (const auto &m : match_all(pattern, graph))
    matches.push_back(m);

  ASSERT_EQ(matches.size(), 2u);

  bool a_b_a = false, b_a_b = false;
  for (const auto &m : matches) {
    if (m[m_X] == A && m[m_Y] == B && m[m_Z] == A && m[e1] == AB && m[e2] == BA)
      a_b_a = true;
    if (m[m_X] == B && m[m_Y] == A && m[m_Z] == B && m[e1] == BA && m[e2] == AB)
      b_a_b = true;
  }
  EXPECT_TRUE(a_b_a && b_a_b);
}

template <typename G>
static const typename G::Edge *
insert_and_get(G &, const typename G::NodeHandle &src,
               const typename G::NodeHandle &dst, int payload) {
  // Insert A->B(payload) and return the concrete Edge*
  auto it_after = src->outgoing().insert(dst, payload); // iterator AFTER insert
  auto it = it_after;
  --it; // now points at the inserted edge
  return it.operator->();
}

TEST(algorithm_match_all, line_same_label_two_step_paths_count) {
  using G = denox::memory::AdjGraph<int, int>;
  using CG = denox::memory::ConstGraph<int, int>;

  // Build a linear chain of N nodes with edges labeled 1 or 2.
  const std::size_t N = 20000; // "a couple thousand"
  G adj;
  std::vector<denox::memory::NodeId> nodes;
  nodes.reserve(N);
  for (std::size_t i = 0; i < N; ++i) {
    nodes.push_back(adj.addNode(static_cast<int>(i))); // node values irrelevant
  }

  std::mt19937 rng(123456); // deterministic
  std::uniform_int_distribution<int> coin(1, 2);

  std::vector<int> labels;
  labels.reserve(N > 0 ? N - 1 : 0);
  for (std::size_t i = 0; i + 1 < N; ++i) {
    int v = coin(rng);
    labels.push_back(v);
    adj.addEdge(nodes[i], nodes[i + 1], v);
  }

  CG graph{adj};

  // Compute expected counts of two-step paths with equal labels.
  std::size_t expected11 = 0, expected22 = 0;
  for (std::size_t i = 0; i + 2 <= labels.size(); ++i) {
    if (labels[i] == 1 && labels[i + 1] == 1)
      ++expected11;
    if (labels[i] == 2 && labels[i + 1] == 2)
      ++expected22;
  }

  // Pattern: X --(rank 1, value==1)--> Y --(rank 1, value==1)--> Z
  denox::algorithm::GraphPattern<int, int> p11;
  auto X11 = p11.matchNode();
  auto e1_1 = X11->matchOutgoing();
  e1_1->matchRank(1);
  e1_1->matchValue([](const int &v) { return v == 1; });
  auto Y11 = e1_1->matchDst();
  auto e2_1 = Y11->matchOutgoing();
  e2_1->matchRank(1);
  e2_1->matchValue([](const int &v) { return v == 1; });
  auto Z11 = e2_1->matchDst();

  std::size_t got11 = 0;
  for (const auto &m : match_all(p11, graph)) {
    (void)m;
    ++got11;
  }
  EXPECT_EQ(got11, expected11);

  // Pattern: X --(rank 1, value==2)--> Y --(rank 1, value==2)--> Z
  denox::algorithm::GraphPattern<int, int> p22;
  auto X22 = p22.matchNode();
  auto e1_2 = X22->matchOutgoing();
  e1_2->matchRank(1);
  e1_2->matchValue([](const int &v) { return v == 2; });
  auto Y22 = e1_2->matchDst();
  auto e2_2 = Y22->matchOutgoing();
  e2_2->matchRank(1);
  e2_2->matchValue([](const int &v) { return v == 2; });
  auto Z22 = e2_2->matchDst();

  std::size_t got22 = 0;
  for (const auto &m : match_all(p22, graph)) {
    (void)m;
    ++got22;
  }
  EXPECT_EQ(got22, expected22);

  // Sanity: total equal-pair matches equals sum of both kinds.
  EXPECT_EQ(got11 + got22, expected11 + expected22);
}

TEST(algorithm_match_all_linked_graph, singular_node_match) {
  using LG = LinkedGraph<int, int>;
  LG g;

  auto A = g.createNode(1);
  auto B = g.createNode(2);
  auto C = g.createNode(1);
  // edges: A -> B (1), A -> C (1)
  A->outgoing().insert(B, 1);
  A->outgoing().insert(C, 1);

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](const int &v) { return v == 1; });

  std::size_t count = 0;
  bool hasA = false, hasC = false;

  // IMPORTANT: iterate directly; do not store matches.
  for (const auto &m : match_all(pattern, A)) {
    ++count;
    if (m[m_X] == A)
      hasA = true;
    if (m[m_X] == C)
      hasC = true;
  }

  ASSERT_EQ(count, 2u);
  EXPECT_TRUE(hasA);
  EXPECT_TRUE(hasC);
}

TEST(algorithm_match_all_linked_graph, multiple_matches_per_node) {
  using LG = LinkedGraph<int, int>;
  LG g;
  auto A = g.createNode(1);
  auto B = g.createNode(1);
  auto C = g.createNode(1);

  // Build edges and keep raw edge identities (Edge*)
  auto itAB = A->outgoing().insert(B, 1);
  auto itAC = A->outgoing().insert(C, 1);
  const LG::Edge *AB = itAB.operator->();
  const LG::Edge *AC = itAC.operator->();

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](const int &v) { return v == 1; });
  auto e = m_X->matchOutgoing();
  e->matchRank(1);
  auto m_Y = e->matchDst();

  std::size_t count = 0;
  bool hasAB = false, hasAC = false;

  for (const auto &m : match_all(pattern, A)) {
    ++count;
    if (m[m_X] == A && m[e].ptr() == AB && m[m_Y] == B)
      hasAB = true;
    if (m[m_X] == A && m[e].ptr() == AC && m[m_Y] == C)
      hasAC = true;
  }

  ASSERT_EQ(count, 2u);
  EXPECT_TRUE(hasAB && hasAC);
}

TEST(algorithm_match_all_linked_graph,
     two_required_outgoing_parallel_edges_same_dst) {
  using LG = LinkedGraph<int, int>;
  LG g;
  auto A = g.createNode(1);
  auto B = g.createNode(2);

  auto itAB1 = A->outgoing().insert(B, 1);
  auto itAB2 = A->outgoing().insert(B, 1);
  const LG::Edge *AB1 = itAB1.operator->();
  const LG::Edge *AB2 = itAB2.operator->();

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](const int &v) { return v == 1; });

  auto e1 = m_X->matchOutgoing();
  e1->matchRank(1);
  auto m_Y1 = e1->matchDst();
  auto e2 = m_X->matchOutgoing();
  e2->matchRank(1);
  auto m_Y2 = e2->matchDst();

  std::size_t count = 0;
  bool has12 = false, has21 = false;

  for (const auto &m : match_all(pattern, A)) {
    ++count;
    if (m[m_X] == A && m[e1].ptr() == AB1 && m[e2].ptr() == AB2 &&
        m[m_Y1] == B && m[m_Y2] == B)
      has12 = true;
    if (m[m_X] == A && m[e1].ptr() == AB2 && m[e2].ptr() == AB1 &&
        m[m_Y1] == B && m[m_Y2] == B)
      has21 = true;
  }

  ASSERT_EQ(count, 2u);
  EXPECT_TRUE(has12 || has21);
}

TEST(algorithm_match_all_linked_graph,
     two_required_outgoing_edges_from_same_node) {
  using LG = LinkedGraph<int, int>;
  LG g;
  auto A = g.createNode(1);
  auto B = g.createNode(2);
  auto C = g.createNode(3);

  auto itAB = A->outgoing().insert(B, 1);
  auto itAC = A->outgoing().insert(C, 1);
  const LG::Edge *AB = itAB.operator->();
  const LG::Edge *AC = itAC.operator->();

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](const int &v) { return v == 1; });

  auto e1 = m_X->matchOutgoing();
  e1->matchRank(1);
  auto m_Y1 = e1->matchDst();
  auto e2 = m_X->matchOutgoing();
  e2->matchRank(1);
  auto m_Y2 = e2->matchDst();

  std::size_t count = 0;
  bool case1 = false, case2 = false;

  for (const auto &m : match_all(pattern, A)) {
    ++count;
    if (m[m_X] == A && m[e1].ptr() == AB && m[m_Y1] == B && m[e2].ptr() == AC &&
        m[m_Y2] == C)
      case1 = true;
    if (m[m_X] == A && m[e1].ptr() == AC && m[m_Y1] == C && m[e2].ptr() == AB &&
        m[m_Y2] == B)
      case2 = true;
  }

  ASSERT_EQ(count, 2u);
  EXPECT_TRUE(case1 || case2);
}

TEST(algorithm_match_all_linked_graph, nested_two_step_paths_from_root_node) {
  using LG = LinkedGraph<int, int>;
  LG g;
  auto A = g.createNode(1);
  auto B = g.createNode(2);
  auto C = g.createNode(3);
  auto D = g.createNode(4);
  auto E = g.createNode(5);
  auto F = g.createNode(6);

  auto itAB = A->outgoing().insert(B, 1);
  auto itAC = A->outgoing().insert(C, 1);
  auto itBD = B->outgoing().insert(D, 1);
  auto itBE = B->outgoing().insert(E, 1);
  auto itCF = C->outgoing().insert(F, 1);
  const LG::Edge *AB = itAB.operator->();
  const LG::Edge *AC = itAC.operator->();
  const LG::Edge *BD = itBD.operator->();
  const LG::Edge *BE = itBE.operator->();
  const LG::Edge *CF = itCF.operator->();

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](const int &v) { return v == 1; });

  auto e1 = m_X->matchOutgoing();
  e1->matchRank(1);
  auto m_Y = e1->matchDst();
  auto e2 = m_Y->matchOutgoing();
  e2->matchRank(1);
  auto m_Z = e2->matchDst();

  std::size_t count = 0;
  bool abd = false, abe = false, acf = false;

  for (const auto &m : match_all(pattern, A)) {
    ++count;
    if (m[e1].ptr() == AB && m[e2].ptr() == BD && m[m_X] == A && m[m_Y] == B &&
        m[m_Z] == D)
      abd = true;
    if (m[e1].ptr() == AB && m[e2].ptr() == BE && m[m_X] == A && m[m_Y] == B &&
        m[m_Z] == E)
      abe = true;
    if (m[e1].ptr() == AC && m[e2].ptr() == CF && m[m_X] == A && m[m_Y] == C &&
        m[m_Z] == F)
      acf = true;
  }

  ASSERT_EQ(count, 3u);
  EXPECT_TRUE(abd && abe && acf);
}

//
// 1) Node value + out-degree filtering (K==0 path)
//    Only A has outdeg==2 and value==1; B,C don’t match.
//
TEST(algorithm_match_all_linked_graph, node_value_and_outdeg_filter) {
  using LG = LinkedGraph<int, int>;
  LG g;
  auto A = g.createNode(1);
  auto B = g.createNode(1);
  auto C = g.createNode(1);
  auto D = g.createNode(0);

  A->outgoing().insert(B, 7);
  A->outgoing().insert(C, 7);
  B->outgoing().insert(D, 7); // B outdeg=1
  // C outdeg=0

  GraphPattern<int, int> p;
  auto X = p.matchNode();
  X->matchValue([](const int &v) { return v == 1; });
  X->matchOutDeg(2);

  std::size_t count = 0;
  bool sawA = false, sawB = false, sawC = false;
  for (const auto &m : match_all(p, A)) {
    ++count;
    if (m[X] == A)
      sawA = true;
    if (m[X] == B)
      sawB = true;
    if (m[X] == C)
      sawC = true;
  }
  ASSERT_EQ(count, 1u);
  EXPECT_TRUE(sawA);
  EXPECT_FALSE(sawB);
  EXPECT_FALSE(sawC);
}

//
// 2) In-degree filtering on a descendant (K==0 at that node).
//    D has indeg==2 (A->D, B->D); only D should match.
//
TEST(algorithm_match_all_linked_graph, indeg_filter_on_descendant) {
  using LG = LinkedGraph<int, int>;
  LG g;
  auto A = g.createNode(1);
  auto B = g.createNode(9);
  auto D = g.createNode(3);

  A->outgoing().insert(D, 1);
  B->outgoing().insert(D, 1); // D indeg==2

  GraphPattern<int, int> p;
  auto X = p.matchNode();
  X->matchValue([](const int &v) { return v == 3; });
  X->matchInDeg(2);

  std::size_t count = 0;
  bool sawD = false;
  for (const auto &m : match_all(p, A)) {
    ++count;
    if (m[X] == D)
      sawD = true;
  }
  ASSERT_EQ(count, 1u);
  EXPECT_TRUE(sawD);
}

//
// 3) Rank-2 hyperedge: dst has a single edge with two sources (S1,S2)->D.
//    From S1, require rank==2 and dst value==3 → exactly one match.
//
TEST(algorithm_match_all_linked_graph, rank2_hyperedge_single_match) {
  using LG = LinkedGraph<int, int>;
  LG g;
  auto S1 = g.createNode(1);
  auto S2 = g.createNode(8);
  auto D = g.createNode(3);

  // Create one rank-2 edge into D from S1 and S2.
  auto it = D->incoming().insert(S1, S2, /*payload*/ 42);
  (void)it;

  GraphPattern<int, int> p;
  auto X = p.matchNode();
  X->matchValue([](const int &v) { return v == 1; });

  auto e = X->matchOutgoing();
  e->matchRank(2);
  auto Y = e->matchDst();
  Y->matchValue([](const int &v) { return v == 3; });

  std::size_t count = 0;
  bool matched = false;
  for (const auto &m : match_all(p, S1)) {
    ++count;
    if (m[X] == S1 && m[Y] == D)
      matched = true;
  }
  ASSERT_EQ(count, 1u);
  EXPECT_TRUE(matched);
}

//
// 4) Nested 3-step path: A->B->C->D with rank==1 edges.
//    Require X->Y (rank1), Y->Z (rank1) → exactly one match: A,B,C.
//
TEST(algorithm_match_all_linked_graph, nested_three_step_path_single) {
  using LG = LinkedGraph<int, int>;
  LG g;
  auto A = g.createNode(1);
  auto B = g.createNode(2);
  auto C = g.createNode(3);
  auto D = g.createNode(4);

  A->outgoing().insert(B, 1);
  B->outgoing().insert(C, 1);
  C->outgoing().insert(D, 1);

  GraphPattern<int, int> p;
  auto X = p.matchNode();
  X->matchValue([](const int &v) { return v == 1; });

  auto e1 = X->matchOutgoing();
  e1->matchRank(1);
  auto Y = e1->matchDst();
  auto e2 = Y->matchOutgoing();
  e2->matchRank(1);
  auto Z = e2->matchDst();

  std::size_t count = 0;
  bool saw = false;
  for (const auto &m : match_all(p, A)) {
    ++count;
    if (m[X] == A && m[Y] == B && m[Z] == C)
      saw = true;
  }
  ASSERT_EQ(count, 1u);
  EXPECT_TRUE(saw);
}

//
// 5) Child filtering: A has two rank-1 edges A->B, A->C,
//    but only B continues to a child that matches Y->Z (value filter).
//    Expect exactly one match.
//
TEST(algorithm_match_all_linked_graph, child_filtering_selects_one_branch) {
  using LG = LinkedGraph<int, int>;
  LG g;
  auto A = g.createNode(1);
  auto B = g.createNode(10);
  auto C = g.createNode(20);
  auto D = g.createNode(99);
  auto E = g.createNode(77);

  A->outgoing().insert(B, 1);
  A->outgoing().insert(C, 1);
  B->outgoing().insert(D, 1); // D passes child dst filter
  C->outgoing().insert(E, 1); // E fails child dst filter

  GraphPattern<int, int> p;
  auto X = p.matchNode();
  X->matchValue([](const int &v) { return v == 1; });

  auto e1 = X->matchOutgoing();
  e1->matchRank(1);
  auto Y = e1->matchDst();
  auto e2 = Y->matchOutgoing();
  e2->matchRank(1);
  auto Z = e2->matchDst();
  Z->matchValue([](const int &v) { return v == 99; }); // Only D==99

  std::size_t count = 0;
  bool saw = false;
  for (const auto &m : match_all(p, A)) {
    ++count;
    if (m[X] == A && m[Y] == B && m[Z] == D)
      saw = true;
  }
  ASSERT_EQ(count, 1u);
  EXPECT_TRUE(saw);
}

//
// 6) Weight predicate (use W=int). A has two edges with different weights.
//    Match only weight==7.
//
TEST(algorithm_match_all_linked_graph, edge_weight_predicate) {
  using LG = LinkedGraph<int, int, int>; // W=int
  LG g;
  auto A = g.createNode(1);
  auto B = g.createNode(2);
  auto C = g.createNode(3);

  A->outgoing().insert(B, /*weight*/ 7, /*payload*/ 111);
  A->outgoing().insert(C, /*weight*/ 9, /*payload*/ 111);

  GraphPattern<int, int, int> p;
  auto X = p.matchNode();
  X->matchValue([](const int &v) { return v == 1; });

  auto e = X->matchOutgoing();
  e->matchRank(1);
  e->matchWeight([](const int &w) { return w == 7; });
  auto Y = e->matchDst();

  std::size_t count = 0;
  bool saw = false;
  for (const auto &m : match_all(p, A)) {
    ++count;
    if (m[X] == A && m[Y] == B)
      saw = true;
  }
  ASSERT_EQ(count, 1u);
  EXPECT_TRUE(saw);
}

//
// 7) Unreachable nodes do not yield matches.
//    There’s a disconnected component with value==1, but starting at A
//    must NOT enumerate it.
//
TEST(algorithm_match_all_linked_graph, unreachable_nodes_are_not_enumerated) {
  using LG = LinkedGraph<int, int>;
  LG g;
  auto A = g.createNode(1);
  auto B = g.createNode(2);
  auto U = g.createNode(1); // unreachable from A

  A->outgoing().insert(B, 5);
  // U has no edges and is disconnected.

  GraphPattern<int, int> p;
  auto X = p.matchNode();
  X->matchValue([](const int &v) { return v == 1; });

  std::size_t count = 0;
  bool sawA = false, sawU = false;
  for (const auto &m : match_all(p, A)) {
    ++count;
    if (m[X] == A)
      sawA = true;
    if (m[X] == U)
      sawU = true;
  }
  ASSERT_EQ(count, 1u);
  EXPECT_TRUE(sawA);
  EXPECT_FALSE(sawU);
}

//
// 8) Out-degree filter combined with a required outgoing edge.
//    A has outdeg==2 and both 1-rank edges; pattern also requires one outgoing.
//    Expect both A->B and A->C matches, but only from A (B/C filtered by
//    outdeg).
//
TEST(algorithm_match_all_linked_graph,
     outdeg_filter_with_one_required_outgoing) {
  using LG = LinkedGraph<int, int>;
  LG g;
  auto A = g.createNode(1);
  auto B = g.createNode(1);
  auto C = g.createNode(1);

  auto itAB = A->outgoing().insert(B, 1);
  auto itAC = A->outgoing().insert(C, 1);
  const LG::Edge *AB = itAB.operator->();
  const LG::Edge *AC = itAC.operator->();

  GraphPattern<int, int> p;
  auto X = p.matchNode();
  X->matchValue([](const int &v) { return v == 1; });
  X->matchOutDeg(2); // filters to A only

  auto e = X->matchOutgoing();
  e->matchRank(1);
  auto Y = e->matchDst();

  std::size_t count = 0;
  bool sawAB = false, sawAC = false;
  for (const auto &m : match_all(p, A)) {
    ++count;
    if (m[X] == A && m[Y] == B && m[e].ptr() == AB)
      sawAB = true;
    if (m[X] == A && m[Y] == C && m[e].ptr() == AC)
      sawAC = true;
  }
  ASSERT_EQ(count, 2u);
  EXPECT_TRUE(sawAB && sawAC);
}

// Add to your LinkedGraph test file

TEST(algorithm_match_all_linked_graph,
     mutate_two_required_outgoing_erase_e1_after_first_yield) {
  using LG = LinkedGraph<int, int>;
  LG g;

  auto A = g.createNode(1);
  auto B = g.createNode(2);
  auto C = g.createNode(3);

  A->outgoing().insert(B, 1);
  A->outgoing().insert(C, 1);

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](const int &v) { return v == 1; });

  auto e1 = m_X->matchOutgoing();
  e1->matchRank(1);
  auto m_Y1 = e1->matchDst();
  (void)m_Y1;
  auto e2 = m_X->matchOutgoing();
  e2->matchRank(1);
  auto m_Y2 = e2->matchDst();
  (void)m_Y2;

  std::size_t count = 0;
  bool erased_once = false;

  for (const auto &m : match_all(pattern, A)) {
    ++count;
    if (!erased_once) {
      // Remove the concrete edge chosen for e1 in the FIRST match.
      m[e1].erase();
      erased_once = true;
    }
  }

  // With only two outgoing edges, removing one immediately eliminates all
  // remaining permutations -> exactly one match total (the first one).
  EXPECT_EQ(count, 1u);
}

TEST(algorithm_match_all_linked_graph,
     mutate_nested_erase_parent_e1_after_first_yield) {
  using LG = LinkedGraph<int, int>;
  LG g;

  auto A = g.createNode(1);
  auto B = g.createNode(2);
  auto C = g.createNode(3);
  auto D = g.createNode(4);
  auto E = g.createNode(5);
  auto F = g.createNode(6);

  // A->{B,C}, B->{D,E}, C->{F}
  A->outgoing().insert(B, 1);
  A->outgoing().insert(C, 1);
  B->outgoing().insert(D, 1);
  B->outgoing().insert(E, 1);
  C->outgoing().insert(F, 1);

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](const int &v) { return v == 1; });

  auto e1 = m_X->matchOutgoing();
  e1->matchRank(1);
  auto m_Y = e1->matchDst();

  auto e2 = m_Y->matchOutgoing();
  e2->matchRank(1);
  auto m_Z = e2->matchDst();

  std::size_t count = 0;
  bool erased_once = false;

  // For validation: ensure no later match uses the erased parent edge
  const LG::Edge *erased_parent_edge = nullptr;
  bool reused_erased_parent = false;

  for (const auto &m : match_all(pattern, A)) {
    ++count;

    if (!erased_once) {
      erased_parent_edge = m[e1].ptr(); // capture identity, then erase
      m[e1].erase();
      erased_once = true;
      continue; // don't inspect further fields from this 'm' after erasing
    }

    // Subsequent matches:
    if (m[e1].ptr() == erased_parent_edge)
      reused_erased_parent = true;
    if (m[m_Y] != B && m[m_Y] != C)
      GTEST_FAIL() << "Unexpected Y node";
    // If we erased AB, we should still see the C-branch (A->C->F).
    // If we erased AC, we should still see the B-branch (A->B->D/E).
    // Detect "other branch" by comparing Y with the parent of the erased edge.
    // We can infer it by checking which branch still appears:
    //   If erased_parent_edge belonged to A->B, then seeing Y==C proves "other
    //   branch". If it belonged to A->C, then seeing Y==B proves "other
    //   branch".
    // We can’t dereference erased edge; just use simple heuristic:
    if (m[m_Y] == B || m[m_Y] == C) {
      // We can't know which parent was erased without extra bookkeeping;
      // it's enough to require that exactly one more match appears after erase.
      // We'll assert count==2 below, which implies we saw exactly one other
      // branch.
      (void)0;
    }
  }

  // Before erase: there are 3 matches total (ABD, ABE, ACF).
  // After erasing the chosen parent edge in the first yield, only the
  // opposite branch remains -> exactly one additional match.
  EXPECT_EQ(count, 2u);
  EXPECT_FALSE(reused_erased_parent);
}

TEST(algorithm_match_all_linked_graph,
     mutate_nested_erase_child_e2_after_first_yield) {
  using LG = LinkedGraph<int, int>;
  LG g;

  auto A = g.createNode(1);
  auto B = g.createNode(2);
  auto C = g.createNode(3);
  auto D = g.createNode(4);
  auto E = g.createNode(5);
  auto F = g.createNode(6);

  // A->{B,C}, B->{D,E}, C->{F}
  A->outgoing().insert(B, 1);
  A->outgoing().insert(C, 1);
  B->outgoing().insert(D, 1);
  B->outgoing().insert(E, 1);
  C->outgoing().insert(F, 1);

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](const int &v) { return v == 1; });

  auto e1 = m_X->matchOutgoing();
  e1->matchRank(1);
  auto m_Y = e1->matchDst();

  auto e2 = m_Y->matchOutgoing();
  e2->matchRank(1);
  auto m_Z = e2->matchDst();
  (void)m_Z;

  std::size_t count = 0;
  bool erased_once = false;

  const LG::Edge *erased_child_edge = nullptr;
  denox::memory::LinkedGraph<int, int>::NodeHandle
      erased_parent; // Y of first match
  bool reused_erased_child = false;
  bool saw_sibling_under_same_parent = false;
  bool saw_other_parent_branch = false;

  for (const auto &m : match_all(pattern, A)) {
    ++count;

    if (!erased_once) {
      erased_parent = m[m_Y];          // capture which Y we were on
      erased_child_edge = m[e2].ptr(); // child's identity
      m[e2].erase();                   // erase only the child edge
      erased_once = true;
      continue; // don't use 'm' after erasing
    }

    // After erasing a child on the chosen parent, we should still
    // see the sibling child on the same parent (if it exists), and
    // also matches from the other parent branch.
    if (m[e2].ptr() == erased_child_edge)
      reused_erased_child = true;

    if (m[m_Y] == erased_parent) {
      // Same parent, child must be the *other* edge
      saw_sibling_under_same_parent = true;
    } else {
      // Other branch (A->C->F or A->B->D/E depending on what was first)
      saw_other_parent_branch = true;
    }
  }

  // Original total is 3 matches; we erased only one child on the first yield,
  // so the two other matches must still appear. Total 3 results.
  EXPECT_EQ(count, 3u);
  EXPECT_FALSE(reused_erased_child);
  EXPECT_TRUE(saw_sibling_under_same_parent);
  EXPECT_TRUE(saw_other_parent_branch);
}

TEST(algorithm_match_all_linked_graph,
     mutate_two_required_from_three_outgoings_erase_first_choice) {
  using LG = LinkedGraph<int, int>;
  LG g;

  auto A = g.createNode(1);
  auto B = g.createNode(2);
  auto C = g.createNode(3);
  auto D = g.createNode(4);

  auto itAB = A->outgoing().insert(B, 1);
  auto itAC = A->outgoing().insert(C, 1);
  auto itAD = A->outgoing().insert(D, 1);
  const LG::Edge *AB = itAB.operator->();
  const LG::Edge *AC = itAC.operator->();
  const LG::Edge *AD = itAD.operator->();

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](const int &v) { return v == 1; });

  auto e1 = m_X->matchOutgoing();
  e1->matchRank(1);
  auto m_Y1 = e1->matchDst();
  (void)m_Y1;
  auto e2 = m_X->matchOutgoing();
  e2->matchRank(1);
  auto m_Y2 = e2->matchDst();
  (void)m_Y2;

  std::size_t count = 0;
  bool erased_once = false;

  // Track which two remaining edges should form the two permutations
  const LG::Edge *erased_edge = nullptr;
  const LG::Edge *remA = nullptr;
  const LG::Edge *remB = nullptr;
  bool saw_remA_remB = false;
  bool saw_remB_remA = false;
  bool saw_erased_in_later_match = false;

  for (const auto &m : match_all(pattern, A)) {
    ++count;

    if (!erased_once) {
      erased_edge = m[e1].ptr();
      // Compute the remaining two pointers
      if (erased_edge == AB) {
        remA = AC;
        remB = AD;
      } else if (erased_edge == AC) {
        remA = AB;
        remB = AD;
      } else {
        remA = AB;
        remB = AC;
      }
      m[e1].erase(); // remove the first chosen edge
      erased_once = true;
      continue; // don't use 'm' after erase
    }

    // Subsequent matches must not use the erased edge at all
    if (m[e1].ptr() == erased_edge || m[e2].ptr() == erased_edge) {
      saw_erased_in_later_match = true;
    }

    // Check that we see exactly the two permutations from the remaining pair
    if (m[e1].ptr() == remA && m[e2].ptr() == remB)
      saw_remA_remB = true;
    if (m[e1].ptr() == remB && m[e2].ptr() == remA)
      saw_remB_remA = true;
  }

  // Initially 3P2 = 6 permutations. We yielded one, then erased one of the
  // three edges; exactly the two permutations of the surviving pair remain => 3
  // total.
  EXPECT_EQ(count, 3u);
  EXPECT_FALSE(saw_erased_in_later_match);
  EXPECT_TRUE(saw_remA_remB);
  EXPECT_TRUE(saw_remB_remA);
}

TEST(algorithm_match_all_linked_graph,
     mutate_nested_erase_child_e2_after_first_yield_2) {
  using LG = LinkedGraph<int, int>;
  LG g;

  auto A = g.createNode(1), B = g.createNode(2), C = g.createNode(3);
  auto D = g.createNode(4), E = g.createNode(5), F = g.createNode(6);
  A->outgoing().insert(B, 1);
  A->outgoing().insert(C, 1);
  B->outgoing().insert(D, 1);
  B->outgoing().insert(E, 1);
  C->outgoing().insert(F, 1);

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](int v) { return v == 1; });
  auto e1 = m_X->matchOutgoing();
  e1->matchRank(1);
  auto m_Y = e1->matchDst();
  auto e2 = m_Y->matchOutgoing();
  e2->matchRank(1);
  auto m_Z = e2->matchDst();

  std::size_t count = 0;
  bool erased_once = false;
  const LG::Edge *erased_child = nullptr;
  bool reused_erased = false;

  for (const auto &m : match_all(pattern, A)) {
    ++count;

    if (!erased_once) {
      erased_child = m[e2].ptr();
      m[e2].erase(); // erase child edge (Y -> Z) of the first match
      erased_once = true;
      continue; // do not touch 'm' further after erase
    }

    if (m[e2].ptr() == erased_child)
      reused_erased = true;
  }

  EXPECT_EQ(count, 3u); // first + two remaining
  EXPECT_FALSE(reused_erased);
}

TEST(algorithm_match_all_linked_graph,
     mutate_nested_insert_new_child_after_first_yield) {
  using LG = LinkedGraph<int, int>;
  LG g;

  auto A = g.createNode(1), B = g.createNode(2), C = g.createNode(3);
  auto D = g.createNode(4), E = g.createNode(5), F = g.createNode(6),
       G = g.createNode(7);
  A->outgoing().insert(B, 1);
  A->outgoing().insert(C, 1);
  B->outgoing().insert(D, 1);
  B->outgoing().insert(E, 1);
  C->outgoing().insert(F, 1);

  GraphPattern<int, int> pattern;
  auto m_X = pattern.matchNode();
  m_X->matchValue([](int v) { return v == 1; });
  auto e1 = m_X->matchOutgoing();
  e1->matchRank(1);
  auto m_Y = e1->matchDst();
  auto e2 = m_Y->matchOutgoing();
  e2->matchRank(1);
  auto m_Z = e2->matchDst();

  std::size_t count = 0;
  bool inserted_once = false;
  bool saw_G = false;

  for (const auto &m : match_all(pattern, A)) {
    ++count;

    if (!inserted_once) {
      // Insert the new child deterministically: after the first outgoing edge.
      auto out = m[m_Y]->outgoing();
      auto em = m[e2];
      out.insert_after(em.nextOutgoingIterator(), G, 1);
      inserted_once = true;
      continue; // don’t read more from 'm' after mutating
    }

    if (m[m_Z] == G)
      saw_G = true;
  }

  EXPECT_EQ(count, 4u); // one extra match from the new child
  EXPECT_TRUE(saw_G);
}

TEST(algorithm_match_all_linked_graph,
     minimize_random_line_by_collapsing_runs) {
  using LG = LinkedGraph<int, int>;
  LG g;

  // Build a long line A0 -> A1 -> ... -> A(N-1)
  const std::size_t N = 6000; // "a couple of thousands"
  std::vector<LG::NodeHandle> nodes;
  nodes.reserve(N);
  for (std::size_t i = 0; i < N; ++i) {
    // Node payload is irrelevant for the rule; just store any ints
    nodes.push_back(g.createNode(static_cast<int>(i + 1)));
  }

  // Randomly label each edge with value 1 or 2
  std::mt19937 rng(123456u);
  std::uniform_int_distribution<int> d12(1, 2);
  for (std::size_t i = 0; i + 1 < N; ++i) {
    const int v = d12(rng);
    // Building the initial line: order within the adjacency is not important
    nodes[i]->outgoing().insert(nodes[i + 1], v);
  }

  // Keep only a handle to the root; release all intermediate nodes to enable
  // LinkedGraph's self-cleanup on cascades.
  LG::NodeHandle root = nodes[0];
  for (std::size_t i = 1; i < N; ++i)
    nodes[i].release();
  nodes.clear();

  // Pattern “two consecutive edges with value == 1”
  GraphPattern<int, int> p1;
  auto X1 = p1.matchNode();
  auto e1v1 = X1->matchOutgoing();
  e1v1->matchRank(1);
  e1v1->matchValue([](int v) { return v == 1; });
  auto Y1 = e1v1->matchDst();
  auto e2v1 = Y1->matchOutgoing();
  e2v1->matchRank(1);
  e2v1->matchValue([](int v) { return v == 1; });
  auto Z1 = e2v1->matchDst();

  // Pattern “two consecutive edges with value == 2”
  GraphPattern<int, int> p2;
  auto X2 = p2.matchNode();
  auto e1v2 = X2->matchOutgoing();
  e1v2->matchRank(1);
  e1v2->matchValue([](int v) { return v == 2; });
  auto Y2 = e1v2->matchDst();
  auto e2v2 = Y2->matchOutgoing();
  e2v2->matchRank(1);
  e2v2->matchValue([](int v) { return v == 2; });
  auto Z2 = e2v2->matchDst();

  // Apply the rewrite rule to a fixpoint:
  // For a match X --v--> Y --v--> Z:
  //   insert X --v--> Z   (after the "next" position of e1)
  //   erase e1
  // We never directly erase the second edge; if Y dies, LinkedGraph will
  // cascade-clean B->C as needed (we only hold 'root' externally).
  auto compress_pass = [&](auto &pat, auto e1H, auto ZH,
                           int vlabel) -> std::size_t {
    std::size_t changes = 0;
    for (const auto &m : match_all(pat, root)) {
      auto src = m[e1H].sourceNode();
      auto dst = m[ZH]; // pin destination before mutating
      auto itNext = m[e1H].nextOutgoingIterator();
      src->outgoing().insert_after(itNext, dst, vlabel);
      m[e1H].erase();
      ++changes;
      // Do not read anything else from 'm' after mutating.
    }
    return changes;
  };

  // Iterate until no changes from either pattern.
  while (true) {
    std::size_t c1 = compress_pass(p1, e1v1, Z1, 1);
    std::size_t c2 = compress_pass(p2, e1v2, Z2, 2);
    if (c1 == 0 && c2 == 0)
      break;
  }

  // Verify result: the reachable structure is still a single line
  // and edge values strictly alternate (no equal-adjacent labels).
  auto cur = root;
  int prev = -1;
  std::size_t steps = 0;
  while (true) {
    auto out = cur->outgoing();
    std::size_t deg = out.size();
    if (deg == 0)
      break; // end of the line
    ASSERT_EQ(deg, 1u) << "Graph is not a simple line after minimization";
    auto it = out.begin();
    const int val = it->value();
    if (prev != -1) {
      ASSERT_NE(val, prev) << "Found two consecutive edges with the same value";
    }
    prev = val;
    cur = it->dst();
    ++steps;
    ASSERT_LT(steps, N + 10) << "Unexpectedly long chain (possible cycle)";
  }

  // The chain got shorter or stayed similar, but must be finite and valid.
  EXPECT_GE(N - 1, steps);
  // Having done at least one rewrite in most random inputs is expected,
  // but not guaranteed (a fully alternating initial line yields 0 changes).
  // So we don't assert total_changes > 0 here.
}

// Helper for readability
template <typename V, typename E>
static std::vector<ConstGraphMatch<V, E>>
collect_all(const GraphPattern<V, E> &pat, const ConstGraph<V, E> &g) {
  std::vector<ConstGraphMatch<V, E>> out;
  for (const auto &m : match_all(pat, g))
    out.push_back(m);
  return out;
}

TEST(algorithm_match_all, node_single_incoming) {
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  EdgeId AB = adj.addEdge(A, B, 7); // single-source

  ConstGraph<int, int> g{adj};

  GraphPattern<int, int> pat;
  auto X = pat.matchNode(); // root node
  X->matchValue([](int v) { return v == 2; });

  auto e_in = X->matchIncoming(); // require one incoming
  e_in->matchValue([](int e) { return e == 7; });
  auto S0 = e_in->matchSrc(0); // constrain the (only) source
  S0->matchValue([](int v) { return v == 1; });

  auto Ms = collect_all(pat, g);
  ASSERT_EQ(Ms.size(), 1u);
  EXPECT_EQ(Ms[0][X], B);
  EXPECT_EQ(Ms[0][e_in], AB);
  EXPECT_EQ(Ms[0][S0], A);
}

TEST(algorithm_match_all,
     node_two_incoming_permutations) {
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId C = adj.addNode(3);
  NodeId B = adj.addNode(2);
  EdgeId AB = adj.addEdge(A, B, 5);
  EdgeId CB = adj.addEdge(C, B, 6);

  ConstGraph<int, int> g{adj};

  GraphPattern<int, int> pat;
  auto X = pat.matchNode();
  X->matchValue([](int v) { return v == 2; });

  auto e1 = X->matchIncoming();
  e1->matchValue([](int e) { return e == 5; });
  auto S1 = e1->matchSrc(0);
  S1->matchValue([](int v) { return v == 1; });

  auto e2 = X->matchIncoming();
  e2->matchValue([](int e) { return e == 6; });
  auto S2 = e2->matchSrc(0);
  S2->matchValue([](int v) { return v == 3; });

  auto Ms = collect_all(pat, g);
  ASSERT_EQ(Ms.size(), 1u); // e1/e2 or e2/e1

  // Just verify both bindings appear among permutations
  bool has_15_36 = false;
  for (const auto &m : Ms) {
    has_15_36 |=
        (m[X] == B) &&
        ((m[e1] == AB && m[e2] == CB) || (m[e1] == CB && m[e2] == AB)) &&
        m[S1] == A && m[S2] == C;
  }
  EXPECT_TRUE(has_15_36);
}

TEST(algorithm_match_all,
     edge_root_two_ordered_sources_and_dst) {
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  NodeId D = adj.addNode(4);

  // Add a hyperedge with sources (A,B) -> D, value 9.
  // If your AdjGraph overload is different, adapt accordingly.
  EdgeId ABD = adj.addEdge(A, B, D, 9);

  ConstGraph<int, int> g{adj};

  GraphPattern<int, int> pat;
  auto e = pat.matchEdge();
  e->matchValue([](int e) { return e == 9; });
  auto SA = e->matchSrc(0);
  SA->matchValue([](int v) { return v == 1; });
  auto SB = e->matchSrc(1);
  SB->matchValue([](int v) { return v == 2; });
  auto Y = e->matchDst();
  Y->matchValue([](int v) { return v == 4; });

  auto Ms = collect_all(pat, g);
  ASSERT_EQ(Ms.size(), 1u);
  EXPECT_EQ(Ms[0][e], ABD);
  EXPECT_EQ(Ms[0][SA], A);
  EXPECT_EQ(Ms[0][SB], B);
  EXPECT_EQ(Ms[0][Y], D);
}

TEST(algorithm_match_all, edge_root_two_sources_wrong_order_no_match) {
  AdjGraph<int,int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  NodeId D = adj.addNode(4);
  adj.addEdge(A,B, D, 9);

  ConstGraph<int,int> g{adj};

  GraphPattern<int,int> pat;
  auto e = pat.matchEdge();           e->matchValue([](int e){ return e==9; });
  auto SA = e->matchSrc(0);           SA->matchValue([](int v){ return v==2; }); // swapped
  auto SB = e->matchSrc(1);           SB->matchValue([](int v){ return v==1; }); // swapped
  auto Y  = e->matchDst();            Y->matchValue([](int v){ return v==4; });

  auto Ms = collect_all(pat, g);
  EXPECT_TRUE(Ms.empty());
}

TEST(algorithm_match_all, node_incoming_then_outgoing_chain) {
  AdjGraph<int,int> adj;
  NodeId A = adj.addNode(1);
  NodeId X = adj.addNode(10);
  NodeId C = adj.addNode(3);
  EdgeId AX = adj.addEdge(A, X, 5);
  EdgeId XC = adj.addEdge(X, C, 7);

  ConstGraph<int,int> g{adj};

  GraphPattern<int,int> pat;
  auto NX  = pat.matchNode(); NX->matchValue([](int v){ return v==10; });

  auto eIn  = NX->matchIncoming(); eIn->matchValue([](int e){ return e==5; });
  auto S0   = eIn->matchSrc(0);    S0->matchValue([](int v){ return v==1; });

  auto eOut = NX->matchOutgoing(); eOut->matchValue([](int e){ return e==7; });
  auto NY   = eOut->matchDst();    NY->matchValue([](int v){ return v==3; });

  auto Ms = collect_all(pat, g);
  ASSERT_EQ(Ms.size(), 1u);
  EXPECT_EQ(Ms[0][NX], X);
  EXPECT_EQ(Ms[0][eIn], AX);
  EXPECT_EQ(Ms[0][S0], A);
  EXPECT_EQ(Ms[0][eOut], XC);
  EXPECT_EQ(Ms[0][NY], C);
}

TEST(algorithm_match_all, node_incoming_rank_mismatch) {
  AdjGraph<int,int> adj;
  NodeId A = adj.addNode(1);
  NodeId X = adj.addNode(2);
  (void)adj.addEdge(A, X, 11); // only one incoming

  ConstGraph<int,int> g{adj};

  GraphPattern<int,int> pat;
  auto NX = pat.matchNode(); NX->matchValue([](int v){ return v==2; });

  auto e1 = NX->matchIncoming(); e1->matchValue([](int){ return true; });
  auto e2 = NX->matchIncoming(); e2->matchValue([](int){ return true; });

  auto Ms = collect_all(pat, g);
  EXPECT_TRUE(Ms.empty());
}

// 7) Edge root: sparse source constraints (only constrain src[1]).
TEST(algorithm_match_all, edge_root_sparse_src_constraint) {
  AdjGraph<int,int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  NodeId C = adj.addNode(3);
  NodeId D = adj.addNode(4);

  // Two candidate hyperedges to the same dst, both value==9:
  EdgeId ABD = adj.addEdge(A,B, D, 9);
  [[maybe_unused]] EdgeId ACD = adj.addEdge(A,C, D, 9);

  ConstGraph<int,int> g{adj};

  GraphPattern<int,int> pat;
  auto e = pat.matchEdge();  e->matchValue([](int e){ return e==9; });
  // Only constrain src[1] to be '2' ⇒ picks ABD only.
  auto S1 = e->matchSrc(1);  S1->matchValue([](int v){ return v==2; });
  auto Y  = e->matchDst();   Y->matchValue([](int v){ return v==4; });

  auto Ms = collect_all(pat, g);
  ASSERT_EQ(Ms.size(), 1u);
  EXPECT_EQ(Ms[0][e], ABD);
  EXPECT_EQ(Ms[0][S1], B);
  EXPECT_EQ(Ms[0][Y], D);
}


TEST(algorithm_match_all, edge_root_concat_ordered_two_srcs) {
  AdjGraph<int,int> adj;
  // Inputs 1 and 3 feeding concat -> output 2
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(3);
  NodeId C = adj.addNode(2);

  // NOTE: adjust to your AdjGraph multi-src API.
  // Example API shape shown; if your API differs, adapt accordingly.
  EdgeId CON = adj.addEdge(std::initializer_list<NodeId>{A, B}, C, /*op tag*/ 42);

  ConstGraph<int,int> g{adj};

  GraphPattern<int,int> pat;
  auto e = pat.matchEdge();
  e->matchRank(2); // concat takes 2 inputs

  // Ordered source constraints (src slot 0 then 1)
  auto S0 = e->matchSrc(0); S0->matchValue([](int v){ return v == 1; });
  auto S1 = e->matchSrc(1); S1->matchValue([](int v){ return v == 3; });

  auto D  = e->matchDst();  D->matchValue([](int v){ return v == 2; });

  std::vector<ConstGraphMatch<int,int>> ms;
  for (const auto& m : match_all(pat, g)) ms.push_back(m);

  ASSERT_EQ(ms.size(), 1u);
  const auto& m = ms[0];
  EXPECT_EQ(m[e],  CON);
  EXPECT_EQ(m[S0], A);
  EXPECT_EQ(m[S1], B);
  EXPECT_EQ(m[D],  C);
}


TEST(algorithm_match_all, edge_root_concat_wrong_order_no_match) {
  AdjGraph<int,int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(3);
  NodeId C = adj.addNode(2);
  (void)adj.addEdge(std::initializer_list<NodeId>{A, B}, C, 42); // same edge

  ConstGraph<int,int> g{adj};

  GraphPattern<int,int> pat;
  auto e = pat.matchEdge();
  e->matchRank(2);
  auto S0 = e->matchSrc(0); S0->matchValue([](int v){ return v == 3; }); // swapped
  auto S1 = e->matchSrc(1); S1->matchValue([](int v){ return v == 1; });
  auto D  = e->matchDst();  D->matchValue([](int v){ return v == 2; });

  size_t count = 0;
  for (const auto& m : match_all(pat, g)) { (void)m; ++count; }
  EXPECT_EQ(count, 0u); // source order is significant
}

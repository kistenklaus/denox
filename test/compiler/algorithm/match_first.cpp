#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "algorithm/pattern_matching/match.hpp"
#include "memory/hypergraph/AdjGraph.hpp"
#include "memory/hypergraph/ConstGraph.hpp"
#include <gtest/gtest.h>

using namespace denox::memory;
using namespace denox::algorithm;

TEST(algorithm_match_first, singular_node_match) {

  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_A = pattern.matchNode();
  m_A->matchValue([](const int &v) { return v == 1; });

  auto m = match_first(pattern, graph);
  ASSERT_TRUE(m.has_value());
  EXPECT_EQ(A, (*m)[m_A]);
}

TEST(algorithm_match_first, singular_neg_node_match) {

  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_A = pattern.matchNode();
  m_A->matchValue([](const int &v) { return v == 0; });

  auto m = match_first(pattern, graph);
  ASSERT_FALSE(m.has_value());
}

TEST(algorithm_match_first, singular_edge_match) {

  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  auto AB = adj.addEdge(A, B, 42);

  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_AB = pattern.matchEdge();
  m_AB->matchValue([](const int &e) { return e == 42; });

  auto m = match_first(pattern, graph);
  ASSERT_TRUE(m.has_value());
  EXPECT_EQ(AB, (*m)[m_AB]);
}

TEST(algorithm_match_first, singular_neg_edge_match) {

  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  auto AB = adj.addEdge(A, B, 42);

  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_AB = pattern.matchEdge();
  m_AB->matchValue([](const int &e) { return e == 0; });

  auto m = match_first(pattern, graph);
  ASSERT_FALSE(m.has_value());
}

TEST(algorithm_match_first, singular_indeg_match) {

  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  adj.addEdge(A, B, 42);
  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_B = pattern.matchNode();
  m_B->matchInDeg(1);

  auto m = match_first(pattern, graph);
  ASSERT_TRUE(m.has_value());
  EXPECT_EQ(B, (*m)[m_B]);
}

TEST(algorithm_match_first, singular_neg_indeg_match) {

  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  adj.addEdge(A, B, 42);
  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_B = pattern.matchNode();
  m_B->matchInDeg(2);

  auto m = match_first(pattern, graph);
  ASSERT_FALSE(m.has_value());
}

TEST(algorithm_match_first, singular_outdeg_match) {

  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  adj.addEdge(A, B, 42);
  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_A = pattern.matchNode();
  m_A->matchOutDeg(1);

  auto m = match_first(pattern, graph);
  ASSERT_TRUE(m.has_value());
  EXPECT_EQ(A, (*m)[m_A]);
}

TEST(algorithm_match_first, singular_neg_outdeg_match) {

  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  adj.addEdge(A, B, 42);
  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_A = pattern.matchNode();
  m_A->matchOutDeg(2);

  auto m = match_first(pattern, graph);
  ASSERT_FALSE(m.has_value());
}

TEST(algorithm_match_first, singular_rank_match_1) {
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  NodeId C = adj.addNode(3);
  auto AB = adj.addEdge(A, B, 42);
  auto ABC = adj.addEdge(A, B, C, 42);

  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_AB = pattern.matchEdge();
  m_AB->matchRank(1);

  auto m = match_first(pattern, graph);
  ASSERT_TRUE(m.has_value());
  EXPECT_EQ(AB, (*m)[m_AB]);
}

TEST(algorithm_match_first, singular_rank_match_2) {

  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  NodeId C = adj.addNode(3);
  auto AB = adj.addEdge(A, B, 42);
  auto ABC = adj.addEdge(A, B, C, 42);

  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_ABC = pattern.matchEdge();
  m_ABC->matchRank(2);

  auto m = match_first(pattern, graph);
  ASSERT_TRUE(m.has_value());
  EXPECT_EQ(ABC, (*m)[m_ABC]);
}

TEST(algorithm_match_first, singular_neg_rank_match) {

  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  NodeId C = adj.addNode(3);
  auto AB = adj.addEdge(A, B, 42);

  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_ABC = pattern.matchEdge();
  m_ABC->matchRank(2);

  auto m = match_first(pattern, graph);
  ASSERT_FALSE(m.has_value());
}

TEST(algorithm_match_first, match_node_with_single_outgoing_edge) {
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  auto AB = adj.addEdge(A, B, 12);
  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_A = pattern.matchNode();
  auto m_AB = m_A->matchOutgoing();

  m_A->matchOutDeg(1);
  m_AB->matchRank(1);
  m_AB->matchValue([](const int &v) { return v == 12; });

  auto m = match_first(pattern, graph);
  ASSERT_TRUE(m.has_value());
}

TEST(algorithm_match_first, match_edge_with_dst) {
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  auto AB = adj.addEdge(A, B, 12);
  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_AB = pattern.matchEdge();
  auto m_B = m_AB->matchDst();

  m_AB->matchRank(1);
  m_B->matchInDeg(1);

  auto m = match_first(pattern, graph);
  ASSERT_TRUE(m.has_value());
}

TEST(algorithm_match_first, match_linear_path) {
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  NodeId C = adj.addNode(3);
  auto AB = adj.addEdge(A, B, 42);
  auto BC = adj.addEdge(B, C, 42);
  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_A = pattern.matchNode();
  auto m_AB = m_A->matchOutgoing();
  auto m_B = m_AB->matchDst();
  auto m_BC = m_B->matchOutgoing();
  auto m_C = m_BC->matchDst();

  m_AB->matchValue([](const int &e) { return e == 42; });
  m_AB->matchRank(1);
  m_BC->matchValue([](const int &e) { return e == 42; });
  m_BC->matchRank(1);
  m_B->matchOutDeg(1);

  auto m = match_first(pattern, graph);
  ASSERT_TRUE(m.has_value());
  EXPECT_EQ(A, (*m)[m_A]);
  EXPECT_EQ(B, (*m)[m_B]);
  EXPECT_EQ(C, (*m)[m_C]);

  EXPECT_EQ(AB, (*m)[m_AB]);
  EXPECT_EQ(BC, (*m)[m_BC]);
}

TEST(algorithm_match_first, dont_match_linear_path) {
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  NodeId C = adj.addNode(3);
  NodeId D = adj.addNode(4);
  auto AB = adj.addEdge(A, B, 42);
  auto BC = adj.addEdge(B, C, 42);
  adj.addEdge(B, D, 42);
  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_A = pattern.matchNode();
  auto m_AB = m_A->matchOutgoing();
  auto m_B = m_AB->matchDst();
  auto m_BC = m_B->matchOutgoing();
  auto m_C = m_BC->matchDst();

  m_AB->matchValue([](const int &e) { return e == 42; });
  m_AB->matchRank(1);
  m_BC->matchValue([](const int &e) { return e == 42; });
  m_BC->matchRank(1);
  m_B->matchOutDeg(1);

  auto m = match_first(pattern, graph);
  ASSERT_FALSE(m.has_value());
}

TEST(algorithm_match_first, match_hyperedge_chain) {
  AdjGraph<int, int> adj;
  NodeId A = adj.addNode(1);
  NodeId B = adj.addNode(2);
  NodeId C = adj.addNode(3);
  NodeId D = adj.addNode(4);
  auto AB_C = adj.addEdge(A, B, C, 123);
  auto CD = adj.addEdge(C, D, 34);
  ConstGraph<int, int> graph{adj};

  GraphPattern<int, int> pattern;
  auto m_AB_C = pattern.matchEdge();
  auto m_C = m_AB_C->matchDst();
  auto m_CD = m_C->matchOutgoing();
  auto m_D = m_CD->matchDst();

  m_AB_C->matchRank(2);
  m_C->matchOutDeg(1);
  m_CD->matchRank(1);

  auto m = match_first(pattern, graph);
  ASSERT_TRUE(m.has_value());

  EXPECT_EQ(AB_C, (*m)[m_AB_C]);
  EXPECT_EQ(C, (*m)[m_C]);
  EXPECT_EQ(CD, (*m)[m_CD]);
  EXPECT_EQ(D, (*m)[m_D]);
}


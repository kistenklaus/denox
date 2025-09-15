#include "memory/hypergraph/LinkedGraph.hpp"
#include <gtest/gtest.h>
#include <vector>

#include "compiler/memory/TestType.hpp"
#include "compiler/memory/ProfileAllocator.hpp" // adjust include path if different
#include <algorithm>

#include <array>
#include <vector>

using Graph = denox::memory::LinkedGraph<int, int>;

TEST(memory_linked_graph, CreateNodes_NoEdges) {
  Graph g;
  auto A = g.createNode(1);
  auto B = g.createNode(2);

  // Fresh nodes have no edges.
  EXPECT_EQ(A->incoming().size(), 0u);
  EXPECT_EQ(A->outgoing().size(), 0u);
  EXPECT_EQ(B->incoming().size(), 0u);
  EXPECT_EQ(B->outgoing().size(), 0u);
}

TEST(memory_linked_graph, InsertViaIncoming_SingleEdge) {
  Graph g;
  auto A = g.createNode(10);
  auto B = g.createNode(20);

  // Add A -> B with payload 7 through B's incoming list.
  auto inB = B->incoming();
  auto it = inB.insert(A, 7);

  EXPECT_EQ(B->incoming().size(), 1u);
  EXPECT_EQ(A->outgoing().size(), 1u);

  // Inspect the edge and its endpoints/payload.
  auto first = B->incoming().begin();
  ASSERT_NE(first, B->incoming().end());
  EXPECT_EQ(first->value(), 7);
  // dst() on a const edge gives const Node&, on non-const gives NodeHandle.
  // Here EdgeIt yields a non-const Edge&, so use value() only and check srcs().
  {
    // Check the single source is exactly A.
    auto s = first->srcs().begin();
    ASSERT_NE(s, first->srcs().end());
    EXPECT_EQ(&(*s), &(*A)); // same Node address
    ++s;
    EXPECT_EQ(s, first->srcs().end());
  }
}

TEST(memory_linked_graph, InsertViaOutgoing_SingleEdge) {
  Graph g;
  auto A = g.createNode(10);
  auto B = g.createNode(20);

  // Add A -> B with payload 42 through A's outgoing list.
  auto outA = A->outgoing();
  auto it = outA.insert(B, 42);

  EXPECT_EQ(A->outgoing().size(), 1u);
  EXPECT_EQ(B->incoming().size(), 1u);

  auto e = A->outgoing().begin();
  ASSERT_NE(e, A->outgoing().end());
  EXPECT_EQ(e->value(), 42);

  // Verify dst of the edge is B (address compare).
  // (Use the const overload: grab const ref to force it.)
  const auto &edge_ref = *e;
  const auto &dst_node = edge_ref.dst();
  EXPECT_EQ(&dst_node, &(*B));
}

TEST(memory_linked_graph, Hyperedge_TwoSources_OneDst) {
  Graph g;
  auto A = g.createNode(1);
  auto X = g.createNode(2);
  auto B = g.createNode(3);

  // Insert (A, X) -> B with payload 99 via B's incoming()
  auto inB = B->incoming();
  inB.insert(*A, *X, 99); // uses the 2-src overload

  EXPECT_EQ(B->incoming().size(), 1u);
  EXPECT_EQ(A->outgoing().size(), 1u);
  EXPECT_EQ(X->outgoing().size(), 1u);

  auto e = B->incoming().begin();
  ASSERT_NE(e, B->incoming().end());
  EXPECT_EQ(e->value(), 99);

  // There should be exactly two sources: A and X (order not enforced here).
  const auto s0 = e->srcs().begin();
  ASSERT_NE(s0, e->srcs().end());
  const auto s1 = std::next(s0);
  ASSERT_NE(s1, e->srcs().end());
  const auto s2 = std::next(s1);
  EXPECT_EQ(s2, e->srcs().end());

  // Check the set {&A,&X} matches the srcs we saw.
  std::array<const Graph::Node *, 2> expect{&(*A), &(*X)};
  std::array<const Graph::Node *, 2> got{&(*s0), &(*s1)};
  // Order might differ; check membership.
  auto contains = [&](const Graph::Node *p) {
    return p == expect[0] || p == expect[1];
  };
  EXPECT_TRUE(contains(got[0]));
  EXPECT_TRUE(contains(got[1]));
  EXPECT_NE(got[0], got[1]);
}

TEST(memory_linked_graph, IncomingErase_NoCascade) {
  Graph g;
  auto A = g.createNode(11);
  auto B = g.createNode(22);

  // A -> B
  B->incoming().insert(A, 5);
  ASSERT_EQ(B->incoming().size(), 1u);
  ASSERT_EQ(A->outgoing().size(), 1u);

  // Erase via incoming() (guaranteed not to cascade thanks to external on B).
  auto inB = B->incoming();
  auto it = inB.erase(inB.begin());
  EXPECT_EQ(it, inB.end());

  EXPECT_EQ(B->incoming().size(), 0u);
  EXPECT_EQ(A->outgoing().size(), 0u);
}

TEST(memory_linked_graph, OutgoingErase_CascadeThroughMiddle) {
  Graph g;
  auto A = g.createNode(1);
  auto B = g.createNode(2);
  auto C = g.createNode(3);

  // Build chain A -> B -> C
  A->outgoing().insert(B, 10);
  B->outgoing().insert(C, 20);

  ASSERT_EQ(A->outgoing().size(), 1u);
  ASSERT_EQ(B->incoming().size(), 1u);
  ASSERT_EQ(B->outgoing().size(), 1u);
  ASSERT_EQ(C->incoming().size(), 1u);

  // Hold an external handle to C so that deleting B doesn't delete C.
  B.release(); // <- drop external_count of B.

  // Now erase A->B via A's outgoing(). That should drop B's live_parent to 0,
  // cascade-delete B, and as part of that, remove B->C from C's incoming().
  auto outA = A->outgoing();
  auto next = outA.erase(outA.begin());
  EXPECT_EQ(next, outA.end());

  EXPECT_EQ(A->outgoing().size(), 0u);
  // C should have lost the edge from B->C during the cascade.
  EXPECT_EQ(C->incoming().size(), 0u);
}

TEST(memory_linked_graph, OutgoingErase_SingletonRingEnds) {
  Graph g;
  auto A = g.createNode(1);
  auto B = g.createNode(2);

  A->outgoing().insert(B, 123);
  ASSERT_EQ(A->outgoing().size(), 1u);

  auto outA = A->outgoing();
  auto it = outA.begin();
  ASSERT_NE(it, outA.end());
  it = outA.erase(it);
  EXPECT_EQ(it, outA.end());
  EXPECT_EQ(A->outgoing().size(), 0u);
}

using Graph = denox::memory::LinkedGraph<int,int>;

static std::vector<int> collect_payloads(Graph::NodeHandle& nh, bool incoming) {
  std::vector<int> v;
  if (incoming) {
    for (auto it = nh->incoming().begin(), e = nh->incoming().end(); it != e; ++it)
      v.push_back(it->value());
  } else {
    for (auto it = nh->outgoing().begin(), e = nh->outgoing().end(); it != e; ++it)
      v.push_back(it->value());
  }
  std::sort(v.begin(), v.end());
  return v;
}

TEST(memory_linked_graph, ParallelEdges_SameSrcDst) {
  Graph g;
  auto A = g.createNode(1);
  auto B = g.createNode(2);

  // Two parallel edges A->B with different payloads.
  A->outgoing().insert(B, 10);
  A->outgoing().insert(B, 20);

  EXPECT_EQ(A->outgoing().size(), 2u);
  EXPECT_EQ(B->incoming().size(), 2u);

  auto outA = collect_payloads(A, /*incoming=*/false);
  auto inB  = collect_payloads(B, /*incoming=*/true);
  EXPECT_EQ(outA, (std::vector<int>{10,20}));
  EXPECT_EQ(inB,  (std::vector<int>{10,20}));

  // Erase exactly one of them via B's incoming ring.
  // Find payload 10 and erase.
  auto inList = B->incoming();
  for (auto it = inList.begin(); it != inList.end(); ) {
    if (it->value() == 10) {
      it = inList.erase(it);
      break;
    } else {
      ++it;
    }
  }

  EXPECT_EQ(A->outgoing().size(), 1u);
  EXPECT_EQ(B->incoming().size(), 1u);

  outA = collect_payloads(A, /*incoming=*/false);
  inB  = collect_payloads(B, /*incoming=*/true);
  EXPECT_EQ(outA, (std::vector<int>{20}));
  EXPECT_EQ(inB,  (std::vector<int>{20}));
}

TEST(memory_linked_graph, ParallelHyperedges_TwoSources_DuplicateEdges) {
  Graph g;
  auto A = g.createNode(1);
  auto X = g.createNode(2);
  auto B = g.createNode(3);

  // Two parallel hyper-edges (A,X)->B with different payloads.
  B->incoming().insert(*A, *X, 7);
  B->incoming().insert(*A, *X, 9);

  EXPECT_EQ(B->incoming().size(), 2u);
  EXPECT_EQ(A->outgoing().size(), 2u);
  EXPECT_EQ(X->outgoing().size(), 2u);

  auto inB = collect_payloads(B, /*incoming=*/true);
  EXPECT_EQ(inB, (std::vector<int>{7,9}));

  // Remove one via X's outgoing (this triggers the “sentinel” path).
  // Find payload 7 on X->outgoing and erase.
  auto outX = X->outgoing();
  for (auto it = outX.begin(); it != outX.end(); ) {
    if (it->value() == 7) {
      it = outX.erase(it);
      break;
    } else {
      ++it;
    }
  }

  // One hyper-edge should remain across all rings.
  EXPECT_EQ(B->incoming().size(), 1u);
  EXPECT_EQ(A->outgoing().size(), 1u);
  EXPECT_EQ(X->outgoing().size(), 1u);
  inB = collect_payloads(B, /*incoming=*/true);
  EXPECT_EQ(inB, (std::vector<int>{9}));
}

TEST(memory_linked_graph, OutgoingIter_EraseAndInsertDuringTraversal) {
  Graph g;
  auto A = g.createNode(0);
  auto B = g.createNode(1);
  auto C = g.createNode(2);
  auto D = g.createNode(3);

  // Start with A->{B:1, C:2}
  A->outgoing().insert(B, 1);
  A->outgoing().insert(C, 2);

  auto out = A->outgoing();
  for (auto it = out.begin(); it != out.end(); ) {
    if (it->value() == 1) {
      // erase A->B, then insert A->D:5 after current position
      it = out.erase(it);
      it = out.insert_after(it, D, 5); // returns ++pos; newly inserted sits behind original
    } else {
      ++it;
    }
  }

  // Final outgoing: {C:2, D:5} in some cyclic order.
  EXPECT_EQ(A->outgoing().size(), 2u);
  auto outA = collect_payloads(A, /*incoming=*/false);
  EXPECT_EQ(outA, (std::vector<int>{2,5}));

  // B lost its incoming; C unchanged; D gained incoming.
  EXPECT_EQ(B->incoming().size(), 0u);
  EXPECT_EQ(C->incoming().size(), 1u);
  EXPECT_EQ(D->incoming().size(), 1u);
}

TEST(memory_linked_graph, PayloadMutationThroughIterator) {
  Graph g;
  auto A = g.createNode(10);
  auto B = g.createNode(20);

  A->outgoing().insert(B, 111);
  ASSERT_EQ(A->outgoing().size(), 1u);
  ASSERT_EQ(B->incoming().size(), 1u);

  // Mutate payload via outgoing iterator.
  auto it = A->outgoing().begin();
  ASSERT_NE(it, A->outgoing().end());
  it->value() = 222;

  // Visible from both sides.
  EXPECT_EQ(A->outgoing().begin()->value(), 222);
  EXPECT_EQ(B->incoming().begin()->value(), 222);
}

TEST(memory_linked_graph, LargeLinearChain_InsertAndSelectiveErase) {
  Graph g;

  constexpr int N = 200; // 200 nodes, 199 edges in a line
  std::vector<Graph::NodeHandle> nodes;
  nodes.reserve(N);
  for (int i = 0; i < N; ++i) nodes.push_back(g.createNode(i));

  // Build chain i -> i+1 with payload i
  for (int i = 0; i < N - 1; ++i) {
    nodes[i]->outgoing().insert(nodes[i+1], i);
  }

  // Basic counts
  EXPECT_EQ(nodes.front()->outgoing().size(), 1u);
  EXPECT_EQ(nodes.back()->incoming().size(), 1u);
  for (int i = 1; i < N - 1; ++i) {
    EXPECT_EQ(nodes[i]->incoming().size(), 1u);
    EXPECT_EQ(nodes[i]->outgoing().size(), 1u);
  }

  // Erase every edge with even payload from its source’s outgoing list.
  for (int i = 0; i < N - 1; ++i) {
    if ((i % 2) == 0) {
      auto out = nodes[i]->outgoing();
      for (auto it = out.begin(); it != out.end(); ) {
        if (it->value() == i) {
          it = out.erase(it);
          break;
        } else {
          ++it;
        }
      }
    }
  }

  // Check remaining outgoing payloads from all nodes are odd i.
  std::vector<int> remaining;
  for (int i = 0; i < N - 1; ++i) {
    for (auto it = nodes[i]->outgoing().begin(); it != nodes[i]->outgoing().end(); ++it)
      remaining.push_back(it->value());
  }
  std::sort(remaining.begin(), remaining.end());

  std::vector<int> expected;
  for (int i = 1; i < N - 1; i += 2) expected.push_back(i);
  EXPECT_EQ(remaining, expected);
}

TEST(memory_linked_graph, CascadeLongChain_MiddleRemoval) {
  Graph g;
  auto A = g.createNode(1);
  auto B = g.createNode(2);
  auto C = g.createNode(3);
  auto D = g.createNode(4);

  // A->B->C->D
  A->outgoing().insert(B, 10);
  B->outgoing().insert(C, 11);
  C->outgoing().insert(D, 12);

  // Keep D alive so the tail isn't destroyed, but middle should collapse.
  auto keepD = D;

  // Drop external for B and C so they can be collected by cascade.
  B.release();
  C.release();

  // Remove A->B; expect B and C to cascade away, removing B->C and C->D.
  auto outA = A->outgoing();
  outA.erase(outA.begin());

  EXPECT_EQ(A->outgoing().size(), 0u);
  EXPECT_EQ(D->incoming().size(), 0u);  // C->D removed by cascade
}

TEST(memory_linked_graph, InsertAfterEnd_EmptyRings) {
  Graph g;
  auto A = g.createNode(1);
  auto B = g.createNode(2);

  // Outgoing empty: insert_after(end()) must work.
  {
    auto outA = A->outgoing();
    auto it = outA.end();
    it = outA.insert_after(it, B, 77);
    EXPECT_EQ(A->outgoing().size(), 1u);
    EXPECT_EQ(B->incoming().size(), 1u);
    EXPECT_EQ(A->outgoing().begin()->value(), 77);
  }

  // Incoming empty on a fresh node C: insert_after(end()) must work.
  auto C = g.createNode(3);
  {
    auto inC = C->incoming();
    auto it = inC.end();
    it = inC.insert_after(it, *A, 88);
    EXPECT_EQ(C->incoming().size(), 1u);
    EXPECT_EQ(A->outgoing().size(), 2u);
    EXPECT_EQ(C->incoming().begin()->value(), 88);
  }
}

TEST(memory_linked_graph, ProbeEdge_RValueInsertErase_NoCopies) {
  denox::memory::LinkedGraph<int, denox::testing::Probe<int>> g;
  auto A = g.createNode(1);
  auto B = g.createNode(2);

  denox::testing::OperationStats es;
  denox::testing::LeakSentinel guard{es};

  // Insert A->B with a temporary Probe payload
  {
    auto incoming = B->incoming();
    incoming.insert(
        A,
        denox::testing::Probe(es, std::in_place, 7)
    );
  }

  // One move-construct into the edge, no copies/assigns yet, not destroyed yet
  EXPECT_EQ(es.move_ctor, 1u);
  EXPECT_EQ(es.copy_ctor, 0u);
  EXPECT_EQ(es.move_assign, 0u);
  EXPECT_EQ(es.copy_assign, 0u);
  EXPECT_EQ(es.dtor, 1u);
  EXPECT_EQ(es.alive, 1u);

  // Validate payload
  auto it = B->incoming().begin();
  ASSERT_NE(it, B->incoming().end());
  EXPECT_EQ(*it->value(), 7);

  // Erase; payload must be destroyed exactly once
  auto inB = B->incoming();
  inB.erase(inB.begin());
  EXPECT_EQ(es.dtor, 2u);
  EXPECT_EQ(es.alive, 0u);
}

TEST(memory_linked_graph, ProbeEdge_LValueInsert_CopiesOnce) {
  denox::memory::LinkedGraph<int, denox::testing::Probe<int>> g;
  auto A = g.createNode(10);
  auto B = g.createNode(20);

  denox::testing::OperationStats es;
  denox::testing::LeakSentinel guard{es};

  denox::testing::Probe<int> payload_lvalue(es, std::in_place, 42);
  // Insert using lvalue -> copy-construct into edge
  B->incoming().insert(A, payload_lvalue);

  EXPECT_EQ(es.copy_ctor, 1u);
  EXPECT_EQ(es.move_ctor, 0u);
  EXPECT_EQ(es.copy_assign, 0u);
  EXPECT_EQ(es.move_assign, 0u);
  // two live: the lvalue + the copy inside the edge
  EXPECT_EQ(es.alive, 2u);

  // Clean up: erase edge first
  auto inB = B->incoming();
  inB.erase(inB.begin());
  // Then lvalue goes out of scope; LeakSentinel verifies parity
}

TEST(memory_linked_graph, ProbeEdge_ParallelEdges_NoExtraOps) {
  denox::memory::LinkedGraph<int, denox::testing::Probe<int>> g;
  auto A = g.createNode(1);
  auto B = g.createNode(2);

  constexpr int K = 6;
  std::array<denox::testing::OperationStats, K> stats{};
  std::array<denox::testing::LeakSentinel, K> sentinels = {
      denox::testing::LeakSentinel{stats[0]},
      denox::testing::LeakSentinel{stats[1]},
      denox::testing::LeakSentinel{stats[2]},
      denox::testing::LeakSentinel{stats[3]},
      denox::testing::LeakSentinel{stats[4]},
      denox::testing::LeakSentinel{stats[5]},
  };

  // Insert K edges A->B, each with its own Probe payload
  for (int i = 0; i < K; ++i) {
    B->incoming().insert(
        A,
        denox::testing::Probe<int>(stats[static_cast<size_t>(i)], std::in_place, 100 + i)
    );
  }

  // During lifetime: each edge payload was move-constructed once, no extra ops
  for (int i = 0; i < K; ++i) {
    const auto& s = stats[static_cast<size_t>(i)];
    EXPECT_EQ(s.move_ctor, 1u) << i;
    EXPECT_EQ(s.copy_ctor, 0u) << i;
    EXPECT_EQ(s.move_assign, 0u) << i;
    EXPECT_EQ(s.copy_assign, 0u) << i;
    EXPECT_EQ(s.dtor, 1u) << i;
    EXPECT_EQ(s.alive, 1u) << i;
  }

  // Erase them all via incoming
  auto inB = B->incoming();
  for (auto it = inB.begin(); it != inB.end(); ) {
    it = inB.erase(it);
  }

  for (int i = 0; i < K; ++i) {
    const auto& s = stats[static_cast<size_t>(i)];
    EXPECT_EQ(s.dtor, 2u) << i;
    EXPECT_EQ(s.alive, 0u) << i;
  }
}

TEST(memory_linked_graph, ProbeEdge_Hyperedge_TwoSources) {
  denox::memory::LinkedGraph<int, denox::testing::Probe<int>> g;
  auto A = g.createNode(1);
  auto X = g.createNode(2);
  auto B = g.createNode(3);

  denox::testing::OperationStats es;
  denox::testing::LeakSentinel guard{es};

  // (A, X) -> B
  auto inB = B->incoming();
  inB.insert_after(inB.begin(), *A, *X, denox::testing::Probe<int>(es, std::in_place, 99));

  // Exactly one payload alive (the edge), created by a single move-ctor
  EXPECT_EQ(es.move_ctor, 1u);
  EXPECT_EQ(es.copy_ctor, 0u);
  EXPECT_EQ(es.alive, 1u);
  EXPECT_EQ(es.dtor, 1u);

  // Check it’s a hyperedge with two sources
  auto e = B->incoming().begin();
  ASSERT_NE(e, B->incoming().end());
  EXPECT_EQ(*e->value(), 99);

  auto s0 = e->srcs().begin();
  ASSERT_NE(s0, e->srcs().end());
  auto s1 = std::next(s0);
  ASSERT_NE(s1, e->srcs().end());
  auto s2 = std::next(s1);
  EXPECT_EQ(s2, e->srcs().end());

  // Remove and verify destruction
  inB.erase(inB.begin());
  EXPECT_EQ(es.dtor, 2u);
  EXPECT_EQ(es.alive, 0u);
}

TEST(memory_linked_graph, ProbeNode_RValueConstruct_CascadeParities) {
  // Node payload is Probe<int>; edge payload is plain int
  denox::memory::LinkedGraph<denox::testing::Probe<int>, int> g;

  denox::testing::OperationStats sA, sB, sC;
  denox::testing::LeakSentinel gA{sA}, gB{sB}, gC{sC};

  auto A = g.createNode(denox::testing::Probe<int>(sA, std::in_place, 10));
  auto B = g.createNode(denox::testing::Probe<int>(sB, std::in_place, 20));
  auto C = g.createNode(denox::testing::Probe<int>(sC, std::in_place, 30));

  // Two edges: A->B, B->C
  A->outgoing().insert(B, 1);
  B->outgoing().insert(C, 2);

  // Drop A and B and remove A->B so cascade can delete B (and B->C)
  auto keepC = C;  // keep C alive to prevent cascading into C
  B.release();     // allow cascade to remove B when it loses last parent
  {
    auto outA = A->outgoing();
    outA.erase(outA.begin()); // removes A->B; B can cascade
  }
  A.release(); // now A can also go

  // At test end, LeakSentinel checks ctor/dtor parity for node payloads.
  // We can also assert “eventually destroyed” now (counts can already be 1).
  EXPECT_EQ(sB.alive, 0u);
  EXPECT_EQ(sB.dtor, sB.total_constructs());
}

TEST(memory_linked_graph, ProbeEdge_DstConstVsHandle_NoCopies) {
  denox::memory::LinkedGraph<int, denox::testing::Probe<int>> g;
  auto A = g.createNode(1);
  auto B = g.createNode(2);

  denox::testing::OperationStats es;
  denox::testing::LeakSentinel guard{es};

  A->outgoing().insert(B, denox::testing::Probe<int>(es, std::in_place, 5));

  auto it = A->outgoing().begin();
  ASSERT_NE(it, A->outgoing().end());

  // Force const edge ref to hit const dst() overload; should not copy/move payload
  const auto& cref = *it;
  const auto& dst_node = cref.dst();
  (void)dst_node;

  EXPECT_EQ(es.copy_ctor + es.move_ctor, 1u); // just the construct into the edge
  EXPECT_EQ(es.copy_assign + es.move_assign, 0u);

  // Clean up to satisfy LeakSentinel
  auto outA = A->outgoing();
  outA.erase(outA.begin());
}


TEST(memory_linked_graph, DstHandlePreventsCascadeThenAllowsAfterRelease) {
  denox::memory::LinkedGraph<int,int> g;
  auto A = g.createNode(1);
  auto B = g.createNode(2);
  auto C = g.createNode(3);

  // A->B->C
  A->outgoing().insert(B, 10);
  B->outgoing().insert(C, 20);

  // Grab a dst handle from A->B (bumps B.external_count)
  auto e = A->outgoing().begin();
  ASSERT_NE(e, A->outgoing().end());
  auto keepB = e->dst();

  // Drop the original B handle so only keepB keeps B alive.
  B.release();

  // Erase A->B; without keepB this would cascade B (and remove B->C).
  A->outgoing().erase(A->outgoing().begin());

  // B is still alive due to keepB; C still sees B->C.
  EXPECT_EQ(keepB->incoming().size(), 0u);
  EXPECT_EQ(keepB->outgoing().size(), 1u);
  EXPECT_EQ(C->incoming().size(), 1u);

  // Now release the held handle; B should cascade and remove B->C.
  keepB.release();
  EXPECT_EQ(C->incoming().size(), 0u);
}

TEST(memory_linked_graph, IteratorDecrement_EmptyAndSingleton) {
  denox::memory::LinkedGraph<int,int> g;

  // Empty incoming ring: --end() stays end()
  auto A = g.createNode(1);
  auto inA = A->incoming();
  auto it = inA.end();
  auto before = it;
  --it;
  // stays end because ring is empty
  EXPECT_EQ(it, before);

  // Singleton outgoing ring: --end() lands on the only element; ++ gets back to end()
  auto B = g.createNode(2);
  A->outgoing().insert(B, 42);

  auto outA = A->outgoing();
  auto eit = outA.end();
  --eit; // should point to the only element
  ASSERT_NE(eit, outA.end());
  EXPECT_EQ(eit->value(), 42);
  ++eit;
  EXPECT_EQ(eit, outA.end());
}

TEST(memory_linked_graph, SizeCache_Incoming_UpdatesIncrementally) {
  denox::memory::LinkedGraph<int,int> g;
  auto A = g.createNode(1);
  auto B = g.createNode(2);
  auto C = g.createNode(3);

  // First insert; size() computes via traversal and caches
  B->incoming().insert(A, 10);
  auto inB = B->incoming();
  EXPECT_EQ(inB.size(), 1u);

  // Insert via incoming(); list tracks size incrementally
  inB.insert_after(inB.begin(), *C, 11);
  EXPECT_EQ(inB.size(), 2u);

  // Erase one; incoming() decrements cached size
  inB.erase(inB.begin());
  EXPECT_EQ(inB.size(), 1u);
}

TEST(memory_linked_graph, SizeCache_Outgoing_InvalidatesOnEraseAndRecounts) {
  denox::memory::LinkedGraph<int,int> g;
  auto A = g.createNode(1);
  auto B = g.createNode(2);
  auto C = g.createNode(3);
  auto D = g.createNode(4);

  A->outgoing().insert(B, 1);
  A->outgoing().insert(C, 2);
  A->outgoing().insert(D, 3);

  auto outA = A->outgoing();
  EXPECT_EQ(outA.size(), 3u);      // computes & caches

  // Erase invalidates cache (internal) and the next size() must still be correct
  outA.erase(outA.begin());
  EXPECT_EQ(outA.size(), 2u);

  // Insert after known iterator bumps size by +1 without full recount
  auto it = outA.begin();
  it = outA.insert_after(it, B, 9);
  EXPECT_EQ(outA.size(), 3u);
}

TEST(memory_linked_graph, ParallelEdges_BulkEraseFromBothSides) {
  denox::memory::LinkedGraph<int,int> g;
  auto A = g.createNode(1);
  auto B = g.createNode(2);

  // Make 20 parallel edges A->B with payloads 0..19
  for (int i = 0; i < 20; ++i) A->outgoing().insert(B, i);
  EXPECT_EQ(A->outgoing().size(), 20u);
  EXPECT_EQ(B->incoming().size(), 20u);

  // Remove evens via outgoing(), odds via incoming()
  {
    auto outA = A->outgoing();
    for (auto it2 = outA.begin(); it2 != outA.end(); ) {
      if ((it2->value() % 2) == 0) {
        it2 = outA.erase(it2);
      } else {
        ++it2;
      }
    }
  }
  {
    auto inB = B->incoming();
    for (auto it2 = inB.begin(); it2 != inB.end(); ) {
      if ((it2->value() % 2) == 1) {
        it2 = inB.erase(it2);
      } else {
        ++it2;
      }
    }
  }

  EXPECT_EQ(A->outgoing().size(), 0u);
  EXPECT_EQ(B->incoming().size(), 0u);
}

#ifndef NDEBUG
// This relies on assertions being enabled.
TEST(memory_linked_graph, DeathTest_SelfEdgeNotAllowed) {
  denox::memory::LinkedGraph<int,int> g;
  auto A = g.createNode(1);
  // Self-edge should trip an internal assert
  EXPECT_DEATH( (void)A->outgoing().insert(A, 123), "" );
}
#endif

TEST(memory_linked_graph, SrcIteration_HyperedgeStableContent) {
  denox::memory::LinkedGraph<int,int> g;
  auto A = g.createNode(1);
  auto B = g.createNode(2);
  auto C = g.createNode(3);

  // (A,B)->C
  auto inC = C->incoming();
  inC.insert(*A, *B, 7);

  auto e = C->incoming().begin();
  ASSERT_NE(e, C->incoming().end());

  // Collect the src addresses
  std::vector<const denox::memory::LinkedGraph<int,int>::Node*> got;
  for (auto si = e->srcs().begin(); si != e->srcs().end(); ++si) {
    got.push_back(&(*si));
  }
  ASSERT_EQ(got.size(), 2u);
  EXPECT_TRUE((got[0] == &(*A) && got[1] == &(*B)) ||
              (got[0] == &(*B) && got[1] == &(*A)));
}


#include "memory/hypergraph/LinkedGraph.hpp"
#include <gtest/gtest.h>

using Alloc = denox::testing::ProfileAllocator;
template<class V=int, class E=int>
using G = denox::memory::LinkedGraph<V,E,denox::memory::NullWeight,Alloc>;

TEST(memory_linked_graph, ProfileAllocator_EmptyGraph_AllFreedOnDestruction) {
  Alloc pa;
  {
    G<> g(pa);
    // no nodes/edges
    // (We don't assert mid-scope counts; pools may lazily allocate.)
  }
  EXPECT_EQ(pa.allocationCount(), 0u);
  EXPECT_EQ(pa.allocatedBytes(), 0u);
}

TEST(memory_linked_graph, ProfileAllocator_NodesOnly_AllFreedOnDestruction) {
  Alloc pa;
  {
    G<> g(pa);
    auto A = g.createNode(1);
    auto B = g.createNode(2);
    auto C = g.createNode(3);
    // release handles so nodes can die before graph destruction (optional)
    A.release(); B.release(); C.release();
  }
  EXPECT_EQ(pa.allocationCount(), 0u);
  EXPECT_EQ(pa.allocatedBytes(), 0u);
  // There should have been some upstream allocations in practice:
  EXPECT_GT(pa.peakAllocationCount(), 0u);
  EXPECT_GT(pa.peakAllocatedBytes(), 0u);
}

TEST(memory_linked_graph, ProfileAllocator_EdgesAndCascade_AllFreedOnDestruction) {
  Alloc pa;
  {
    G<> g(pa);
    auto A = g.createNode(1);
    auto B = g.createNode(2);
    auto C = g.createNode(3);
    auto D = g.createNode(4);

    // Build a small DAG with multiple edges and a hyperedge
    A->outgoing().insert(B, 10);
    B->outgoing().insert(C, 20);
    A->outgoing().insert(C, 30);
    C->incoming().insert(*A, *B, 40); // hyperedge (A,B)->C

    // Force a cascade path at runtime (optional):
    // Drop external refs to B and C, then erase A->B to cascade B and clean up its edges.
    B.release();
    C.release();
    auto outA = A->outgoing();
    if (outA.begin() != outA.end())
      outA.erase(outA.begin());
    // keep A and D alive until scope end
  }
  EXPECT_EQ(pa.allocationCount(), 0u);
  EXPECT_EQ(pa.allocatedBytes(), 0u);
  EXPECT_GT(pa.peakAllocationCount(), 0u);
  EXPECT_GT(pa.peakAllocatedBytes(), 0u);
}

TEST(memory_linked_graph, ProfileAllocator_ParallelEdges_Stress_AllFreedOnDestruction) {
  Alloc pa;
  {
    G<> g(pa);
    constexpr int N = 200;
    std::vector<G<>::NodeHandle> nodes;
    nodes.reserve(N);
    for (int i = 0; i < N; ++i) nodes.push_back(g.createNode(i));

    // Create lots of edges (including parallel edges) to ensure pools grab slabs
    for (int i = 0; i+1 < N; ++i) {
      nodes[i]->outgoing().insert(nodes[i+1], i);
      nodes[i]->outgoing().insert(nodes[i+1], i + 1000); // parallel edge
      if (i+2 < N) {
        // sprinkle a few “skip” edges
        nodes[i]->outgoing().insert(nodes[i+2], i + 2000);
      }
    }

    // Drop many handles to allow partial cascades during scope (optional)
    for (int i = 1; i+1 < N; ++i) nodes[i].release();
  }
  EXPECT_EQ(pa.allocationCount(), 0u);
  EXPECT_EQ(pa.allocatedBytes(), 0u);
  EXPECT_GT(pa.peakAllocationCount(), 0u);
  EXPECT_GT(pa.peakAllocatedBytes(), 0u);
}

TEST(memory_linked_graph, ProfileAllocator_HyperedgesOnly_AllFreedOnDestruction) {
  Alloc pa;
  {
    G<> g(pa);
    auto A = g.createNode(1);
    auto B = g.createNode(2);
    auto C = g.createNode(3);
    auto D = g.createNode(4);

    // A hyperedge fan-in to D
    auto inD = D->incoming();
    inD.insert(*A, *B, 7);
    inD.insert(*A, *C, 8);
    inD.insert(*B, *C, 9);

    // Erase some via outgoing to exercise sentinel path
    auto outB = B->outgoing();
    for (auto it = outB.begin(); it != outB.end(); ) {
      if (it->value() == 7 || it->value() == 9) it = outB.erase(it);
      else ++it;
    }
  }
  EXPECT_EQ(pa.allocationCount(), 0u);
  EXPECT_EQ(pa.allocatedBytes(), 0u);
  EXPECT_GT(pa.peakAllocationCount(), 0u);
  EXPECT_GT(pa.peakAllocatedBytes(), 0u);
}

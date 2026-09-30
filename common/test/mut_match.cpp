#include "denox/algorithm/pattern_matching/mut_match.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

namespace {
using Graph = denox::memory::LinkedGraph<int, int>;
using Pattern = denox::algorithm::GraphPattern<int, int>;
using EdgeControl =
    denox::algorithm::pattern_matching::details::EdgeMatchControl<int, int>;
using Direction = denox::algorithm::pattern_matching::details::MatchDirection;

TEST(MutMatch, IncomingControlPreservesBothContinuationsAndInvalidatesAliases) {
  Graph graph;
  auto a = graph.createNode(0);
  auto b = graph.createNode(1);
  auto c = graph.createNode(2);
  auto d = graph.createNode(3);
  auto edge = a->outgoing().insert(c, 10);
  auto outgoingNext = a->outgoing().insert_after(edge, d, 20);
  b->outgoing().insert(c, 30);
  auto incoming = c->incoming().begin();
  while (incoming->value() != 10)
    ++incoming;
  auto incomingNext = incoming;
  ++incomingNext;
  EdgeControl::Context context;
  EdgeControl out(a, edge, Direction::Outgoing, &context);
  EdgeControl in(c, incoming, Direction::Incoming, &context);
  denox::algorithm::EdgeMatch<int, int> match(&in);
  EXPECT_EQ(match.sourceNode(), a);
  EXPECT_EQ(match.outgoingIterator(), edge);
  EXPECT_EQ(match.value(), 10);
  match.erase();
  EXPECT_TRUE(in.dirty());
  EXPECT_TRUE(out.dirty());
  EXPECT_EQ(in.ptr(), nullptr);
  EXPECT_EQ(out.ptr(), nullptr);
  EXPECT_EQ(in.nextIterator(), incomingNext);
  EXPECT_EQ(out.nextIterator(), outgoingNext);
  EXPECT_EQ(match.nextOutgoingIterator(), outgoingNext);
  EXPECT_THROW(match.value(), std::logic_error);
  EXPECT_EQ(c->incoming().size(), 1u);
  EXPECT_EQ(a->outgoing().size(), 1u);
}

TEST(MutMatch, IncomingHyperedgeControlErasesEverySourceEntry) {
  Graph graph;
  auto a = graph.createNode(0);
  auto b = graph.createNode(1);
  auto c = graph.createNode(2);
  a->outgoing().insert(b, c, 10);
  EdgeControl control(c, c->incoming().begin(), Direction::Incoming);
  EXPECT_EQ(control.sourceNode(), a);
  control.erase();
  EXPECT_EQ(a->outgoing().size(), 0u);
  EXPECT_EQ(b->outgoing().size(), 0u);
  EXPECT_EQ(c->incoming().size(), 0u);
}

TEST(MutMatch, ErasingSavedSuccessorRepairsContinuation) {
  Graph graph;
  auto a = graph.createNode(0);
  auto b = graph.createNode(1);
  auto c = graph.createNode(2);
  auto first = a->outgoing().insert(b, 10);
  auto second = a->outgoing().insert_after(first, c, 20);
  EdgeControl::Context context;
  EdgeControl one(a, first, Direction::Outgoing, &context);
  EdgeControl two(a, second, Direction::Outgoing, &context);
  one.erase();
  two.erase();
  EXPECT_EQ(one.nextIterator(), a->outgoing().end());
  EXPECT_EQ(one.nextOutgoingIterator(), a->outgoing().end());
}

TEST(MutMatch, MatchesOutgoingChain) {
  Graph graph;
  auto root = graph.createNode(0);
  auto middle = graph.createNode(1);
  auto leaf = graph.createNode(2);
  root->outgoing().insert(middle, 10);
  middle->outgoing().insert(leaf, 20);

  Pattern pattern;
  auto rootPattern = pattern.matchNode();
  auto first = rootPattern->matchOutgoing();
  auto middlePattern = first->matchDst();
  auto second = middlePattern->matchOutgoing();
  auto leafPattern = second->matchDst();
  (void)leafPattern;

  unsigned matches = 0;
  for (const auto &match : denox::algorithm::match_all(pattern, root)) {
    ++matches;
    EXPECT_EQ(match[rootPattern]->id(), root->id());
    EXPECT_EQ(match[first].value(), 10);
    EXPECT_EQ(match[middlePattern]->id(), middle->id());
    EXPECT_EQ(match[second].value(), 20);
    EXPECT_EQ(match[leafPattern]->id(), leaf->id());
  }
  EXPECT_EQ(matches, 1u);
}

TEST(MutMatch, MatchesConvConvAddBackwardsWithOrderedSources) {
  Graph graph;
  auto a = graph.createNode(1), b = graph.createNode(2);
  auto ca = graph.createNode(3), cb = graph.createNode(4);
  auto output = graph.createNode(5);
  a->outgoing().insert(ca, 10);
  b->outgoing().insert(cb, 10);
  ca->outgoing().insert(cb, output, 20);
  Pattern pattern;
  auto x = pattern.matchNode();
  auto add = x->matchIncoming();
  add->matchRank(2);
  add->matchValue([](int v) { return v == 20; });
  auto left = add->matchSrc(0), right = add->matchSrc(1);
  left->matchInDeg(1);
  left->matchOutDeg(1);
  right->matchInDeg(1);
  right->matchOutDeg(1);
  auto convA = left->matchIncoming(), convB = right->matchIncoming();
  convA->matchValue([](int v) { return v == 10; });
  convB->matchValue([](int v) { return v == 10; });
  auto inputA = convA->matchSrc(0), inputB = convB->matchSrc(0);
  for (bool reject : {false, true}) {
    inputA->matchValue([reject](int v) { return v == (reject ? 2 : 1); });
    unsigned count = 0;
    for (const auto &match : denox::algorithm::match_all(pattern, a)) {
      ++count;
      EXPECT_EQ(match[x], output);
      EXPECT_EQ(match[left], ca);
      EXPECT_EQ(match[right], cb);
      EXPECT_EQ(match[inputA], a);
      EXPECT_EQ(match[inputB], b);
      EXPECT_EQ(match[convA].value(), 10);
      EXPECT_EQ(match[convB].value(), 10);
    }
    EXPECT_EQ(count, reject ? 0u : 1u);
  }
}

TEST(MutMatch, CombinesIncomingAndOutgoingProducerAlternatives) {
  Graph graph;
  auto a = graph.createNode(0), b = graph.createNode(1);
  auto c = graph.createNode(2);
  a->outgoing().insert(b, 10);
  a->outgoing().insert(b, 11);
  b->outgoing().insert(c, 20);
  Pattern pattern;
  auto node = pattern.matchNode();
  auto incoming = node->matchIncoming();
  incoming->matchSrc(0)->matchValue([](int v) { return v == 0; });
  auto outgoing = node->matchOutgoing();
  outgoing->matchDst()->matchValue([](int v) { return v == 2; });
  std::vector<int> values;
  for (const auto &match : denox::algorithm::match_all(pattern, a)) {
    EXPECT_EQ(match[node], b);
    EXPECT_EQ(match[outgoing].value(), 20);
    values.push_back(match[incoming].value());
  }
  std::sort(values.begin(), values.end());
  EXPECT_EQ(values, (std::vector<int>{10, 11}));
}

// Payloads 10 and 20 stand for Conv and Add. The generator starts at x,
// which has no outgoing edges; every binding must come from incoming matching.
struct ConvConvAddPattern {
  Pattern pattern;
  Pattern::NP x = pattern.matchNode();
  Pattern::EP add = x->matchIncoming();
  Pattern::NP ca = add->matchSrc(0), cb = add->matchSrc(1);
  Pattern::EP convA = ca->matchIncoming(), convB = cb->matchIncoming();
  Pattern::NP a = convA->matchSrc(0), b = convB->matchSrc(0);

  ConvConvAddPattern() {
    add->matchRank(2);
    add->matchValue([](int op) { return op == 20; });
    for (auto conv : {convA, convB}) {
      conv->matchRank(1);
      conv->matchValue([](int op) { return op == 10; });
    }
  }
};

TEST(MutMatch, ConvConvAddStartingAtOutputBindsEntirePattern) {
  Graph graph;
  auto a = graph.createNode(1), b = graph.createNode(2);
  auto ca = graph.createNode(3), cb = graph.createNode(4);
  auto x = graph.createNode(5);
  auto ea = a->outgoing().insert(ca, 10);
  auto eb = b->outgoing().insert(cb, 10);
  auto sum = ca->outgoing().insert(cb, x, 20);
  ConvConvAddPattern p;
  unsigned count = 0;
  for (const auto &m : denox::algorithm::match_all(p.pattern, x)) {
    ++count;
    EXPECT_EQ(m[p.x], x);
    EXPECT_EQ(m[p.ca], ca);
    EXPECT_EQ(m[p.cb], cb);
    EXPECT_EQ(m[p.a], a);
    EXPECT_EQ(m[p.b], b);
    EXPECT_EQ(m[p.convA].ptr(), ea.operator->());
    EXPECT_EQ(m[p.convB].ptr(), eb.operator->());
    EXPECT_EQ(m[p.add].ptr(), sum.operator->());
  }
  EXPECT_EQ(count, 1u);
}

TEST(MutMatch, ConvConvAddStartingAtOutputEnumeratesProducerCombinations) {
  Graph graph;
  auto a = graph.createNode(1), b = graph.createNode(2);
  auto ca = graph.createNode(3), cb = graph.createNode(4);
  auto x = graph.createNode(5);
  for (int i = 0; i < 2; ++i)
    a->outgoing().insert(ca, 10);
  for (int i = 0; i < 3; ++i)
    b->outgoing().insert(cb, 10);
  ca->outgoing().insert(cb, x, 20);
  ConvConvAddPattern p;
  std::vector<std::pair<const Graph::Edge *, const Graph::Edge *>> pairs;
  for (const auto &m : denox::algorithm::match_all(p.pattern, x)) {
    EXPECT_EQ(m[p.a], a);
    EXPECT_EQ(m[p.b], b);
    const auto pair = std::make_pair(m[p.convA].ptr(), m[p.convB].ptr());
    EXPECT_EQ(std::find(pairs.begin(), pairs.end(), pair), pairs.end());
    pairs.push_back(pair); // Identity only; this test does not mutate edges.
  }
  EXPECT_EQ(pairs.size(), 6u);
}

TEST(MutMatch, ConvConvAddStartingAtOutputRejectsWrongOpsAndRanks) {
  // Independently break Add's tag/rank and each Conv's tag/rank.
  for (unsigned defect = 0; defect < 6; ++defect) {
    SCOPED_TRACE(defect);
    Graph graph;
    auto a = graph.createNode(1), b = graph.createNode(2);
    auto ca = graph.createNode(3), cb = graph.createNode(4);
    auto x = graph.createNode(5);
    if (defect == 3)
      a->outgoing().insert(b, ca, 10);
    else
      a->outgoing().insert(ca, defect == 2 ? 99 : 10);
    if (defect == 5)
      b->outgoing().insert(a, cb, 10);
    else
      b->outgoing().insert(cb, defect == 4 ? 99 : 10);
    if (defect == 1)
      ca->outgoing().insert(x, 20);
    else
      ca->outgoing().insert(cb, x, defect == 0 ? 99 : 20);
    ConvConvAddPattern p;
    unsigned count = 0;
    for ([[maybe_unused]] const auto &m :
         denox::algorithm::match_all(p.pattern, x))
      ++count;
    EXPECT_EQ(count, 0u);
  }
}

TEST(MutMatch, SparseSourceConstraintsAndMissingSource) {
  Graph graph;
  auto a = graph.createNode(0), b = graph.createNode(1);
  auto c = graph.createNode(2);
  a->outgoing().insert(b, c, 10);
  for (unsigned index : {1u, 2u}) {
    Pattern pattern;
    auto edge = pattern.matchNode()->matchIncoming();
    auto source = edge->matchSrc(index);
    unsigned count = 0;
    for (const auto &match : denox::algorithm::match_all(pattern, a)) {
      ++count;
      EXPECT_EQ(match[source], b);
    }
    EXPECT_EQ(count, index == 1 ? 1u : 0u);
  }
}

TEST(MutMatch, ErasesIncomingMatchesWhileEnumerating) {
  Graph graph;
  auto a = graph.createNode(0), b = graph.createNode(1);
  a->outgoing().insert(b, 10);
  a->outgoing().insert(b, 11);
  Pattern pattern;
  auto edge = pattern.matchNode()->matchIncoming();
  edge->matchSrc(0);
  unsigned count = 0;
  for (const auto &match : denox::algorithm::match_all(pattern, a)) {
    ASSERT_LT(count, 2u);
    auto matched = match[edge];
    matched.erase();
    ++count;
  }
  EXPECT_EQ(count, 2u);
  EXPECT_EQ(a->outgoing().size(), 0u);
  EXPECT_EQ(b->incoming().size(), 0u);
}

TEST(MutMatch, EnumeratesOutgoingAlternatives) {
  Graph graph;
  auto root = graph.createNode(0);
  auto left = graph.createNode(1);
  auto right = graph.createNode(2);
  root->outgoing().insert(left, 11);
  root->outgoing().insert(right, 22);

  Pattern pattern;
  auto rootPattern = pattern.matchNode();
  auto edgePattern = rootPattern->matchOutgoing();
  auto dstPattern = edgePattern->matchDst();
  (void)dstPattern;

  std::vector<int> values;
  for (const auto &match : denox::algorithm::match_all(pattern, root))
    values.push_back(match[edgePattern].value());

  ASSERT_EQ(values.size(), 2u);
  EXPECT_NE(std::find(values.begin(), values.end(), 11), values.end());
  EXPECT_NE(std::find(values.begin(), values.end(), 22), values.end());
}

TEST(MutMatch, AppliesEdgeAndDestinationPredicates) {
  Graph graph;
  auto root = graph.createNode(0);
  auto accepted = graph.createNode(1);
  auto rejected = graph.createNode(2);
  root->outgoing().insert(accepted, 7);
  root->outgoing().insert(rejected, 9);

  Pattern pattern;
  auto rootPattern = pattern.matchNode();
  auto edgePattern = rootPattern->matchOutgoing();
  auto destinationPattern = edgePattern->matchDst();
  edgePattern->matchValue([](int value) { return value == 7; });
  destinationPattern->matchValue([](int value) { return value == 1; });

  unsigned matches = 0;
  for (const auto &match : denox::algorithm::match_all(pattern, root)) {
    ++matches;
    EXPECT_EQ(match[edgePattern].value(), 7);
    EXPECT_EQ(match[destinationPattern]->id(), accepted->id());
  }
  EXPECT_EQ(matches, 1u);
}

TEST(MutMatch, SearchesEveryReachableRoot) {
  Graph graph;
  auto root = graph.createNode(0);
  auto middle = graph.createNode(1);
  auto leaf = graph.createNode(2);
  root->outgoing().insert(middle, 10);
  middle->outgoing().insert(leaf, 20);

  Pattern pattern;
  auto nodePattern = pattern.matchNode();
  auto edgePattern = nodePattern->matchOutgoing();
  auto destinationPattern = edgePattern->matchDst();
  (void)destinationPattern;

  std::vector<int> values;
  for (const auto &match : denox::algorithm::match_all(pattern, root))
    values.push_back(match[edgePattern].value());

  ASSERT_EQ(values.size(), 2u);
  EXPECT_NE(std::find(values.begin(), values.end(), 10), values.end());
  EXPECT_NE(std::find(values.begin(), values.end(), 20), values.end());
}

TEST(MutMatch, DoesNotReuseOneEdgeForTwoRequirements) {
  Graph graph;
  auto root = graph.createNode(0);
  auto left = graph.createNode(1);
  auto right = graph.createNode(2);
  root->outgoing().insert(left, 11);
  root->outgoing().insert(right, 22);

  Pattern pattern;
  auto rootPattern = pattern.matchNode();
  auto first = rootPattern->matchOutgoing();
  auto second = rootPattern->matchOutgoing();
  (void)first;
  (void)second;

  unsigned matches = 0;
  for (const auto &match : denox::algorithm::match_all(pattern, root)) {
    ++matches;
    EXPECT_NE(match[first].ptr(), match[second].ptr());
  }
  EXPECT_EQ(matches, 2u);
}
// Positive payloads represent slice offsets; composing slices adds offsets.
// This mirrors SliceSlice's pattern and exact insert-before-erase sequence.
void expectSinglePassFusion(unsigned length, bool retainIntermediates,
                            bool addSibling) {
  Graph graph;
  auto root = graph.createNode(0);
  std::vector<Graph::NodeHandle> nodes{root};
  for (unsigned i = 1; i <= length; ++i) {
    auto next = graph.createNode(static_cast<int>(i));
    nodes.back()->outgoing().insert(next, static_cast<int>(i));
    nodes.push_back(next);
  }
  auto leaf = nodes.back();
  auto sibling = graph.createNode(-1);
  if (addSibling)
    root->outgoing().insert(sibling, -1);
  if (!retainIntermediates)
    nodes.clear(); // Exercise collection as matcher-held handles are released.

  Pattern pattern;
  auto a = pattern.matchNode();
  auto ab = a->matchOutgoing();
  auto b = ab->matchDst();
  auto bc = b->matchOutgoing();
  auto c = bc->matchDst();
  ab->matchRank(1);
  bc->matchRank(1);
  ab->matchValue([](int value) { return value > 0; });
  bc->matchValue([](int value) { return value > 0; });
  b->matchInDeg(1);
  b->matchOutDeg(1);

  unsigned rewrites = 0;
  // Deliberately invoke match_all only once: restarting would hide failures
  // to discover newly inserted fused edges during this traversal.
  for (const auto &match : denox::algorithm::match_all(pattern, root)) {
    ASSERT_LT(rewrites, length - 1) << "Repeated or stale match";
    auto source = match[a];
    auto destination = match[c];
    auto first = match[ab];
    auto second = match[bc];
    const int offset = first.value() + second.value();
    source->outgoing().insert_after(first.nextOutgoingIterator(), destination,
                                    offset);
    first.erase();
    ++rewrites;
    // No EdgeMatch or iterator escapes the current yield.
  }

  EXPECT_EQ(rewrites, length - 1);
  unsigned fused = 0, siblings = 0;
  for (const auto &edge : root->outgoing()) {
    if (edge.value() == -1) {
      ++siblings;
      EXPECT_EQ(edge.dst().id(), sibling->id());
    } else {
      ++fused;
      EXPECT_EQ(edge.dst().id(), leaf->id());
      EXPECT_EQ(edge.value(), static_cast<int>(length * (length + 1) / 2));
    }
  }
  EXPECT_EQ(fused, 1u);
  EXPECT_EQ(siblings, addSibling ? 1u : 0u);
  EXPECT_EQ(root->outgoing().size(), addSibling ? 2u : 1u);
}

TEST(MutMatch, FusesSliceChainsInSingleTraversal) {
  for (unsigned length : {2u, 3u, 4u, 8u}) {
    SCOPED_TRACE(length);
    expectSinglePassFusion(length, false, false);
  }
}

TEST(MutMatch, FusesTwoSlicesWithPinnedIntermediate) {
  // Keeping the old intermediate alive also keeps its outgoing edge alive.
  // Longer chains then have a shared destination (indegree 2), intentionally
  // excluded by SliceSlice's matchInDeg(1) constraint.
  expectSinglePassFusion(2, true, false);
}

TEST(MutMatch, FusesSliceChainWithoutLosingSiblingEdge) {
  expectSinglePassFusion(4, false, true);
}
} // namespace

#pragma once

#include "algorithm/pattern_matching/GraphPatternControlBlock.hpp"
#include "algorithm/pattern_matching/NodePattern.fwd.hpp"
#include "memory/container/shared_ptr.hpp"
#include "memory/container/vector.hpp"
#include "memory/hypergraph/ConstGraph.hpp"
#include "memory/hypergraph/LinkedGraph.hpp"
#include <functional>
#include <limits>
#include <vector>
namespace denox::algorithm {

template <typename V, typename E, typename W> class EdgePattern {
public:
  using rank_t = unsigned int;
  static constexpr rank_t DONT_CARE = std::numeric_limits<rank_t>::max();

  using CB = pattern_matching::details::GraphPatternControlBlock<V, E, W>;
  EdgePattern(CB *p, std::size_t id)
      : m_p(p), m_id(id), m_dst(nullptr),
        m_valuePredicate([](const E &) { return true; }),
        m_weightPredicate([](const W &) { return true; }) {}

  void matchValue(std::function<bool(const E &e)> predicate) {
    m_valuePredicate = predicate;
  }

  void matchWeight(std::function<bool(const W &e)> predicate) {
    m_weightPredicate = predicate;
  }

  NodePatternHandle<V, E, W> matchDst() {
    auto handle = m_p->matchNode();
    m_dst = handle;
    return handle;
  }

  NodePatternHandle<V, E, W> matchSrc(std::size_t i) {
    auto handle = m_p->matchNode();
    if (i >= m_srcs.size()) {
      m_srcs.resize(i+1);
    }
    m_srcs[i] = handle;
    return handle;
  }

  void matchRank(rank_t rank) { m_rank = rank; }

  bool operator()(const memory::ConstGraph<V, E, W> &graph,
                  memory::EdgeId eid) const {
    if (m_rank != DONT_CARE && m_rank != graph.src(eid).size()) {
      return false;
    }

    return m_valuePredicate(graph.get(eid)) &&
           m_weightPredicate(graph.weight(eid));
  }

  template <typename Allocator>
  bool mutable_predicate(
      const typename memory::LinkedGraph<V, E, W, Allocator>::Edge &edge) const {
    if (m_rank != DONT_CARE && m_rank != edge.srcs().size()) {
      return false;
    }

    return m_valuePredicate(edge.value()) && m_weightPredicate(edge.weight());
  }

  std::size_t getId() const { return m_id; }

  const CB *details() const { return m_p; }

  NodePatternHandle<V, E, W> getDst() { return m_dst; }
  std::span<const NodePatternHandle<V, E, W>> getSrcs() const { return m_srcs; }

private:
  CB *m_p;
  std::size_t m_id;
  NodePatternHandle<V, E, W> m_dst; // <- must exists if != nullptr
  std::vector<NodePatternHandle<V, E, W>> m_srcs; // <- elements may be nullptr!
  std::function<bool(const E &)> m_valuePredicate;
  std::function<bool(const W &)> m_weightPredicate;
  rank_t m_rank = DONT_CARE;
};

} // namespace denox::algorithm

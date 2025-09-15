#pragma once

#include "algorithm/pattern_matching/EdgePattern.fwd.hpp"
#include "algorithm/pattern_matching/GraphPatternControlBlock.hpp"
#include "memory/hypergraph/ConstGraph.hpp"
#include "memory/hypergraph/LinkedGraph.hpp"
#include "memory/hypergraph/NullWeight.hpp"
#include <functional>
#include <limits>
#include <vector>
namespace denox::algorithm {

template <typename V, typename E, typename W = memory::NullWeight>
class NodePattern {
public:
  using deg_t = unsigned int;
  static constexpr deg_t DONT_CARE = std::numeric_limits<deg_t>::max();
  using CB = pattern_matching::details::GraphPatternControlBlock<V, E, W>;

  NodePattern(CB *p, std::size_t id)
      : m_p(p), m_id(id), m_valuePred([](const V &) { return true; }) {}

  NodePattern(const NodePattern &) = delete;
  NodePattern &operator=(const NodePattern &) = delete;

  NodePattern(NodePattern &&) = delete;
  NodePattern &operator=(NodePattern &&) = delete;

  void matchValue(std::function<bool(const V &)> predicate) {
    m_valuePred = predicate;
  }

  void matchInDeg(deg_t indeg) { m_indeg = indeg; }

  void matchOutDeg(deg_t outdeg) { m_outdeg = outdeg; }

  EdgePatternHandle<V, E, W> matchOutgoing() {
    auto handle = m_p->matchEdge();
    m_outgoing.push_back(handle);
    return handle;
  }

  bool operator()(const memory::ConstGraph<V, E, W> &graph,
                  memory::NodeId nid) const {
    if (m_indeg != DONT_CARE && graph.incoming(nid).size() != m_indeg) {
      return false;
    }
    if (m_outdeg != DONT_CARE && graph.outgoing(nid).size() != m_outdeg) {
      return false;
    }

    const auto &v = graph.get(nid);
    return m_valuePred(v);
  }

  template <typename Allocator>
  bool mutable_predicate(
      const typename memory::LinkedGraph<V, E, W, Allocator>::NodeHandle &node)
      const {
    if (m_indeg != DONT_CARE && node->incoming().size() != m_indeg) {
      return false;
    }
    if (m_outdeg != DONT_CARE && node->outgoing().size() != m_outdeg) {
      return false;
    }

    const auto &v = node->value();
    return m_valuePred(v);
  }

  std::size_t getId() const { return m_id; }

  const CB *details() const { return m_p; }

  std::span<const EdgePatternHandle<V, E, W>> getOutgoing() const {
    return m_outgoing;
  }

private:
  CB *m_p;
  std::size_t m_id;
  std::function<bool(const V &)> m_valuePred;
  std::vector<EdgePatternHandle<V, E, W>> m_outgoing; // <- all must exist!
  deg_t m_indeg = DONT_CARE;
  deg_t m_outdeg = DONT_CARE;
};

} // namespace denox::algorithm

#pragma once

#include "algorithm/pattern_matching/NodePattern.fwd.hpp"
#include "algorithm/pattern_matching/NodePattern.hpp"
#include "memory/hypergraph/EdgeId.hpp"
#include "memory/hypergraph/NodeId.hpp"

#include "algorithm/pattern_matching/EdgePattern.fwd.hpp"
#include "algorithm/pattern_matching/EdgePattern.hpp"
#include "memory/hypergraph/NullWeight.hpp"
#include <cassert>

namespace denox::algorithm {

template <typename V, typename E, typename W = memory::NullWeight> class ConstGraphMatch {
public:
  ConstGraphMatch(std::size_t nodeCount, std::size_t edgeCount)
      : m_nodeMatches(nodeCount), m_edgeMatches(edgeCount) {}

  memory::NodeId operator[](const NodePatternHandle<V, E, W> &pattern) const {
    return m_nodeMatches[pattern->getId()];
  }

  memory::EdgeId operator[](const EdgePatternHandle<V, E, W> &pattern) const {
    return m_edgeMatches[pattern->getId()];
  }

  void registerMatch(const NodePatternHandle<V, E, W> &pattern,
                     memory::NodeId id) {
    m_nodeMatches[pattern->getId()] = id;
  }

  void registerMatch(const EdgePatternHandle<V, E, W> &pattern,
                     memory::EdgeId id) {
    m_edgeMatches[pattern->getId()] = id;
  }

  [[nodiscard]] bool mergeMatches(const ConstGraphMatch<V, E, W> &match) {
    assert(match.m_nodeMatches.size() == m_nodeMatches.size());
    assert(match.m_edgeMatches.size() == m_edgeMatches.size());
    for (std::size_t i = 0; i < match.m_nodeMatches.size(); ++i) {
      if (match.m_nodeMatches[i] == memory::NodeId{} ||
          m_nodeMatches[i] == memory::NodeId{}) {
        continue;
      }
      if (match.m_nodeMatches[i] != m_nodeMatches[i]) {
        return false; // <- collision!
      }
    }
    for (std::size_t i = 0; i < match.m_edgeMatches.size(); ++i) {
      if (match.m_edgeMatches[i] == memory::EdgeId{} ||
          m_edgeMatches[i] == memory::EdgeId{}) {
        continue;
      }
      if (match.m_edgeMatches[i] != m_edgeMatches[i]) {
        return false; // <- collision!
      }
    }
    for (std::size_t i = 0; i < match.m_nodeMatches.size(); ++i) {
      if (match.m_nodeMatches[i] != memory::NodeId{}) {
        m_nodeMatches[i] = match.m_nodeMatches[i];
      }
    }
    for (std::size_t i = 0; i < match.m_edgeMatches.size(); ++i) {
      if (match.m_edgeMatches[i] != memory::EdgeId{}) {
        m_edgeMatches[i] = match.m_edgeMatches[i];
      }
    }
    return true;
  }

private:
  std::vector<memory::NodeId> m_nodeMatches;
  std::vector<memory::EdgeId> m_edgeMatches;
};

} // namespace denox::algorithm

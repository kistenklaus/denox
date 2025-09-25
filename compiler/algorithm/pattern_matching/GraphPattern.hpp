#pragma once

#include "algorithm/pattern_matching/EdgePattern.fwd.hpp"
#include "algorithm/pattern_matching/GraphPatternControlBlock.hpp"
#include "algorithm/pattern_matching/NodePattern.fwd.hpp"
#include "memory/hypergraph/NullWeight.hpp"
#include <memory>
#include <stdexcept>
#include <variant>

namespace denox::algorithm {

template <typename V, typename E, typename W = memory::NullWeight>
class GraphPattern {
public:
  using NP = NodePatternHandle<V, E, W>;

  using EP = EdgePatternHandle<V, E, W>;

  using Root = std::variant<std::monostate, NP, EP>;

  GraphPattern()
      : m_controlBlock(
            std::make_unique<pattern_matching::details::
                                 GraphPatternControlBlock<V, E, W>>()) {}

  NP matchNode() {
    if (!std::holds_alternative<std::monostate>(m_root)) {
      throw std::runtime_error(
          "Either match nodes or edges, we do not support both.");
    }
    auto handle = m_controlBlock->matchNode();
    m_root = handle;
    return handle;
  }

  EP matchEdge() {
    if (!std::holds_alternative<std::monostate>(m_root)) {
      throw std::runtime_error(
          "Either match nodes or edges, we do not support both.");
    }
    auto handle = m_controlBlock->matchEdge();
    m_root = handle;
    return handle;
  }

  const Root &root() const { return m_root; }

  const pattern_matching::details::GraphPatternControlBlock<V, E, W> *
  details() const {
    return m_controlBlock.get();
  }

private:
  std::unique_ptr<pattern_matching::details::GraphPatternControlBlock<V, E, W>>
      m_controlBlock;

  Root m_root;
};

} // namespace denox::algorithm

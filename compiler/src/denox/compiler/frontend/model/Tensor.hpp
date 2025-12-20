#pragma once

#include "denox/memory/container/optional.hpp"
#include "denox/memory/dtype/dtype.hpp"
#include "denox/memory/hypergraph/NodeId.hpp"
#include "denox/memory/tensor/ActivationLayout.hpp"
#include "denox/compiler/frontend/model/ModelControlBlock.hpp"
#include "denox/symbolic/Symbolic.hpp"

namespace denox::compiler {

class Model;

class Tensor {
public:
  friend Model;

  memory::optional<memory::ActivationLayout> layout() const {
    return m_controlBlock->hypergraph.get(m_nodeId).m_layout;
  }

  void setLayout(memory::optional<memory::ActivationLayout> layout) {
    m_controlBlock->hypergraph.get(m_nodeId).m_layout = layout;
  }

  memory::optional<memory::Dtype> type() const {
    return m_controlBlock->hypergraph.get(m_nodeId).m_type;
  }

  void setType(memory::optional<memory::Dtype> type) {
    m_controlBlock->hypergraph.get(m_nodeId).m_type = type;
  }

  unsigned int channels() const {
    return m_controlBlock->hypergraph.get(m_nodeId).m_channels;
  }

  Symbolic width() const {
    return Symbolic(
        &m_controlBlock->symGraph,
        m_controlBlock->hypergraph.get(m_nodeId).m_extent.x.asSym());
  }

  Symbolic height() const {
    return Symbolic(
        &m_controlBlock->symGraph,
        m_controlBlock->hypergraph.get(m_nodeId).m_extent.y.asSym());
  }

  std::uint64_t id() const { return static_cast<std::uint64_t>(m_nodeId); }
  Tensor() : m_nodeId(memory::NodeId(0)), m_controlBlock(nullptr) {}

private:
  Tensor(memory::NodeId id, details::model::ModelControlBlock *controlBlock)
      : m_nodeId(id), m_controlBlock(controlBlock) {}

  memory::NodeId m_nodeId;
  details::model::ModelControlBlock *m_controlBlock;
};

} // namespace denox::compiler

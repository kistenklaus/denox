#pragma once

#include "denox/compiler/frontend/model/ModelControlBlock.hpp"
#include "denox/memory/hypergraph/NodeId.hpp"
#include "denox/symbolic/Symbolic.hpp"

namespace denox::compiler {

class Model;

class Tensor {
public:
  friend Model;

  TensorDataType type() const {
    return m_controlBlock->hypergraph.get(m_nodeId).type();
  }

  void setType(TensorDataType type) {
    m_controlBlock->hypergraph.get(m_nodeId).setType(type);
  }

  Sym channels() const {
    return m_controlBlock->hypergraph.get(m_nodeId).channels();
  }

  Symbolic width() const {
    return Symbolic(&m_controlBlock->symGraph,
                    m_controlBlock->hypergraph.get(m_nodeId).width());
  }

  Symbolic height() const {
    return Symbolic(&m_controlBlock->symGraph,
                    m_controlBlock->hypergraph.get(m_nodeId).height());
  }

  TensorStorage storage() const {
    return m_controlBlock->hypergraph.get(m_nodeId).storage();
  }

  void setStorage(TensorStorage storage) {
    m_controlBlock->hypergraph.get(m_nodeId).setStorage(storage);
  }

  TensorFormat format() const {
    return m_controlBlock->hypergraph.get(m_nodeId).format();
  }

  void setFormat(TensorFormat format) {
    m_controlBlock->hypergraph.get(m_nodeId).setFormat(format);
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

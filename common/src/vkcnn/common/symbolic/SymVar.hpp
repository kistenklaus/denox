#pragma once

#include "vkcnn/common/hypergraph/NodeId.hpp"
#include <cstddef>
namespace vkcnn {

class SymAdjGraph;
class SymConstGraph;
class SymEval;
class SymId;

class SymVar {
public:
  friend SymAdjGraph;
  friend SymConstGraph;
  friend SymEval;
  friend SymId;

private:
  explicit SymVar(std::size_t varId, hypergraph::NodeId nodeId)
      : m_varId(varId), m_nodeId(nodeId) {}
  std::size_t m_varId;
  hypergraph::NodeId m_nodeId;
};

} // namespace vkcnn

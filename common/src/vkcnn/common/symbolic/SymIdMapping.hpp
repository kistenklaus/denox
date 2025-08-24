#pragma once

#include "vkcnn/common/hypergraph/NodeId.hpp"
#include <vector>
namespace vkcnn {

class SymConstGraph;

class SymIdMapping {
public:
  friend SymConstGraph;
  hypergraph::NodeId operator()(hypergraph::NodeId id) const {
    return m_mapping[id];
  }

private:
  SymIdMapping(std::vector<hypergraph::NodeId> mapping) : m_mapping(mapping) {}
  std::vector<hypergraph::NodeId> m_mapping;
};

} // namespace vkcnn

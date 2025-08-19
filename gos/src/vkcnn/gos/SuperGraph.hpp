#pragma once

#include "vkcnn/common/cg/ComputeGraph.hpp"
#include "vkcnn/common/hypergraph/AdjGraph.hpp"

namespace vkcnn::gos {

class SuperGraph;

namespace details {
struct SuperGraphControlBlock {
  friend SuperGraph;

private:
  hypergraph::AdjGraph<int, int> hypergraph;
};

} // namespace details

class SuperGraph {
public:
private:
  std::shared_ptr<details::SuperGraphControlBlock> m_controlBlock;
};

} // namespace vkcnn::gos

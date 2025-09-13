#pragma once

#include "memory/hypergraph/ConstGraph.hpp"
#include "memory/hypergraph/NodeId.hpp"
#include "model/ComputeOp.hpp"
#include "model/ComputeTensor.hpp"
#include <bitset>
namespace denox::compiler {

class IFusionRule {
public:
  virtual ~IFusionRule() = default;

  virtual bool apply(const memory::ConstGraph<ComputeTensor, ComputeOp> in,
                     memory::AdjGraph<ComputeTensor, ComputeOp> &out,
                     memory::NodeId inId, memory::NodeId outId,
                     std::vector<bool> &touched) const = 0;

private:
};

} // namespace denox::compiler

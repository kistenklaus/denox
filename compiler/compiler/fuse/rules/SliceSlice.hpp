#pragma once

#include "compiler/fuse/IFusionRule.hpp"
#include "memory/hypergraph/EdgeId.hpp"
namespace denox::compiler::fusion {

class SliceSlice final : public IFusionRule {
public:
  bool apply(const memory::ConstGraph<ComputeTensor, ComputeOp> in,
             memory::AdjGraph<ComputeTensor, ComputeOp> &out,
             memory::NodeId inId, memory::NodeId outId,
             std::vector<bool> &touched) const final override {
    auto outgoing = in.outgoing(inId);
    if (outgoing.size() != 1) {
      return false;
    }
    memory::EdgeId outEdge = outgoing[0];
    const ComputeOp &op = in.get(outEdge);
    if (op.tag() != ComputeOpTag::Slice) {
      return false;
    }
    memory::NodeId n2 = in.dst(outEdge);

    auto outgoing2 = in.outgoing(n2);
    if (outgoing2.size() != 1) {
      return false;
    }
    auto outEdge2 = outgoing2[0];
    const ComputeOp op2 = in.get(outEdge2);
    if (op2.tag() != ComputeOpTag::Slice) {
      return false;
    }
    memory::NodeId n3 = in.dst(outEdge2);
    touched[static_cast<std::size_t>(inId)] = true;
    touched[static_cast<std::size_t>(n2)] = true;
    touched[static_cast<std::size_t>(n3)] = true;

    return false;
  }

private:
};

} // namespace denox::compiler::fusion

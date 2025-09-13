#include "compiler/fuse/fusion.hpp"
#include "compiler/fuse/rules/SliceSlice.hpp"
#include "memory/container/dynamic_bitset.hpp"
#include "memory/container/vector.hpp"
#include "memory/hypergraph/AdjGraph.hpp"
#include "memory/hypergraph/NodeId.hpp"
#include <array>

namespace denox::compiler {

static fusion::SliceSlice rule_sliceSlice;

static IFusionRule *fusion_rules[] = {
    &rule_sliceSlice,
};

static std::optional<memory::ConstGraph<ComputeTensor, ComputeOp>>
apply_rule(const memory::ConstGraph<ComputeTensor, ComputeOp> &in,
           const IFusionRule &rule) {
  memory::AdjGraph<ComputeTensor, ComputeOp> out;

  memory::dynamic_bitset touched(in.nodeCount());

  // maps in NodeIds to out NodeIds.
  memory::vector<memory::NodeId> remap;
  remap.reserve(in.nodeCount());
  for (std::uint64_t id = 0; id < in.nodeCount(); ++id) {
    remap.emplace_back(id);
  }

  for (std::uint64_t id = 0; id < in.nodeCount(); ++id) {
    memory::NodeId inId{id};
    if (touched[id]) {
      continue;
    }
    memory::NodeId outId{remap[id]};

    if (rule.apply(in, out, inId, outId, touched)) {
      // case handled by rule!
      // may have modify touched and out
      continue;
    } else {
    }
  }

  return memory::ConstGraph<ComputeTensor, ComputeOp>{out};
}


memory::ConstGraph<ComputeTensor, ComputeOp>
fusion_pass(const memory::ConstGraph<ComputeTensor, ComputeOp>& graph) {
  return graph;
}

} // namespace denox::compiler

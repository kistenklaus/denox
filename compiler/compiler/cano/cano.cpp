#include "compiler/cano/cano.hpp"
#include "algorithm/pattern_matching/match.hpp"
#include "compiler/cano/rules/IFusionRule.hpp"
#include "compiler/cano/rules/SliceSlice.hpp"
#include "compiler/ir/LinkedModel.hpp"
#include "diag/logging.hpp"
#include "memory/hypergraph/LinkedGraph.hpp"
#include <stdexcept>

namespace denox::compiler {

LinkedModel canonicalize(const Model &model) {
  // 1. Build LinkedGraph
  using LinkedGraph = memory::LinkedGraph<ComputeTensor, ComputeOp>;
  auto [mapping, graph] = LinkedGraph::from(model.graph());
  LinkedGraph::NodeHandle input = mapping[model.getInput().id()];
  LinkedGraph::NodeHandle output = mapping[model.getOutput().id()];
  // NOTE: mapping.clear() drops all references to internal tensors, which
  // may implicitly remove dead branches.
  // Afterwards the graph only contains nodes and operations
  // which are dependent on the input.
  // However, it may still contain nodes or edges, which do not
  // contribute to the output!
  mapping.clear();

  if (output->incoming().size() == 0) {
    DENOX_ERROR("Models output does not depend on the input. Denox does not "
                "support constant outputs!");
    throw std::runtime_error("Failed to canonicalize.");
  }



  cano::SliceSlice sliceSliceRule;

  cano::IFusionRule *rules[] = {
      &sliceSliceRule,
  };

  for (const auto &rule : rules) {
    const auto &pattern = rule->pattern();
    for (const auto &match : algorithm::match_all(pattern, input)) {
      rule->apply(match);
    }
  }
  LinkedModel m{
      .graph = std::move(graph),
      .input = std::move(input),
      .output = std::move(output),
  };

  return m;
}

} // namespace denox::compiler

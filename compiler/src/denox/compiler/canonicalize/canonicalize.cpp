#include "compiler/cano/cano.hpp"

#include "Options.hpp"
#include "compiler/cano/rules/IFusionRule.hpp"
#include "compiler/cano/rules/SliceSlice.hpp"
#include "compiler/ir/CanoModel.hpp"
#include "denox/algorithm/pattern_matching/match.hpp"
#include "denox/diag/logging.hpp"
#include "denox/memory/hypergraph/LinkedGraph.hpp"
#include "denox/symbolic/SymGraph.hpp"
#include <stdexcept>

namespace denox::compiler {

CanoModel canonicalize(const Model &model, const Options &options) {
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

  memory::vector<cano::IFusionRule *> rules;

  cano::SliceSlice sliceSliceRule;
  if (options.fusionRules.enableSliceSliceFusion) {
    rules.push_back(&sliceSliceRule);
  }

  SymGraph symGraph = model.symGraph();

  for (const auto &rule : rules) {
    const auto &pattern = rule->pattern();
    for (const auto &match : algorithm::match_all(pattern, input)) {
      rule->apply(symGraph, match);
    }
  }
  CanoModel m{
      .graph = std::move(graph),
      .input = std::move(input),
      .output = std::move(output),
      .symGraph = std::move(symGraph),
  };

  return m;
}

} // namespace denox::compiler

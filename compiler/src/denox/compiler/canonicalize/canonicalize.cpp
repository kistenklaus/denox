#include "denox/compiler/canonicalize/canonicalize.hpp"

#include "denox/algorithm/pattern_matching/match.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/compiler/canonicalize/CanoModel.hpp"
#include "denox/compiler/canonicalize/rules/IFusionRule.hpp"
#include "denox/compiler/canonicalize/rules/SliceSlice.hpp"
#include "denox/diag/logging.hpp"
#include "denox/memory/hypergraph/LinkedGraph.hpp"
#include "denox/symbolic/SymGraph.hpp"
#include <stdexcept>

namespace denox::compiler {

CanoModel canonicalize(const Model &model) {
  // 1. Build LinkedGraph
  using LinkedGraph = memory::LinkedGraph<TensorDescriptor, ComputeOp>;
  auto [mapping, graph] = LinkedGraph::from(model.graph());

  std::vector<LinkedGraph::NodeHandle> inputs;
  for (const auto &input : model.getInputs()) {
    inputs.push_back(mapping[input.id()]);
  }
  std::vector<LinkedGraph::NodeHandle> outputs;
  for (const auto &output : model.getOutputs()) {
    outputs.push_back(mapping[output.id()]);
  }

  // NOTE: mapping.clear() drops all references to internal tensors, which
  // may implicitly remove dead branches.
  // Afterwards the graph only contains nodes and operations
  // which are dependent on the input.
  // However, it may still contain nodes or edges, which do not
  // contribute to the output!
  mapping.clear();

  for (const auto &output : outputs) {
    if (output->incoming().size() == 0) {
      DENOX_ERROR("Models output does not depend on the input. Denox does not "
                  "support constant outputs!");
      throw std::runtime_error("Failed to canonicalize.");
    }
  }
  cano::SliceSlice sliceSliceRule;

  memory::vector<cano::IFusionRule *> rules{&sliceSliceRule};

  SymGraph symGraph = model.symGraph();

  LinkedGraph::NodeHandle root = graph.createNode(
      TensorDescriptor(Sym::Const(0), Sym::Const(0), Sym::Const(0),
        TensorStorage::Optimal, TensorFormat::Optimal, TensorDataType::Auto));

  for (const auto &input : inputs) {
    root->outgoing().insert(input, ComputeOp{});
  }

  for (const auto &rule : rules) {
    const auto &pattern = rule->pattern();
    for (const auto &match : algorithm::match_all(pattern, root)) {
      rule->apply(symGraph, match);
    }
  }
  CanoModel m{
      .graph = std::move(graph),
      .inputs = std::move(inputs),
      .outputs = std::move(outputs),
      .symGraph = std::move(symGraph),
  };

  return m;
}

} // namespace denox::compiler

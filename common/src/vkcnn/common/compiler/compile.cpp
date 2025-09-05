#include "./compile.hpp"
#include "vkcnn/common/compiler/debug.hpp"
#include "vkcnn/common/compiler/specialize_tensors.hpp"
#include "vkcnn/common/hypergraph/ConstGraph.hpp"
#include "vkcnn/common/mr/MemoryRequirements.hpp"
#include <fmt/base.h>

namespace vkcnn {

CompiledModel compile(Model model, const CompileOptions &options) {

  fmt::println("Model:");
  debug::print_const_graph(
      hypergraph::ConstGraph<ComputeTensor, ComputeOp>(model.graph()));

  hypergraph::ConstGraph<compiler::SpecializedTensor, ComputeOp>
      specializedGraph = compiler::specialize_tensors(model, options);

  hypergraph::AdjGraph<int, int> supergraph;
  // Build supergraph, from specialized graph.

  // Freeze supergraph, convert into a datastructure more suitable for
  // traversal.
  hypergraph::ConstGraph<int, int> constSupergraph(supergraph);

  CmdOpBuffer cmdBuffer;
  CmdOpSchedule schedule{cmdBuffer};
  MemoryRequirements memoryRequirements;

  // NOTE: It would be really nice if CompileModel maps linear to memory!
  return CompiledModel{std::move(schedule), std::move(memoryRequirements)};
}

} // namespace vkcnn

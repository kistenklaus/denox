#include "./compile.hpp"

namespace vkcnn {

CompiledModel compile(const Model &model) {

  hypergraph::ConstGraph<ComputeTensor, ComputeOp> graph = model.freeze();
  CmdOpBuffer cmd;


  CmdOpSchedule schedule(std::move(cmd));
  MemoryRequirements memoryRequirements;

  return CompiledModel{std::move(schedule), std::move(memoryRequirements)};
}
} // namespace vkcnn

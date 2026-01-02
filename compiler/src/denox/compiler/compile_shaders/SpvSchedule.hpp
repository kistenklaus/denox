#pragma once

#include "denox/compiler/compile_shaders/SpvDispatch.hpp"
#include "denox/compiler/implement/Tensor.hpp"
#include "denox/symbolic/SymGraph.hpp"

namespace denox::compiler {

struct SpvSchedule {
  SymGraph symGraph;
  memory::vector<Tensor> tensors;
  memory::vector<SpvDispatch> dispatches;
  memory::vector<SpirvBinary> binaries;
};

} // namespace denox::compiler

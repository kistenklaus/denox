#pragma once

#include "denox/compiler/compile_shaders/SpvDispatch.hpp"
#include "denox/compiler/placement/Buffer.hpp"
#include "denox/compiler/placement/TensorInitalizer.hpp"
#include "denox/compiler/placement/TensorView.hpp"
#include "denox/symbolic/SymGraph.hpp"

namespace denox::compiler {

struct SpvSchedule {
  SymGraph symGraph;

  memory::vector<TensorView> tensors;
  memory::vector<Buffer> buffers;
  memory::vector<TensorInitializer> initializers;

  memory::vector<SpvDispatch> dispatches;
  memory::vector<SpirvBinary> binaries;

  memory::small_vector<uint64_t, 2> inputs;
  memory::small_vector<uint64_t, 2> outputs;
};

} // namespace denox::compiler

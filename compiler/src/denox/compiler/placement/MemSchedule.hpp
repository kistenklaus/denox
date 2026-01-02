#pragma once

#include "denox/compiler/implement/ComputeDispatch.hpp"
#include "denox/compiler/placement/Buffer.hpp"
#include "denox/compiler/placement/TensorInitalizer.hpp"
#include "denox/compiler/placement/TensorView.hpp"
#include "denox/symbolic/SymGraph.hpp"

namespace denox::compiler {

struct MemSchedule {
  SymGraph symGraph;

  memory::vector<TensorView> tensors;
  memory::vector<Buffer> buffers;
  memory::vector<TensorInitializers> initializers;

  memory::vector<ComputeDispatch> dispatches;

  // indexes into tensors.
  memory::small_vector<uint64_t, 2> inputs;
  memory::small_vector<uint64_t, 2> outputs;
};

} // namespace denox::compiler

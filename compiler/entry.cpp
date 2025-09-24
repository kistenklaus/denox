#include "entry.hpp"
#include "algorithm/count_children.hpp"
#include "compiler/cano/cano.hpp"
#include "compiler/dce.hpp"
#include "compiler/freeze.hpp"
#include "compiler/spec.hpp"
#include "diag/logging.hpp"
#include "diag/unreachable.hpp"
#include "frontend/onnx/onnx.hpp"
#include "memory/hypergraph/ConstGraph.hpp"
#include "memory/hypergraph/LinkedGraph.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include "model/ComputeTensor.hpp"
#include "model/Model.hpp"
#include <fmt/base.h>
#include <google/protobuf/port.h>

namespace denox::compiler {

static Model frontend(memory::span<const std::byte> raw,
                      const Options &options) {
  switch (options.srcType) {
  case SrcType::Onnx: {
    io::Path onnx_dir;
    if (options.srcPath.has_value()) {
      onnx_dir = options.srcPath->absolute().parent();
    } else {
      onnx_dir = options.cwd;
    }
    auto model = denox::onnx::read(raw, onnx_dir);
    model.getInput().setLayout(options.inputLayout);
    model.getOutput().setLayout(options.outputLayout);
    return model;
  }
  }
  denox::compiler::diag::unreachable();
}

void entry(memory::span<const std::byte> raw, const Options &options) {
  // 1. Import model
  Model model = frontend(raw, options);
  // 1. Canoncalize:
  // Stuff that the pytorch exporter doesn't do by default like
  // fusion of slice -> slice, or fusing operations like
  // batch norm into convolutions and so on.
  // Basically just bringing the ir into a canonical form.
  LinkedModel emodel = compiler::canonicalize(model);

  // 2. Specialize:
  // Resolve dtypes and layouts, by possibly
  // duplicating intermediate tensors
  // (e.g. one for allowed each layout)
  compiler::specialize(emodel, memory::ActivationLayout::supported());

  // 3. Dead Code Elimination.
  // Remove all intermediate tensors and edges, if
  // they do not contribute to the output.
  // This should generally be an empty set, but it's good
  // to check.
  AdjModel amodel = compiler::dce(emodel);

  // 4. Freeze:
  // From this point we switch to a datastructure more suited
  // for graph traversal.
  ConstModel cmodel = compiler::freeze(amodel);

  // 5. Implementation:
  // Build supergraph!

  // 6. Schedule
  // Find shortest path through the supergraph, and
  // generate shader sources for all shaders along the path.

  // 7. Memory-Planning:

  // 8. Serialize:
  // Write schedule and memory plan into a DNX buffer.
}
} // namespace denox::compiler

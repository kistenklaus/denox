#include "entry.hpp"
#include "Options.hpp"
#include "compiler/cano/cano.hpp"
#include "compiler/dce.hpp"
#include "compiler/impl/impl.hpp"
#include "compiler/lifeness.hpp"
#include "compiler/placement/placement.hpp"
#include "compiler/spec.hpp"
#include "diag/unreachable.hpp"
#include "frontend/onnx/onnx.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include "model/ComputeTensor.hpp"
#include "model/Model.hpp"
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

  Model model = frontend(raw, options);

  fmt::println("\x1B[32m\x1B[1m{:=^40}\x1B[0m", "Imported=Model");
  fmt::println("{}", model.to_string());
  SymGraph symGraph = model.symGraph();

  CanoModel canoModel = compiler::canonicalize(model, options);

  Lifetimes lifetimes = compiler::lifeness(canoModel);

  SpecModel specModel = compiler::specialize(
      canoModel, lifetimes, memory::ActivationLayout::supported());

  OpModel opModel = compiler::dce(specModel);

  ImplModel implModel = compiler::implement(opModel, symGraph, options);

  compiler::placement(implModel);

  // TODO: Produce barriers. 
  // - Somehow find a way to get information about who is reading and 
  //   who is writing. (Missing from our IR).

  // TODO: Compute lifetimes of buffers based on compModel.
  // - lifetimes based on dispatch index.
  // - this can and should includes weights.
}
} // namespace denox::compiler

#include "entry.hpp"
#include "algorithm/count_children.hpp"
#include "compiler/cano/cano.hpp"
#include "compiler/dce.hpp"
#include "compiler/impl.hpp"
#include "compiler/spec.hpp"
#include "diag/unreachable.hpp"
#include "frontend/onnx/onnx.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include "model/ComputeTensor.hpp"
#include "model/Model.hpp"
#include <absl/strings/str_format.h>
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
  Model model = frontend(raw, options);
  SymGraph symGraph = model.symGraph();

  CanoModel canoModel = compiler::canonicalize(model);

  SpecModel specModel =
      compiler::specialize(canoModel, memory::ActivationLayout::supported());

  OpModel opModel = compiler::dce(specModel);

  compiler::implement(opModel, symGraph);
}
} // namespace denox::compiler

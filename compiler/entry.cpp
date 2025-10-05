#include "entry.hpp"
#include "Options.hpp"
#include "compiler/cano/cano.hpp"
#include "compiler/dce.hpp"
#include "compiler/impl/impl.hpp"
#include "compiler/lifeness.hpp"
#include "compiler/placement/placement.hpp"
#include "compiler/spec.hpp"
#include "compiler/sym_compile.hpp"
#include "compiler/sym_table.hpp"
#include "diag/invalid_argument.hpp"
#include "diag/summary.hpp"
#include "diag/unreachable.hpp"
#include "dnx/serialize.hpp"
#include "frontend/onnx/onnx.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include "model/ComputeTensor.hpp"
#include "model/Model.hpp"

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
    auto model = denox::onnx::read(raw, onnx_dir, options);
    model.getInput().setLayout(options.inputLayout);
    model.getOutput().setLayout(options.outputLayout);
    return model;
  }
  }
  denox::compiler::diag::unreachable();
}

flatbuffers::DetachedBuffer entry(memory::span<const std::byte> raw,
                                  const Options &options) {
  Model model = frontend(raw, options);
  assert(model.getInput().type().has_value());

  if (options.verbose) {
    fmt::println("\x1B[32m\x1B[1m{:=^40}\x1B[0m", "Imported=Model");
    fmt::println("{}", model.to_string());
  }
  if (!options.inputLayout.supports(model.getInput().channels())) {
    compiler::diag::invalid_argument();
  }
  if (!options.outputLayout.supports(model.getOutput().channels())) {
    compiler::diag::invalid_argument();
  }

  SymGraph symGraph = model.symGraph();

  CanoModel canoModel = compiler::canonicalize(model, options);

  Lifetimes lifetimes = compiler::lifeness(canoModel);

  SpecModel specModel = compiler::specialize(
      canoModel, lifetimes, memory::ActivationLayout::supported());

  OpModel opModel = compiler::dce(specModel);

  ImplModel implModel = compiler::implement(opModel, symGraph, options);

  CompModel compModel = compiler::placement(implModel);

  SymTable symTable = compiler::sym_table(model, options);

  auto [symIR, symCount] = compiler::compile_sym_and_remap(compModel);

  auto dnx = dnx::serialize(compModel, symIR);

  if (!options.quite) {
    diag::print_summary(model, implModel, compModel, symIR, symCount, dnx);
  }

  return dnx;
}
} // namespace denox::compiler

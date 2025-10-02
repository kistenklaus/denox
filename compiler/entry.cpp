#include "entry.hpp"
#include "Options.hpp"
#include "compiler/cano/cano.hpp"
#include "compiler/dce.hpp"
#include "compiler/impl/impl.hpp"
#include "compiler/lifeness.hpp"
#include "compiler/placement/placement.hpp"
#include "compiler/spec.hpp"
#include "compiler/sym_compile.hpp"
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
    auto model = denox::onnx::read(raw, onnx_dir);
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

  fmt::println("\x1B[32m\x1B[1m{:=^40}\x1B[0m", "Imported=Model");
  fmt::println("{}", model.to_string());
  SymGraph symGraph = model.symGraph();

  CanoModel canoModel = compiler::canonicalize(model, options);

  Lifetimes lifetimes = compiler::lifeness(canoModel);

  SpecModel specModel = compiler::specialize(
      canoModel, lifetimes, memory::ActivationLayout::supported());

  OpModel opModel = compiler::dce(specModel);

  ImplModel implModel = compiler::implement(opModel, symGraph, options);

  CompModel compModel = compiler::placement(implModel);

  auto [symIR, symCount] = compiler::compile_sym_and_remap(compModel);

  auto dnx = dnx::serialize(compModel, symIR);

  fmt::println("\n\x1B[31mSummary:\x1B[0m");

  fmt::println("\u2022 {:<20} : {}", "Inputs ", compModel.inputs.size());
  fmt::println("\u2022 {:<20} : {}", "Outputs ", compModel.outputs.size());

  fmt::println("\u2022 {:<20} : {}", "Number-Of-Dispatches",
               implModel.dispatches.size());

  std::size_t spirvByteSize = 0;
  for (std::size_t d = 0; d < compModel.dispatches.size(); ++d) {
    spirvByteSize += compModel.dispatches[d].src.size;
  }
  if (spirvByteSize > 1000000) {
    fmt::println("\u2022 {:<20} : {:.1f}MB", "SPIRV-ByteSize",
                 static_cast<float>(spirvByteSize) / 1000000.0f);
  } else if (spirvByteSize > 1000) {
    fmt::println("\u2022 {:<20} : {:.1f}KB", "SPIRV-ByteSize",
                 static_cast<float>(spirvByteSize) / 1000.0f);
  } else {
    fmt::println("\u2022 {:<20} : {}B", "SPIRV-ByteSize", spirvByteSize);
  }

  std::size_t parameterByteSize = 0;
  for (std::size_t p = 0; p < implModel.parameters.size(); ++p) {
    parameterByteSize += implModel.parameters[p].data.size();
  }
  if (parameterByteSize > 1000000) {
    fmt::println("\u2022 {:<20} : {:.1f}MB", "Parameter-ByteSize",
                 static_cast<float>(parameterByteSize) / 1000000.0f);
  } else if (parameterByteSize > 1000) {
    fmt::println("\u2022 {:<20} : {:.1f}KB", "Parameter-ByteSize",
                 static_cast<float>(parameterByteSize) / 1000.0f);
  } else {
    fmt::println("\u2022 {:<20} : {}B", "Parameter-ByteSize",
                 parameterByteSize);
  }
  std::size_t dnxByteSize = dnx.size();
  if (dnxByteSize > 1000000) {
    fmt::println("\u2022 {:<20} : {:.1f}MB", "DNX-ByteSize",
                 static_cast<float>(dnxByteSize) / 1000000.0f);
  } else if (dnxByteSize > 1000) {
    fmt::println("\u2022 {:<20} : {:.1f}KB", "DNX-ByteSize",
                 static_cast<float>(dnxByteSize) / 1000.0f);
  } else {
    fmt::println("\u2022 {:<20} : {}B", "DNX-ByteSize", dnxByteSize);
  }

  fmt::println("\u2022 {:<20} : {}", "Dynamic Variables", symIR.varCount);
  fmt::println("\u2022 {:<20} : {}", "Dynamic Expressions", symCount);
  fmt::println("\u2022 {:<20} : {}", "SymIR OpCount", symIR.ops.size());
  fmt::println("\u2022 {:<20} : {}", "Amount of buffers",
               compModel.buffers.size());
  fmt::println("\u2022 {:<20} : {}", "Tensor views ", compModel.tensors.size());





  return dnx;
}
} // namespace denox::compiler

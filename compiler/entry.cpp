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
#include "db/Db.hpp"
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

  CanoModel canoModel = compiler::canonicalize(model, options);

  SymGraph &symGraph = canoModel.symGraph;

  Lifetimes lifetimes = compiler::lifeness(canoModel);

  SpecModel specModel = compiler::specialize(
      canoModel, lifetimes, memory::ActivationLayout::supported());

  OpModel opModel = compiler::dce(specModel);

  ImplModel implModel = compiler::implement(opModel, symGraph, options);

  CompModel compModel = compiler::placement(implModel);

  SymTable symTable = compiler::sym_table(model, options);

  auto [symIR, symCount] = compiler::compile_sym_and_remap(compModel, symTable);

  auto dnx = dnx::serialize(compModel, symIR, symTable, model.getInputName(),
                            model.getOutputName());

  if (options.summarize) {
    diag::print_summary(model, implModel, compModel, symIR, symCount, dnx);
  }

  return dnx;
}

void populate(const io::Path &dbpath, memory::span<const std::byte> raw,
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

  CanoModel canoModel = compiler::canonicalize(model, options);

  SymGraph &symGraph = canoModel.symGraph;

  Lifetimes lifetimes = compiler::lifeness(canoModel);

  SpecModel specModel = compiler::specialize(
      canoModel, lifetimes, memory::ActivationLayout::supported());

  OpModel opModel = compiler::dce(specModel);

  ImplDb implDb = compiler::implement_all(opModel, symGraph, options);

  auto db = compiler::Db::open(dbpath);

  std::vector<uint64_t> binaryRemap(implDb.shaderBinaries.size());
  for (size_t i = 0; i < implDb.shaderBinaries.size(); ++i) {
    uint64_t id = db.addShaderBinary(implDb.shaderBinaries[i]);
    binaryRemap[i] = id;
  }

  std::vector<uint64_t> dispatchRemap(implDb.dispatches.size());
  for (size_t i = 0; i < implDb.dispatches.size(); ++i) {
    auto& dispatch = implDb.dispatches[i];
    dispatch.binaryId = static_cast<uint32_t>(binaryRemap[dispatch.binaryId]);
    uint64_t id = db.addComputeDispatch(dispatch);
    dispatchRemap[i] = id;
  }

  for (size_t i = 0; i < implDb.ops.size(); ++i) {
    auto& op = implDb.ops[i];   
    for (size_t i = 0; i < op.dispatches.size(); ++i) {
      op.dispatches[i] = static_cast<uint32_t>(dispatchRemap[op.dispatches[i]]);
    }
    db.addOp(op);
  }


  db.close();

  // CompModel compModel = compiler::placement(implModel);
  //
  // SymTable symTable = compiler::sym_table(model, options);
  //
  // auto [symIR, symCount] = compiler::compile_sym_and_remap(compModel,
  // symTable);
  //
  // auto dnx = dnx::serialize(compModel, symIR, symTable, model.getInputName(),
  //                           model.getOutputName());
  //
  // if (options.summarize) {
  //   diag::print_summary(model, implModel, compModel, symIR, symCount, dnx);
  // }
  //
  // return dnx;
}

} // namespace denox::compiler

#include "denox/compiler/compile.hpp"
#include "denox/compiler/canonicalize/canonicalize.hpp"
#include "denox/compiler/compile_shaders/compile_shaders.hpp"
#include "denox/compiler/compile_symbols/SymProgram.hpp"
#include "denox/compiler/compile_symbols/compile_symbols.hpp"
#include "denox/compiler/dce/dce.hpp"
#include "denox/compiler/dce/prune_dead_supergraph.hpp"
#include "denox/compiler/frontend/frontend.hpp"
#include "denox/compiler/implement/implement.hpp"
#include "denox/compiler/lifeness/Lifetimes.hpp"
#include "denox/compiler/lifeness/lifeness.hpp"
#include "denox/compiler/placement/placement.hpp"
#include "denox/compiler/rebind_descriptors/rebind_descriptors.hpp"
#include "denox/compiler/selection/OptSchedule.hpp"
#include "denox/compiler/selection/selection.hpp"
#include "denox/compiler/serialize/serialize.hpp"
#include "denox/compiler/specialization/specialization.hpp"
#include "denox/glsl/GlslCompiler.hpp"
#include "denox/spirv/SpirvTools.hpp"
#include <chrono>

denox::memory::vector<std::byte>
denox::compile(memory::span<const std::byte> onnx, memory::optional<Db> odb,
               const compiler::CompileOptions &options) {

  diag::Logger logger("denox.compile", true);

  Db db = odb.value_or(Db::open({}));

  spirv::SpirvTools spirvTools(options.deviceInfo);
  spirv::GlslCompiler glslCompiler(&spirvTools, options.deviceInfo,
                                   options.spirv.debugInfo);


  compiler::Model model = compiler::frontend(onnx, options);
  compiler::CanoModel cano = compiler::canonicalize(model);
  compiler::Lifetimes lifetimes = compiler::lifeness(cano);
  compiler::SpecModel specModel = compiler::specialize(cano, lifetimes);
  compiler::ConstModel cmodel = compiler::dce(specModel);

  compiler::SuperGraph supergraph = compiler::implement(
      cmodel, cano.symGraph, &glslCompiler, options, logger);

  compiler::prune_dead_supergraph(supergraph);

  compiler::OptSchedule optSchedule = compiler::select_schedule(
      std::move(supergraph), db, model, options, logger);

  compiler::MemSchedule memSchedule = compiler::placement(optSchedule);

  compiler::SpvSchedule schedule = compiler::compile_shaders(
      std::move(memSchedule), model, db, &glslCompiler, options, logger);
  // compiler::rebind_descriptors(schedule, options, &spirvTools);
  compiler::SymProgram sprog =
      compiler::compile_symbols(schedule, model, options, logger);
  memory::vector<std::byte> dnxbuf =
      compiler::serialize(schedule, sprog, model, options);

  logger.info("[100%] Built dnx artefact");
  return dnxbuf;
}

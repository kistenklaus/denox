#include "denox/compiler/populate.hpp"
#include "denox/algorithm/align_up.hpp"
#include "denox/common/PushConstant.hpp"
#include "denox/compiler/assumed_symeval/assumed_symeval.hpp"
#include "denox/compiler/canonicalize/canonicalize.hpp"
#include "denox/compiler/dce/dce.hpp"
#include "denox/compiler/dce/prune_topological.hpp"
#include "denox/compiler/frontend/frontend.hpp"
#include "denox/compiler/implement/implement.hpp"
#include "denox/compiler/lifeness/Lifetimes.hpp"
#include "denox/compiler/lifeness/lifeness.hpp"
#include "denox/compiler/populate/populate.hpp"
#include "denox/compiler/specialization/specialization.hpp"
#include "denox/db/Db.hpp"
#include "denox/diag/progress.hpp"
#include "denox/glsl/GlslCompiler.hpp"
#include "denox/glsl/GlslCompilerInstance.hpp"
#include "denox/runtime/context.hpp"
#include "denox/spirv/SpirvTools.hpp"
#include "denox/symbolic/SymGraphEval.hpp"
#include <cmath>
#include <fmt/format.h>

void denox::populate(Db db, memory::span<const std::byte> onnx,
                     const compiler::CompileOptions &options) {

  diag::Logger logger("denox.populate", true);
  diag::Progress progress;

  spirv::SpirvTools spirvTools(options.deviceInfo);
  spirv::GlslCompiler glslCompiler(&spirvTools, options.deviceInfo,
                                   options.spirv.debugInfo);

  compiler::Model model = compiler::frontend(onnx, options);
  compiler::CanoModel cano = compiler::canonicalize(model);
  compiler::Lifetimes lifetimes = compiler::lifeness(cano);
  compiler::SpecModel specModel = compiler::specialize(cano, lifetimes);
  compiler::ConstModel cmodel = compiler::dce(specModel);
  compiler::SuperGraph supergraph =
      compiler::implement(cmodel, cano.symGraph, &glslCompiler, options, logger,
                          progress.sub_progress(0, 0.1f));

  // compiler::prune_dead_supergraph(supergraph);
  compiler::prune_topological(supergraph, cmodel, progress.sub_progress(0, 0.12f),
                              logger);

  // Evaluate symbols to their assumed values!
  SymGraphEval symeval = compiler::assumed_symeval(supergraph.symGraph,
                                                   model.valueNames(), options);

  compiler::populate(supergraph, db, symeval,
                     progress.sub_progress(0.13f, 1.0f), logger, options);
}

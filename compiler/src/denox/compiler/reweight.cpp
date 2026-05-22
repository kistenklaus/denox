#include "denox/compiler/reweight.hpp"

#include "denox/common/SHA256.hpp"
#include "denox/compiler/canonicalize/canonicalize.hpp"
#include "denox/compiler/dce/dce.hpp"
#include "denox/compiler/dce/prune_dead_supergraph.hpp"
#include "denox/compiler/dce/prune_topological.hpp"
#include "denox/compiler/frontend/frontend.hpp"
#include "denox/compiler/implement/SuperGraphEdge.hpp"
#include "denox/compiler/implement/implement.hpp"
#include "denox/compiler/lifeness/lifeness.hpp"
#include "denox/compiler/recover_options/recover_options.hpp"
#include "denox/compiler/reweight_dnx/reweight_dnx.hpp"
#include "denox/compiler/specialization/specialization.hpp"
#include "denox/glsl/GlslCompiler.hpp"
#include "denox/io/fs/FileCache.hpp"
#include "denox/spirv/SpirvTools.hpp"

#include <cstring>
#include <fmt/printf.h>
#include <stdexcept>

void denox::reweight(memory::span<std::byte> dnx,
                     memory::span<const std::byte> onnx,
                     const diag::Logger &logger) {
  diag::Progress progress{};

  compiler::CompileOptions options = compiler::recover_options(dnx);

  spirv::SpirvTools spirvTools(options.deviceInfo);
  io::FileCache fileCache;
  spirv::GlslCompiler glslCompiler(&spirvTools, &fileCache, options.deviceInfo,
                                   options.spirv.debugInfo);

  compiler::Model model = compiler::frontend(onnx, options);

  compiler::CanoModel cano = compiler::canonicalize(model);
  compiler::Lifetimes lifetimes = compiler::lifeness(cano);
  compiler::SpecModel specModel = compiler::specialize(cano, lifetimes);

  compiler::ConstModel cmodel = compiler::dce(specModel);

  compiler::SuperGraph supergraph =
      compiler::implement(cmodel, cano.symGraph, &glslCompiler, options, logger,
                          progress.sub_progress(0.0f, 0.2f));

  if (options.optimizationLevel >= 5) {
    compiler::prune_dead_supergraph(supergraph, cmodel);
  } else {
    compiler::prune_topological(supergraph, cmodel,
                                progress.sub_progress(0.21f, 0.28f), logger);
  }

  compiler::reweight_dnx(dnx, supergraph);

  progress.step(logger, 1.0f, "Reweighted dnx artefact");
}

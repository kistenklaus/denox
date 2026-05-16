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
#include "denox/compiler/select_dnx_edges/select_dnx_edges.hpp"
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

  fmt::println("{}", model.to_string());

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

  fmt::println("edge-count: {}", supergraph.graph.edgeCount());
  fmt::println("node-count: {}", supergraph.graph.nodeCount());
  fmt::println("tensor-count: {}", supergraph.tensors.size());

  // SHA256Builder hasher;
  // for (uint32_t e = 0; e < supergraph.graph.edgeCount(); ++e) {
  //   memory::EdgeId eid{e};
  //   const auto &edge = supergraph.graph.get(eid);
  //   for (const auto &dispatch : edge.dispatches) {
  //     SHA256 hash = dispatch.glsl.fast_sha256();
  //     hasher.update(memory::span{reinterpret_cast<const uint8_t *>(hash.h),
  //                                sizeof(uint32_t) * 8});
  //   }
  // }
  // SHA256 hash = hasher.finalize();
  // fmt::println("edge-hash: {}", hash);

  compiler::select_dnx_edges(dnx, supergraph);

  // memory::vector<compiler::SuperGraphEdge> redges;
  // TODO: parse edges from dnx artefact.
  // NOTE: SuperGraphEdge is not the correct datastructure here.

  // fmt::println("edge-count: {}", supergraph.graph.edgeCount());
  // TODO: keep only supergraph edges, that are in redges.
  // It's probably fine just to do a linear scan per edge,
  // kind of slow put probably fast enough.
  // Could be a bit slow with opt=5, but that is not recommended anyway.
}

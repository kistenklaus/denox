#include "denox/compiler/compile.hpp"
#include "denox/compiler/assumed_symeval/assumed_symeval.hpp"
#include "denox/compiler/canonicalize/canonicalize.hpp"
#include "denox/compiler/compile_shaders/compile_shaders.hpp"
#include "denox/compiler/compile_symbols/SymProgram.hpp"
#include "denox/compiler/compile_symbols/compile_symbols.hpp"
#include "denox/compiler/dce/dce.hpp"
#include "denox/compiler/dce/prune_dead_supergraph.hpp"
#include "denox/compiler/dce/prune_topological.hpp"
#include "denox/compiler/frontend/frontend.hpp"
#include "denox/compiler/implement/implement.hpp"
#include "denox/compiler/lifeness/Lifetimes.hpp"
#include "denox/compiler/lifeness/lifeness.hpp"
#include "denox/compiler/placement/placement.hpp"
#include "denox/compiler/populate/populate.hpp"
#include "denox/compiler/selection/OptSchedule.hpp"
#include "denox/compiler/selection/selection.hpp"
#include "denox/compiler/serialize/serialize.hpp"
#include "denox/compiler/specialization/specialization.hpp"
#include "denox/diag/logging.hpp"
#include "denox/diag/progress.hpp"
#include "denox/glsl/GlslCompiler.hpp"
#include "denox/runtime/db.hpp"
#include "denox/spirv/SpirvTools.hpp"
#include <dnx.h>

denox::memory::vector<std::byte>
denox::compile(memory::span<const std::byte> onnx, memory::optional<Db> odb,
               const memory::optional<runtime::ContextHandle> ctx,
               const compiler::CompileOptions &options,
               const diag::Logger &logger) {
  diag::Progress progress{};

  runtime::ContextHandle context;
  if (ctx) {
    context = *ctx;
  } else {
    context = runtime::Context::make(options.deviceInfo.name.c_str(),
                                     options.deviceInfo.apiVersion);
  }
  Db db = [&]() -> Db {
    if (odb.has_value()) {
      return *odb;
    } else {
      return Db::open({});
    }
  }();

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

  // {
  //   SHA256Builder hasher;
  //   for (uint32_t e = 0; e < supergraph.graph.edgeCount(); ++e) {
  //     memory::EdgeId eid{e};
  //     const auto &edge = supergraph.graph.get(eid);
  //     for (const auto &dispatch : edge.dispatches) {
  //       SHA256 hash = dispatch.glsl.fast_sha256();
  //       hasher.update(memory::span{reinterpret_cast<const uint8_t *>(hash.h),
  //                                  sizeof(uint32_t) * 8});
  //     }
  //   }
  //   SHA256 hash = hasher.finalize();
  //   fmt::println("edge-hash: {}", hash);
  // }

  SymGraphEval symeval = compiler::assumed_symeval(supergraph.symGraph,
                                                   model.valueNames(), options);

  if (options.benchOptions.minSamples > 0) {
    compiler::populate(supergraph, db, symeval,
                       progress.sub_progress(0.29f, 0.5f), logger, options);
    auto runtimeDb = runtime::Db::open(context, db);
    runtime::DbBenchOptions benchOptions;
    benchOptions.maxRelativeError = options.benchOptions.maxRelativeError;
    benchOptions.minSamples = options.benchOptions.minSamples;
    benchOptions.maxSamples = options.benchOptions.maxSamples;
    benchOptions.saveProgress = options.benchOptions.saveProgress;
    benchOptions.jobs = options.jobs;
    runtimeDb->bench(benchOptions, progress.sub_progress(0.5f, 0.95f), logger);
  }

  compiler::SuperGraph supergraphCopy = supergraph;

  compiler::OptSchedule optSchedule = compiler::select_schedule(
      std::move(supergraph), db, model, symeval, options,
      progress.sub_progress(0.95f, 0.99f), logger);

  compiler::MemSchedule memSchedule = compiler::placement(
      optSchedule, progress.sub_progress(0.95f, 0.97f), logger);

  SHA256Builder hasher;
  for (const auto &dis : memSchedule.dispatches) {
    // fmt::println("name: {}", dis.info.name);
    // fmt::println("shader-sha: {}", dis.glsl.fast_sha256());
    // fmt::println("preamble: \n{}", dis.glsl.getPreamble());
    hasher.update(
        memory::span{reinterpret_cast<uint8_t *>(dis.glsl.fast_sha256().h),
                     8 * sizeof(uint32_t)});
  }
  fmt::println("SHADER-HASH: {}", hasher.finalize());

  compiler::SpvSchedule schedule = compiler::compile_shaders(
      std::move(memSchedule), model, db, &glslCompiler, options, logger);

  compiler::SymProgram sprog = compiler::compile_symbols(
      schedule, model, options, progress.sub_progress(0.97f, 0.98f), logger);
  memory::vector<std::byte> dnxbuf =
      compiler::serialize(schedule, sprog, model, options);

  { // small sanity check (TODO remove me later)
    const dnx::Model* dnx = denox::dnx::GetModel(dnxbuf.data());
    const uint32_t dispatchCount = dnx->dispatches()->size();
    for (uint32_t d = 0; d < dispatchCount; ++d) {
      // foreach dispatch in dnx 
      //  search for edge with a dispatch, which has the name binary source hash.
      const dnx::ComputeDispatch* dnxDispatch = dnx->dispatches()->Get(d);
      const uint32_t dnxBinaryId = dnxDispatch->binary_id();
      const dnx::ShaderBinary* dnxBinary = dnx->shader_binaries()->Get(dnxBinaryId);
      SHA256 dnxSourceHash;
      std::memcpy(dnxSourceHash.h, 
          dnxBinary->source_hash()->hash()->data(), sizeof(uint32_t) * 8);
      fmt::println("DNX SourceHash: {}", dnxSourceHash);
  
      bool match = false;
  
      for (uint32_t e = 0; e < supergraphCopy.graph.edgeCount(); ++e) {
        const memory::EdgeId eid{e};
        const auto& edge = supergraphCopy.graph.get(eid);
        for (const auto& dispatch : edge.dispatches) {
          SHA256 hash = dispatch.glsl.fast_sha256();
          if (hash == dnxSourceHash) {
            match = true;
          }
        }
      }
      if (match) {
        fmt::println("dnx dispatch in supergraph");
      } else {
        fmt::println("dnx dispatch not found in supergraph");
      }
    }
  }

  progress.step(logger, 1.0f, "Build dnx artefact");
  return dnxbuf;
}

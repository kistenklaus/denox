#include "denox/compiler/populate.hpp"
#include "denox/algorithm/align_up.hpp"
#include "denox/common/PushConstant.hpp"
#include "denox/compiler/assumed_symeval/assumed_symeval.hpp"
#include "denox/compiler/canonicalize/canonicalize.hpp"
#include "denox/compiler/dce/dce.hpp"
#include "denox/compiler/frontend/frontend.hpp"
#include "denox/compiler/implement/implement.hpp"
#include "denox/compiler/lifeness/Lifetimes.hpp"
#include "denox/compiler/lifeness/lifeness.hpp"
#include "denox/compiler/specialization/specialization.hpp"
#include "denox/db/Db.hpp"
#include "denox/diag/not_implemented.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/glsl/GlslCompiler.hpp"
#include "denox/glsl/GlslCompilerInstance.hpp"
#include "denox/spirv/SpirvTools.hpp"
#include "denox/symbolic/SymGraphEval.hpp"
#include <fmt/format.h>
#include <unordered_set>

void denox::populate(Db db, memory::span<const std::byte> onnx,
                     const compiler::CompileOptions &options) {

  diag::Logger logger("denox.populate", true);

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

  // Collect all uncached glsl sources

  struct GlslCompilationUnit {
    spirv::GlslCompilerInstance glsl;
    SHA256 hash;
  };

  std::unordered_set<SHA256> sourceExists;

  std::unordered_set<std::string> tmp;
  memory::vector<GlslCompilationUnit> units;
  for (uint32_t e = 0; e < supergraph.graph.edgeCount(); ++e) {
    memory::EdgeId eid{e};
    const compiler::SuperGraphEdge &edge{supergraph.graph.get(eid)};
    for (const compiler::ComputeDispatch &dispatch : edge.dispatches) {
      SHA256 hash = dispatch.glsl.fast_sha256();
      auto cached = db.query_shader_binary(hash);
      if (!cached && !sourceExists.contains(hash) &&
          !tmp.contains(dispatch.glsl.key())) {
        units.emplace_back(dispatch.glsl, hash);
        sourceExists.insert(hash);
        tmp.insert(dispatch.glsl.key());
      }
    }
  }

  // Compile all glsl source units.
  for (size_t i = 0; i < units.size(); ++i) {

    const uint32_t percentage =
        50 +
        static_cast<uint32_t>(std::floor(static_cast<float>(i + 1) * 50.0f /
                                         static_cast<float>(units.size())));
    logger.info("[{:>3}%] {}Building SPIR-V compute shader {}{} {}", percentage,
                logger.green(),
                units[i].glsl.getSourcePath().relative_to(io::Path::cwd()),
                logger.reset(), units[i].hash);

    SpirvBinary binary = *units[i].glsl.compile();
    db.insert_binary(units[i].hash, binary);
  }

  // Evaluate symbols to their assumed values!
  SymGraphEval symeval = compiler::assumed_symeval(supergraph.symGraph,
                                                   model.valueNames(), options);

  // Register all dispatches in the database

  for (size_t e = 0; e < supergraph.graph.edgeCount(); ++e) {
    memory::EdgeId eid{e};
    const compiler::SuperGraphEdge &edge = supergraph.graph.get(eid);
    for (const compiler::ComputeDispatch &dispatch : edge.dispatches) {
      SHA256 hash = dispatch.glsl.fast_sha256();
      SpirvBinary binary = *db.query_shader_binary(hash);

      uint32_t wgX = static_cast<uint32_t>(*symeval[dispatch.workgroupCountX]);
      uint32_t wgY = static_cast<uint32_t>(*symeval[dispatch.workgroupCountY]);
      uint32_t wgZ = static_cast<uint32_t>(*symeval[dispatch.workgroupCountZ]);
      memory::small_vector<uint8_t, 4 * 4> pcbuf;
      for (const auto &pc : dispatch.pushConstants) {
        if (pc.isDynamic()) {
          switch (pc.type().kind()) {
          case memory::DtypeKind::F16:
          case memory::DtypeKind::F32:
          case memory::DtypeKind::F64:
          case memory::DtypeKind::U64:
          case memory::DtypeKind::I64:
            diag::not_implemented();
          case memory::DtypeKind::U32: {
            uint32_t x = static_cast<uint32_t>(*symeval[pc.sym()]);
            size_t offset = algorithm::align_up(pcbuf.size(), 4);
            pcbuf.resize(offset + 4, 0);
            std::memcpy(pcbuf.data() + offset, &x, 4);
            break;
          }
          case memory::DtypeKind::I32: {
            int32_t x = static_cast<int32_t>(*symeval[pc.sym()]);
            size_t offset = algorithm::align_up(pcbuf.size(), 4);
            pcbuf.resize(offset + 4, 0);
            std::memcpy(pcbuf.data() + offset, &x, 4);
            break;
          }
          default:
            diag::unreachable();
          }
        } else {
          switch (pc.type().kind()) {
          case memory::DtypeKind::F16:
          case memory::DtypeKind::F32:
          case memory::DtypeKind::F64:
          case memory::DtypeKind::U64:
          case memory::DtypeKind::I64:
            diag::not_implemented();
          case memory::DtypeKind::U32: {
            uint32_t x = pc.u32();
            size_t offset = algorithm::align_up(pcbuf.size(), 4);
            pcbuf.resize(offset + 4, 0);
            std::memcpy(pcbuf.data() + offset, &x, 4);
            break;
          }
          case memory::DtypeKind::I32: {
            int32_t x = pc.i32();
            size_t offset = algorithm::align_up(pcbuf.size(), 4);
            pcbuf.resize(offset + 4, 0);
            std::memcpy(pcbuf.data() + offset, &x, 4);
            break;
          }
          default:
            diag::unreachable();
          }
        }
      }

      memory::small_vector<DbTensorBinding, 4> bindings;
      for (const auto &binding : dispatch.bindings) {
        bindings.push_back(DbTensorBinding{
            .set = binding.set,
            .binding = binding.binding,
            .access = binding.accessFlag,
            .byteSize = static_cast<uint64_t>(
                *symeval[supergraph.tensors[binding.tensorId.index].size]),
            .alignment = supergraph.tensors[binding.tensorId.index].alignment,
        });
      }
      db.insert_dispatch(hash, pcbuf, wgX, wgY, wgZ, bindings, binary);
    }
  }

  logger.info("[100%] Database {} populated", db.path());
}

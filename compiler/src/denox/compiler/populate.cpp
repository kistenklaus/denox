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

  compiler::prune_topological(supergraph);

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
                units[i].glsl.getSourcePath().relative_to(io::Path::assets()),
                logger.reset(), units[i].hash);

    SpirvBinary binary = *units[i].glsl.compile();
    db.insert_binary(units[i].hash, binary);
  }

  // Evaluate symbols to their assumed values!
  SymGraphEval symeval = compiler::assumed_symeval(supergraph.symGraph,
                                                   model.valueNames(), options);

  // Register all dispatches in the database
  size_t new_dispatch_count = 0;
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
        const auto &tensor = supergraph.tensors[binding.tensorId.index];

        memory::optional<uint32_t> width;
        if (tensor.info.width.has_value()) {
          width = *symeval[*tensor.info.width];
        }
        memory::optional<uint32_t> height;
        if (tensor.info.height.has_value()) {
          height = *symeval[*tensor.info.height];
        }
        memory::optional<uint32_t> channels;
        if (tensor.info.channels.has_value()) {
          channels = *symeval[*tensor.info.channels];
        }
        memory::optional<TensorDataType> type;
        if (tensor.info.type != TensorDataType::Auto) {
          type = tensor.info.type;
        }
        bool isParam = false;
        for (const auto &param : edge.parameters) {
          if (param.tensorId.index == binding.tensorId.index) {
            isParam = true;
            break;
          }
        }

        bindings.push_back(DbTensorBinding{
            .set = binding.set,
            .binding = binding.binding,
            .access = binding.accessFlag,
            .byteSize = static_cast<uint64_t>(*symeval[tensor.size]),
            .alignment = tensor.alignment,
            .format = tensor.info.format,
            .storage = tensor.info.storage,
            .width = width,
            .height = height,
            .channels = channels,
            .type = type,
            .is_param = isParam});
      }

      memory::optional<memory::string> operation = dispatch.info.operation;
      memory::optional<memory::string> shader_name = dispatch.info.name;
      memory::optional<memory::string> config = dispatch.info.config;
      memory::optional<uint64_t> memory_reads;
      if (dispatch.info.memoryReads) {
        memory_reads = *symeval[*dispatch.info.memoryReads];
      }
      memory::optional<uint64_t> memory_writes;
      if (dispatch.info.memoryWrites) {
        memory_writes = *symeval[*dispatch.info.memoryWrites];
      }
      memory::optional<uint64_t> flops;
      if (dispatch.info.flops) {
        flops = *symeval[*dispatch.info.flops];
      }
      memory::optional<bool> coopmat = dispatch.info.coopmat;

      memory::optional<std::span<const uint32_t>> input_bindings;
      if (dispatch.info.input_bindings) {
        input_bindings.emplace(dispatch.info.input_bindings->begin(), dispatch.info.input_bindings->end());
      }
      memory::optional<std::span<const uint32_t>> output_bindings;
      if (dispatch.info.output_bindings) {
        output_bindings.emplace(dispatch.info.output_bindings->begin(), dispatch.info.output_bindings->end());
      }

      bool new_dispatch = db.insert_dispatch(hash, pcbuf, wgX, wgY, wgZ, bindings, binary,
                         operation, shader_name, config, memory_reads,
                         memory_writes, flops, coopmat, input_bindings, output_bindings);
      if (new_dispatch) {
        new_dispatch_count++;
      }
    }
  }
  // TODO: Remove me 
  logger.info("{} new dispatches", new_dispatch_count);

  logger.info("[100%] Database {} populated", db.path());
}

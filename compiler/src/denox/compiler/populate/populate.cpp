#include "denox/compiler/populate/populate.hpp"
#include "denox/algorithm/align_up.hpp"
#include "denox/diag/progress.hpp"

namespace denox::compiler {

void populate(const compiler::SuperGraph &supergraph, Db &db,
              const SymGraphEval &symeval, diag::Progress progressbar,
              diag::Logger &logger,
              [[maybe_unused]] const CompileOptions &options) {
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

  uint32_t jj = options.jobs;

  memory::vector<std::thread> threads(jj);
  uint32_t thread_count = static_cast<uint32_t>(threads.size());
  std::atomic<uint32_t> progess = 0;
  std::mutex mutex;
  for (uint32_t tid = 0; tid < thread_count; ++tid) {
    threads[tid] = std::thread([tid, thread_count, &units, &logger, &db, &mutex,
                                &progess, &progressbar] {
      for (size_t i = tid; i < units.size(); i += thread_count) {
        SpirvBinary binary = *units[i].glsl.compile();

        std::lock_guard lck{mutex};
        uint32_t idx = progess.fetch_add(1);
        float prog = static_cast<float>(idx + 1) /
                     (static_cast<float>(units.size() + 1));
        progressbar.step_inplace(
            logger, prog, idx == 0, "Building SPIR-V compute shader {} {}{}{}",
            units[i].glsl.getSourcePath().relative_to(io::Path::assets()),
            logger.gray(), units[i].hash, logger.reset());
        db.insert_binary(units[i].hash, binary);
      }
    });
  }
  for (uint32_t tid = 0; tid < thread_count; ++tid) {
    threads[tid].join();
  }
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
        input_bindings.emplace(dispatch.info.input_bindings->begin(),
                               dispatch.info.input_bindings->end());
      }
      memory::optional<std::span<const uint32_t>> output_bindings;
      if (dispatch.info.output_bindings) {
        output_bindings.emplace(dispatch.info.output_bindings->begin(),
                                dispatch.info.output_bindings->end());
      }

      bool new_dispatch = db.insert_dispatch(
          hash, pcbuf, wgX, wgY, wgZ, bindings, binary, operation, shader_name,
          config, memory_reads, memory_writes, flops, coopmat, input_bindings,
          output_bindings, dispatch.requirements.fixedSubgroupSize);
      if (new_dispatch) {
        new_dispatch_count++;
      }
    }
  }

  db.atomic_writeback();
  if (new_dispatch_count == 0) {
    progressbar.step(
        logger, 1.0f,
        "All required dispatch configurations already present in database");
  } else {

    progressbar.step_inplace(
        logger, 1.0f, false, "{}Database populated. Added {} new dispatches{}",
        logger.green(), new_dispatch_count, logger.reset());
  }
}

} // namespace denox::compiler

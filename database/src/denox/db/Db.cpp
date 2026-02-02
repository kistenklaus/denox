#include "denox/db/Db.hpp"
#include "denox/algorithm/hash_combine.hpp"
#include "denox/common/SHA256.hpp"
#include "denox/common/version.hpp"
#include "denox/db/DbEnv.hpp"
#include "denox/db/DbIndex.hpp"
#include "denox/db/DbMapped.hpp"
#include "denox/db/DbShaderBinary.hpp"
#include "denox/diag/invalid_argument.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/diag/logging.hpp"
#include "denox/diag/not_implemented.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/io/fs/File.hpp"
#include "flatbuffers/verifier.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <db.h>
#include <filesystem>
#include <fmt/format.h>
#include <ratio>

denox::Db denox::Db::open(const io::Path &path) {
  auto out = std::make_shared<DbMapped>();
  out->m_path = path;
  auto index = std::make_shared<DbIndex>();
  if (!path.exists()) {
    return Db{std::move(out), std::move(index)};
  }
  if (path.is_dir()) {
    DENOX_ERROR("invalid database path: {} is a directory", path);
    diag::invalid_argument();
  }
  out->m_path = path;

  auto file = io::File::open(path, io::File::OpenMode::Read);

  std::vector<std::uint8_t> buffer(file.size());
  file.read_exact(
      std::span{reinterpret_cast<std::byte *>(buffer.data()), buffer.size()});

  flatbuffers::Verifier verifier(buffer.data(), buffer.size());
  if (!denox::db::VerifyDbBuffer(verifier)) {
    DENOX_ERROR("invalid database format!");
    diag::invalid_argument();
  }

  const db::Db *db = db::GetDb(buffer.data());

  if (db->env() != nullptr) {
    const uint32_t env_count = db->env()->size();
    std::vector<DbEnv> envs;
    envs.reserve(env_count);
    for (uint32_t i = 0; i < env_count; ++i) {
      const auto *env = db->env()->Get(i);

      DbClockMode clock_mode;
      switch (env->clock_mode()) {
      case db::ClockMode_Unavailable:
        clock_mode = DbClockMode::Unavailable;
        break;
      case db::ClockMode_None:
        clock_mode = DbClockMode::None;
        break;
      case db::ClockMode_Base:
        clock_mode = DbClockMode::Base;
        break;
      case db::ClockMode_Maximum:
        clock_mode = DbClockMode::Maximum;
        break;
      }

      envs.push_back(DbEnv{
          .device = env->device()->str(),
          .os = env->device()->str(),
          .driver_version = env->driver_version()->str(),
          .denox_version = env->denox_version()->str(),
          .denox_commit_hash = env->denox_commit_hash()->str(),
          .start_timestamp = env->start_timestamp(),
          .clock_mode = clock_mode,
          .l2_warmup_iterations = env->l2_warmup_iterations(),
          .jit_warmup_iterations = env->jit_warmup_iterations(),
          .measurement_iterations = env->measurement_iterations(),
      });
    }
    out->environments = envs;
  }

  if (db->shader_binaries() != nullptr) {
    uint32_t binary_count = db->shader_binaries()->size();
    out->binaries.resize(binary_count);
    for (uint32_t i = 0; i < binary_count; ++i) {
      const db::ShaderBinary *binary = db->shader_binaries()->Get(i);
      out->binaries[i].spvBinary.spv.assign(binary->spirv()->begin(),
                                            binary->spirv()->end());
      assert(binary->src_sha256()->size() == 8);
      std::memcpy(out->binaries[i].hash.h, binary->src_sha256()->data(),

                  sizeof(uint32_t) * 8);
      // build index.
      if (index->binary_index.contains(out->binaries[i].hash)) {
        DENOX_ERROR("glsl source SHA256 collision. This should be incredibly "
                    "unlikely please contact us, by creating a github issue.");
        diag::invalid_state();
      }
      index->binary_index[out->binaries[i].hash] = i;
    }
  }

  if (db->dispatches() != nullptr) {
    uint32_t dispatchCount = db->dispatches()->size();
    out->dispatches.resize(dispatchCount);
    for (uint32_t i = 0; i < dispatchCount; ++i) {
      const db::ComputeDispatch *dispatch = db->dispatches()->Get(i);

      auto &out_dispatch = out->dispatches[i];

      out_dispatch.binaryId = dispatch->binary_id();
      out_dispatch.workgroupCountX = dispatch->workgroup_count_x();
      out_dispatch.workgroupCountY = dispatch->workgroup_count_y();
      out_dispatch.workgroupCountZ = dispatch->workgroup_count_z();
      out_dispatch.pushConstant.assign(dispatch->push_constant()->begin(),
                                       dispatch->push_constant()->end());
      out_dispatch.hash = dispatch->hash();

      uint32_t bindingCount = dispatch->bindings()->size();
      out->dispatches[i].bindings.resize(bindingCount);
      for (uint32_t b = 0; b < bindingCount; ++b) {
        const db::TensorBinding *binding = dispatch->bindings()->Get(b);
        out_dispatch.bindings[b].binding = binding->binding();
        out_dispatch.bindings[b].set = binding->set();
        switch (binding->access()) {
        case db::Access_ReadOnly:
          out_dispatch.bindings[b].access = Access::ReadOnly;
          break;
        case db::Access_WriteOnly:
          out_dispatch.bindings[b].access = Access::WriteOnly;
          break;
        case db::Access_ReadWrite:
          out_dispatch.bindings[b].access = Access::ReadWrite;
          break;
        }
        out_dispatch.bindings[b].byteSize = binding->tensor_byte_size();
        out_dispatch.bindings[b].alignment =
            static_cast<uint16_t>(binding->tensor_min_align());
        switch (binding->format()) {
        case db::TensorFormat_UNKNOWN:
          out_dispatch.bindings[b].format = TensorFormat::Optimal;
          break;
        case db::TensorFormat_SSBO_HWC:
          out_dispatch.bindings[b].format = TensorFormat::SSBO_HWC;
          break;
        case db::TensorFormat_SSBO_CHW:
          out_dispatch.bindings[b].format = TensorFormat::SSBO_CHW;
          break;
        case db::TensorFormat_SSBO_CHWC8:
          out_dispatch.bindings[b].format = TensorFormat::SSBO_CHWC8;
          break;
        case db::TensorFormat_TEX_RGBA:
          out_dispatch.bindings[b].format = TensorFormat::TEX_RGBA;
          break;
        case db::TensorFormat_TEX_RGB:
          out_dispatch.bindings[b].format = TensorFormat::TEX_RGB;
          break;
        case db::TensorFormat_TEX_RG:
          out_dispatch.bindings[b].format = TensorFormat::TEX_RG;
          break;
        case db::TensorFormat_TEX_R:
          out_dispatch.bindings[b].format = TensorFormat::TEX_R;
          break;
        }

        switch (binding->storage()) {
        case db::TensorStorage_StorageBuffer:
          out_dispatch.bindings[b].storage = TensorStorage::StorageBuffer;
          break;
        case db::TensorStorage_StorageImage:
          out_dispatch.bindings[b].storage = TensorStorage::StorageImage;
          break;
        case db::TensorStorage_SampledStorageImage:
          out_dispatch.bindings[b].storage = TensorStorage::SampledStorageImage;
          break;
        }

        if (binding->info() != nullptr) {
          const auto *info = binding->info();
          if (info->width()) {
            out_dispatch.bindings[b].width = info->width();
          }
          if (info->height()) {
            out_dispatch.bindings[b].height = info->height();
          }
          if (info->channels()) {
            out_dispatch.bindings[b].channels = info->channels();
          }
          switch (info->dtype()) {
          case db::TensorDataType_Float16:
            out_dispatch.bindings[b].type = TensorDataType::Float16;
            break;
          case db::TensorDataType_Unknown:
            break;
          }
          out_dispatch.bindings[b].is_param = info->is_param();
        }
      }

      if (dispatch->time() != nullptr) {
        out_dispatch.time.emplace();
        std::vector<DbSample> samples;
        const uint32_t sample_count = dispatch->time()->samples()->size();
        samples.reserve(sample_count);
        for (uint32_t i = 0; i < sample_count; ++i) {
          const auto &sample = dispatch->time()->samples()->Get(i);
          samples.push_back(DbSample{
              .timestamp = sample->timestamp(),
              .latency_ns = sample->latency_ns(),
              .env = sample->env(),
          });
        }
        out_dispatch.time->samples = samples;
        out_dispatch.time->mean_latency_ns =
            dispatch->time()->mean_latency_ns();
        out_dispatch.time->std_derivation_ns =
            dispatch->time()->std_derivation_ns();
      }

      if (dispatch->info() != nullptr) {
        const auto *info = dispatch->info();
        if (info->operation()) {
          out_dispatch.operation = info->operation()->str();
        }
        if (info->shader_name()) {
          out_dispatch.shader_name = info->shader_name()->str();
        }
        if (info->config()) {
          out_dispatch.config = info->config()->str();
        }
        if (info->memory_reads()) {
          out_dispatch.memory_reads = info->memory_reads();
        }
        if (info->memory_writes()) {
          out_dispatch.memory_writes = info->memory_writes();
        }
        if (info->flops()) {
          out_dispatch.flops = info->flops();
        }
        out_dispatch.coopmat = info->coopmat();
        if (info->input_bindings()) {
          out_dispatch.input_bindings.emplace(info->input_bindings()->begin(),
                                              info->input_bindings()->end());
        }
        if (info->output_bindings()) {
          out_dispatch.output_bindings.emplace(info->output_bindings()->begin(),
                                               info->output_bindings()->end());
        }
      }

      // build index.
      auto &bucket = index->dispatch_buckets[out_dispatch.hash];
      bucket.push_back(i);
    }
  }

  return Db(std::move(out), std::move(index));
}

bool denox::Db::atomic_writeback() const {
  if (m_db->m_path.empty()) {
    return false;
  }

  io::Path tmpPath = m_db->m_path.with_extension("db.tmp");

  flatbuffers::FlatBufferBuilder fbb(1 << 16);

  const size_t env_count = m_db->environments.size();
  std::vector<flatbuffers::Offset<db::BenchEnv>> envs;
  envs.reserve(env_count);
  for (size_t i = 0; i < env_count; ++i) {
    const auto &env = m_db->environments[i];

    db::ClockMode clockMode = db::ClockMode_Unavailable;
    switch (env.clock_mode) {
    case DbClockMode::Unavailable:
      break;
    case DbClockMode::None:
      clockMode = db::ClockMode_None;
      break;
    case DbClockMode::Base:
      clockMode = db::ClockMode_Base;
      break;
    case DbClockMode::Maximum:
      clockMode = db::ClockMode_Maximum;
      break;
    }

    envs.push_back(db::CreateBenchEnv(
        fbb, fbb.CreateString(env.device), fbb.CreateString(env.os),
        fbb.CreateString(env.driver_version),
        fbb.CreateString(env.denox_version),
        fbb.CreateString(env.denox_commit_hash), env.start_timestamp, clockMode,
        env.l2_warmup_iterations, env.jit_warmup_iterations,
        env.measurement_iterations));
  }
  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<db::BenchEnv>>>
      envsVec = fbb.CreateVector(envs);

  std::vector<flatbuffers::Offset<db::ShaderBinary>> binaries;
  binaries.reserve(m_db->binaries.size());
  for (const DbShaderBinary &binary : m_db->binaries) {
    binaries.push_back(db::CreateShaderBinary(
        fbb, fbb.CreateVector<uint32_t>(binary.hash.h, 8),
        fbb.CreateVector<uint32_t>(binary.spvBinary.spv)));
  }
  auto binariesVec = fbb.CreateVector(binaries);

  std::vector<flatbuffers::Offset<db::ComputeDispatch>> dispatches;
  dispatches.reserve(m_db->dispatches.size());

  for (const auto &dispatch : m_db->dispatches) {
    std::vector<flatbuffers::Offset<db::TensorBinding>> bindings;
    bindings.reserve(dispatch.bindings.size());
    for (const auto &binding : dispatch.bindings) {
      db::Access access;
      switch (binding.access) {
      case Access::ReadOnly:
        access = db::Access_ReadOnly;
        break;
      case Access::WriteOnly:
        access = db::Access_WriteOnly;
        break;
      case Access::ReadWrite:
        access = db::Access_ReadWrite;
        break;
      default:
        diag::unreachable();
      }
      db::TensorFormat format;
      switch (binding.format) {
      case TensorFormat::Optimal:
        format = db::TensorFormat_UNKNOWN;
        break;
      case TensorFormat::SSBO_HWC:
        format = db::TensorFormat_SSBO_HWC;
        break;
      case TensorFormat::SSBO_CHW:
        format = db::TensorFormat_SSBO_CHW;
        break;
      case TensorFormat::SSBO_CHWC8:
        format = db::TensorFormat_SSBO_CHWC8;
        break;
      case TensorFormat::TEX_RGBA:
        format = db::TensorFormat_TEX_RGBA;
        break;
      case TensorFormat::TEX_RGB:
        format = db::TensorFormat_TEX_RGB;
        break;
      case TensorFormat::TEX_RG:
        format = db::TensorFormat_TEX_RG;
        break;
      case TensorFormat::TEX_R:
        format = db::TensorFormat_TEX_R;
        break;
      }
      db::TensorStorage storage;
      switch (binding.storage) {
      case TensorStorage::Optimal:
        assert(false && "trying to serialize TensorStorage::Optimal");
        diag::invalid_state();
      case TensorStorage::StorageBuffer:
        storage = db::TensorStorage_StorageBuffer;
        break;
      case TensorStorage::StorageImage:
        storage = db::TensorStorage_StorageImage;
        break;
      case TensorStorage::SampledStorageImage:
        storage = db::TensorStorage_SampledStorageImage;
        break;
      }

      flatbuffers::Offset<db::TensorBindingInfo> info;
      if (binding.width.has_value() || binding.height.has_value() ||
          binding.channels.has_value() || binding.type.has_value() ||
          binding.is_param) {
        db::TensorDataType dtype = db::TensorDataType_Unknown;
        if (binding.type.has_value()) {
          switch (*binding.type) {
          case TensorDataType::Auto:
            break;
          case TensorDataType::Float16:
            dtype = db::TensorDataType_Float16;
            break;
          case TensorDataType::Float32:
            diag::not_implemented();
          case TensorDataType::Float64:
            diag::not_implemented();
          }
        }

        info = db::CreateTensorBindingInfo(
            fbb, binding.width.value_or(0), binding.height.value_or(0),
            binding.channels.value_or(0), dtype, binding.is_param);
      }

      bindings.push_back(db::CreateTensorBinding(
          fbb, binding.set, binding.binding, access, format, storage,
          binding.byteSize, binding.alignment, info));
    }
    flatbuffers::Offset<db::Timing> time = 0;
    if (dispatch.time.has_value()) {
      size_t sample_count = dispatch.time->samples.size();
      std::vector<flatbuffers::Offset<db::Sample>> samples;
      samples.reserve(sample_count);
      for (uint32_t i = 0; i < sample_count; ++i) {
        samples.push_back(db::CreateSample(fbb,
                                           dispatch.time->samples[i].timestamp,
                                           dispatch.time->samples[i].latency_ns,
                                           dispatch.time->samples[i].env));
      }

      time = db::CreateTiming(fbb, fbb.CreateVector(samples),
                              dispatch.time->mean_latency_ns,
                              dispatch.time->std_derivation_ns);
    }

    flatbuffers::Offset<db::ComputeDispatchInfo> info;
    if (dispatch.operation.has_value() || dispatch.shader_name.has_value() ||
        dispatch.config.has_value() || dispatch.memory_reads.has_value() ||
        dispatch.memory_writes.has_value() || dispatch.flops.has_value() ||
        dispatch.coopmat.has_value() || dispatch.input_bindings.has_value() ||
        dispatch.output_bindings.has_value()) {

      flatbuffers::Offset<flatbuffers::String> operation;
      flatbuffers::Offset<flatbuffers::String> shader_name;
      flatbuffers::Offset<flatbuffers::String> config;
      uint64_t memory_reads = dispatch.memory_reads.value_or(0);
      uint64_t memory_writes = dispatch.memory_writes.value_or(0);
      uint64_t flops = dispatch.flops.value_or(0);
      bool coopmat = dispatch.coopmat.value_or(false);
      flatbuffers::Offset<flatbuffers::Vector<uint32_t>> input_bindings;
      flatbuffers::Offset<flatbuffers::Vector<uint32_t>> output_bindings;

      if (dispatch.operation) {
        operation = fbb.CreateString(*dispatch.operation);
      }
      if (dispatch.shader_name) {
        shader_name = fbb.CreateString(*dispatch.shader_name);
      }
      if (dispatch.config) {
        config = fbb.CreateString(*dispatch.config);
      }
      if (dispatch.input_bindings) {
        input_bindings = fbb.CreateVector(*dispatch.input_bindings);
      }
      if (dispatch.output_bindings) {
        output_bindings = fbb.CreateVector(*dispatch.output_bindings);
      }

      info = db::CreateComputeDispatchInfo(
          fbb, operation, shader_name, config, memory_reads, memory_writes,
          flops, coopmat, input_bindings, output_bindings);
    }

    dispatches.push_back(db::CreateComputeDispatch(
        fbb, dispatch.binaryId, dispatch.workgroupCountX,
        dispatch.workgroupCountY, dispatch.workgroupCountZ,
        fbb.CreateVector<uint8_t>(dispatch.pushConstant), dispatch.hash,
        fbb.CreateVector(bindings), time, info));
  }

  auto dispatchesVec = fbb.CreateVector(dispatches);

  db::DbBuilder builder(fbb);
  builder.add_version(0);
  builder.add_shader_binaries(binariesVec);
  builder.add_dispatches(dispatchesVec);
  builder.add_env(envsVec);
  auto db = builder.Finish();
  denox::db::FinishDbBuffer(fbb, db);

  flatbuffers::DetachedBuffer detachedBuffer = fbb.Release();

  flatbuffers::Verifier verifier(detachedBuffer.data(), detachedBuffer.size());
  if (!db::VerifyDbBuffer(verifier)) {
    DENOX_ERROR("Failed to write db to \'{}\'", m_db->m_path);
    diag::invalid_state();
  }

  io::File tmpFile = io::File::open(tmpPath, io::File::OpenMode::Create |
                                                 io::File::OpenMode::Truncate |
                                                 io::File::OpenMode::Write);
  tmpFile.write_exact(
      std::span{reinterpret_cast<const std::byte *>(detachedBuffer.data()),
                detachedBuffer.size()});

  std::filesystem::rename(tmpPath.str(), m_db->m_path.str());

  return true;
}

denox::Db::Db(std::shared_ptr<DbMapped> db, std::shared_ptr<DbIndex> index)
    : m_db(std::move(db)), m_index(std::move(index)) {}

std::optional<denox::SpirvBinary>
denox::Db::query_shader_binary(const SHA256 &srcHash) const {
  auto it = m_index->binary_index.find(srcHash);
  if (it == m_index->binary_index.end()) {
    return std::nullopt;
  } else {
    uint32_t binaryId = static_cast<uint32_t>(it->second);
    return m_db->binaries[binaryId].spvBinary;
  }
}

bool denox::Db::insert_binary(const SHA256 &srcHash,
                              const SpirvBinary &binary) {
  auto cached = query_shader_binary(srcHash);
  if (cached) {
    if (!std::ranges::equal(binary.spv, cached->spv)) {
      DENOX_ERROR("Failed to insert binary into database: SHA256 collision in "
                  "database.");
      diag::invalid_state();
    }
    return false;
  }
  uint32_t binaryId = static_cast<uint32_t>(m_db->binaries.size());
  m_index->binary_index.emplace(srcHash, binaryId);
  m_db->binaries.emplace_back(DbShaderBinary{
      .hash = srcHash,
      .spvBinary = binary,
  });
  return true;
}

std::optional<std::chrono::duration<float, std::milli>>
denox::Db::query_dispatch_latency(const SHA256 &srcHash,
                                  std::span<const uint8_t> pushConstant,
                                  uint32_t workgroupCountX,
                                  uint32_t workgroupCountY,
                                  uint32_t workgroupCountZ) const {
  uint64_t hash = std::hash<SHA256>{}(srcHash);
  for (uint8_t b : pushConstant) {
    hash = algorithm::hash_combine(hash, b);
  }
  hash = algorithm::hash_combine(hash, workgroupCountX);
  hash = algorithm::hash_combine(hash, workgroupCountY);
  hash = algorithm::hash_combine(hash, workgroupCountZ);

  auto it = m_index->dispatch_buckets.find(hash);
  if (it == m_index->dispatch_buckets.end()) {
    return std::nullopt;
  }

  const auto &bucket = it->second;

  // linear search
  for (size_t dispatch_index : bucket) {
    const auto &dispatch = m_db->dispatches[dispatch_index];
    if (dispatch.workgroupCountX != workgroupCountX) {
      continue;
    }
    if (dispatch.workgroupCountY != workgroupCountY) {
      continue;
    }
    if (dispatch.workgroupCountZ != workgroupCountZ) {
      continue;
    }
    if (dispatch.pushConstant.size() != pushConstant.size()) {
      continue;
    }
    if (std::memcmp(dispatch.pushConstant.data(), pushConstant.data(),
                    dispatch.pushConstant.size()) != 0) {
      continue;
    }
    const auto &binary = m_db->binaries[dispatch.binaryId];
    if (binary.hash != srcHash) {
      continue;
    }
    if (!dispatch.time.has_value()) {
      return std::nullopt;
    }
    if (dispatch.time->samples.size() == 0) {
      return std::nullopt;
    }
    std::chrono::duration<uint64_t, std::nano> ns(
        dispatch.time->mean_latency_ns);
    return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
        ns);
  }
  return std::nullopt;
}

bool denox::Db::insert_dispatch(
    const SHA256 &srcHash, std::span<const uint8_t> pushConstant,
    uint32_t workgroupCountX, uint32_t workgroupCountY,
    uint32_t workgroupCountZ, std::span<const DbTensorBinding> bindings,
    const SpirvBinary &spvBinary, memory::optional<memory::string> operation,
    memory::optional<memory::string> shader_name,
    memory::optional<memory::string> config,
    memory::optional<uint64_t> memory_reads,
    memory::optional<uint64_t> memory_writes, memory::optional<uint64_t> flops,
    memory::optional<bool> coopmat,
    memory::optional<std::span<const uint32_t>> input_bindings,
    memory::optional<std::span<const uint32_t>> output_bindings) {
  uint32_t binaryId;
  {
    auto it = m_index->binary_index.find(srcHash);
    if (it == m_index->binary_index.end()) {
      DbShaderBinary binary;
      binary.hash = srcHash;
      binary.spvBinary = spvBinary;
      binaryId = static_cast<uint32_t>(m_db->binaries.size());
      m_db->binaries.push_back(binary);
      m_index->binary_index[srcHash] = binaryId;
    } else {
      binaryId = static_cast<uint32_t>(it->second);
      const auto &existing = m_db->binaries[binaryId].spvBinary;

      if (existing.spv.size() != spvBinary.spv.size()) {
        DENOX_ERROR("SPIR-V mismatch for identical shader hash! (different "
                    "binary-size)");
        diag::invalid_state();
      }
      if (std::memcmp(existing.spv.data(), spvBinary.spv.data(),
                      sizeof(uint32_t) * spvBinary.spv.size()) != 0) {
        DENOX_ERROR("SPIR-V mismatch for identical shader hash!");
        diag::invalid_state();
      }
    }
  }

  const auto &binary = m_db->binaries[binaryId];

  uint64_t hash = std::hash<SHA256>{}(binary.hash);
  for (uint8_t b : pushConstant) {
    hash = algorithm::hash_combine(hash, b);
  }
  hash = algorithm::hash_combine(hash, workgroupCountX);
  hash = algorithm::hash_combine(hash, workgroupCountY);
  hash = algorithm::hash_combine(hash, workgroupCountZ);

  auto it = m_index->dispatch_buckets.find(hash);
  if (it != m_index->dispatch_buckets.end()) {
    const auto &bucket = it->second;
    // linear search
    for (size_t dispatch_index : bucket) {
      const auto &dispatch = m_db->dispatches[dispatch_index];
      if (dispatch.workgroupCountX != workgroupCountX) {
        continue;
      }
      if (dispatch.workgroupCountY != workgroupCountY) {
        continue;
      }
      if (dispatch.workgroupCountZ != workgroupCountZ) {
        continue;
      }
      if (dispatch.pushConstant.size() != pushConstant.size()) {
        continue;
      }
      if (std::memcmp(dispatch.pushConstant.data(), pushConstant.data(),
                      dispatch.pushConstant.size()) != 0) {
        continue;
      }
      const auto &binary = m_db->binaries[dispatch.binaryId];
      if (binary.hash != srcHash) {
        continue;
      }
      return false;
    }
  }
  DbComputeDispatch dispatch;
  dispatch.binaryId = binaryId;
  dispatch.workgroupCountX = workgroupCountX;
  dispatch.workgroupCountY = workgroupCountY;
  dispatch.workgroupCountZ = workgroupCountZ;
  dispatch.pushConstant.assign(pushConstant.begin(), pushConstant.end());
  dispatch.hash = hash;
  dispatch.bindings.assign(bindings.begin(), bindings.end());
  dispatch.time = std::nullopt;
  dispatch.operation = operation;
  dispatch.shader_name = shader_name;
  dispatch.config = config;
  dispatch.memory_reads = memory_reads;
  dispatch.memory_writes = memory_writes;
  dispatch.flops = flops;
  dispatch.coopmat = coopmat.value_or(false);
  if (input_bindings) {
    dispatch.input_bindings.emplace(input_bindings->begin(),
                                    input_bindings->end());
  }
  if (output_bindings) {
    dispatch.output_bindings.emplace(output_bindings->begin(),
                                     output_bindings->end());
  }
  uint32_t dispatchIndex = static_cast<uint32_t>(m_db->dispatches.size());
  m_db->dispatches.push_back(std::move(dispatch));
  m_index->dispatch_buckets[hash].emplace_back(dispatchIndex);
  return true;
}

std::span<const denox::DbShaderBinary> denox::Db::binaries() const {
  return m_db->binaries;
}

std::span<const denox::DbComputeDispatch> denox::Db::dispatches() const {
  return m_db->dispatches;
}

void denox::Db::add_dispatch_benchmark_result(uint32_t dispatch_index,
                                              std::vector<DbSample> samples) {
  if (samples.empty()) {
    return;
  }

  assert(dispatch_index < m_db->dispatches.size());
  auto &dispatch = m_db->dispatches[dispatch_index];
  for (const DbSample &sample : samples) {
    if (!dispatch.time.has_value()) {
      dispatch.time.emplace();
      dispatch.time->samples.push_back(sample);
      dispatch.time->mean_latency_ns = sample.latency_ns;
      dispatch.time->std_derivation_ns = 0;
      continue;
    }
    const double prev_n = static_cast<double>(dispatch.time->samples.size());
    const double prev_mean_ns =
        static_cast<double>(dispatch.time->mean_latency_ns);
    const double prev_std_ns =
        static_cast<double>(dispatch.time->std_derivation_ns);

    const double latency_ns = static_cast<double>(sample.latency_ns);

    const double new_n = static_cast<double>(dispatch.time->samples.size() + 1);
    const double new_mean_ns = (prev_n * prev_mean_ns + 1 * latency_ns) / new_n;
    const double new_var_ns =
        std::max(0.0, (prev_n * (std::pow(prev_std_ns, 2) +
                                 std::pow(prev_mean_ns - new_mean_ns, 2)) +
                       std::pow(latency_ns - new_mean_ns, 2)) /
                          new_n);
    const double new_std_ns = std::sqrt(new_var_ns);

    dispatch.time->samples.push_back(sample);
    dispatch.time->mean_latency_ns = static_cast<uint64_t>(new_mean_ns);
    dispatch.time->std_derivation_ns = static_cast<uint64_t>(new_std_ns);
  }
}
const denox::io::Path &denox::Db::path() const { return m_db->m_path; }

uint32_t denox::Db::create_bench_environment(
    std::string device, std::string os, std::string driver_version,
    std::string denox_commit_hash, uint64_t start_timestamp,
    DbClockMode clockMode, uint16_t l2_warmup_iterations,
    uint16_t jit_warmup_iterations, uint16_t measurement_iterations) {
  uint32_t id = static_cast<uint32_t>(m_db->environments.size());
  m_db->environments.push_back(DbEnv{
      .device = device,
      .os = os,
      .driver_version = driver_version,
      .denox_commit_hash = denox_commit_hash,
      .start_timestamp = start_timestamp,
      .clock_mode = clockMode,
      .l2_warmup_iterations = l2_warmup_iterations,
      .jit_warmup_iterations = jit_warmup_iterations,
      .measurement_iterations = measurement_iterations,
  });
  return id;
}

std::span<const denox::DbEnv> denox::Db::envs() const {
  return m_db->environments;
}

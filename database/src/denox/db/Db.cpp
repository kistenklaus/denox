#include "denox/db/Db.hpp"
#include "denox/algorithm/hash_combine.hpp"
#include "denox/common/SHA256.hpp"
#include "denox/db/DbIndex.hpp"
#include "denox/db/DbMapped.hpp"
#include "denox/db/DbShaderBinary.hpp"
#include "denox/diag/invalid_argument.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/diag/logging.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/io/fs/File.hpp"
#include "flatbuffers/verifier.h"
#include <chrono>
#include <cstring>
#include <db.h>
#include <ratio>

denox::Db denox::Db::open(const io::Path &path) {
  auto out = std::make_shared<DbMapped>();
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
        out_dispatch.bindings[b].alignment = binding->tensor_min_align();
      }

      if (dispatch->time() != nullptr) {
        out_dispatch.time.emplace();
        out_dispatch.time->samples = dispatch->time()->samples();
        out_dispatch.time->latency_ns = dispatch->time()->latency_ns();
        out_dispatch.time->std_derivation_ns =
            dispatch->time()->std_derivation_ns();
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
      bindings.push_back(
          db::CreateTensorBinding(fbb, binding.set, binding.binding, access,
                                  binding.byteSize, binding.alignment));
    }
    flatbuffers::Offset<db::Timing> time = 0;
    if (dispatch.time.has_value()) {
      time = db::CreateTiming(fbb, dispatch.time->samples,
                              dispatch.time->latency_ns,
                              dispatch.time->std_derivation_ns);
    }
    dispatches.push_back(db::CreateComputeDispatch(
        fbb, dispatch.binaryId, dispatch.workgroupCountX,
        dispatch.workgroupCountY, dispatch.workgroupCountZ,
        fbb.CreateVector<uint8_t>(dispatch.pushConstant), dispatch.hash,
        fbb.CreateVector(bindings), time));
  }

  auto dispatchesVec = fbb.CreateVector(dispatches);

  db::DbBuilder builder(fbb);
  builder.add_version(0);
  builder.add_shader_binaries(binariesVec);
  builder.add_dispatches(dispatchesVec);
  builder.Finish();

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
    uint32_t binaryId = it->second;
    return m_db->binaries[binaryId].spvBinary;
  }
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
  for (uint32_t dispatch_index : bucket) {
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
    if (dispatch.time->samples == 0) {
      return std::nullopt;
    }
    std::chrono::duration<uint64_t, std::nano> ns(dispatch.time->latency_ns);
    return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
        ns);
  }
  return std::nullopt;
}

bool denox::Db::insert_dispatch(const SHA256 &srcHash,
                                std::span<const uint8_t> pushConstant,
                                uint32_t workgroupCountX,
                                uint32_t workgroupCountY,
                                uint32_t workgroupCountZ,
                                std::span<const DbTensorBinding> bindings,
                                const SpirvBinary &spvBinary) {
  uint32_t binaryId;
  {
    auto it = m_index->binary_index.find(srcHash);
    if (it == m_index->binary_index.end()) {
      DbShaderBinary binary;
      binary.hash = srcHash;
      binary.spvBinary = spvBinary;
      binaryId = m_db->binaries.size();
      m_db->binaries.push_back(binary);
      m_index->binary_index[srcHash] = binaryId;
    } else {
      binaryId = it->second;
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
    for (uint32_t dispatch_index : bucket) {
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

void denox::Db::add_dispatch_benchmark_result(
    uint32_t dispatch_index, uint32_t samples,
    std::chrono::duration<float, std::milli> latency,
    std::chrono::duration<float, std::milli> std_derivation) {
  if (samples == 0) {
    return;
  }
  assert(samples != 1 || std_derivation.count() == 0);
  assert(dispatch_index < m_db->dispatches.size());

  auto &dispatch = m_db->dispatches[dispatch_index];

  const double n2 = static_cast<double>(samples);
  const double mean2 =
      std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(
          latency)
          .count();
  const double std2 =
      std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(
          std_derivation)
          .count();

  if (!dispatch.time.has_value()) {
    dispatch.time.emplace();
    dispatch.time->samples = samples;
    dispatch.time->latency_ns = static_cast<uint64_t>(mean2);
    dispatch.time->std_derivation_ns = static_cast<uint64_t>(std2);
    return;
  }

  const double n1 = static_cast<double>(dispatch.time->samples);
  const double mean1 = static_cast<double>(dispatch.time->latency_ns);
  const double std1 = static_cast<double>(dispatch.time->std_derivation_ns);

  const double n = n1 + n2;

  const double mean = (n1 * mean1 + n2 * mean2) / n;

  const double var =
      std::max(0.0, (n1 * (std1 * std1 + (mean1 - mean) * (mean1 - mean)) +
                     n2 * (std2 * std2 + (mean2 - mean) * (mean2 - mean))) /
                        n);

  const double std = std::sqrt(var);

  dispatch.time->samples = static_cast<uint64_t>(n);
  dispatch.time->latency_ns = static_cast<uint64_t>(mean);
  dispatch.time->std_derivation_ns = static_cast<uint64_t>(std);
}

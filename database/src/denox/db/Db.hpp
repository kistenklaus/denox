#pragma once

#include "denox/db/DbComputeDispatch.hpp"
#include "denox/db/DbConvergence.hpp"
#include "denox/db/DbEnv.hpp"
#include "denox/db/DbShaderBinary.hpp"
#include "denox/db/DbTensorBinding.hpp"
#include "denox/db/sqlite/sqlite.hpp"
#include "denox/io/fs/Path.hpp"
#include "denox/memory/container/hashmap.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/span.hpp"
#include "denox/spirv/SpirvBinary.hpp"

#include <chrono>
#include <memory>
#include <mutex>

namespace denox {

class Db {
public:
  static Db open(const io::Path &path);

  void checkpoint();

  std::optional<SpirvBinary> query_shader_binary(const SHA256 &srcHash) const;

  std::optional<std::chrono::duration<float, std::milli>>
  query_dispatch_latency(const SHA256 &srcHash,
                         std::span<const uint8_t> pushConstant,
                         uint32_t workgroupCountX, uint32_t workgroupCountY,
                         uint32_t workgroupCountZ) const;

  bool insert_dispatch(
      const SHA256 &srcHash, std::span<const uint8_t> pushConstant,
      uint32_t workgroupCountX, uint32_t workgroupCountY,
      uint32_t workgroupCountZ, std::span<const DbTensorBinding> bindings,
      const SpirvBinary &binary,
      memory::optional<memory::string> operation = memory::nullopt,
      memory::optional<memory::string> shader_name = memory::nullopt,
      memory::optional<memory::string> config = memory::nullopt,
      memory::optional<uint64_t> memory_reads = memory::nullopt,
      memory::optional<uint64_t> memory_writes = memory::nullopt,
      memory::optional<uint64_t> flops = memory::nullopt,
      memory::optional<bool> coopmat = memory::nullopt,
      memory::optional<std::span<const uint32_t>> input_bindings =
          memory::nullopt,
      memory::optional<std::span<const uint32_t>> output_bindings =
          memory::nullopt,
      memory::optional<uint32_t> subgroupSize = memory::nullopt);

  bool insert_dispatch(
      const SHA256 &srcHash, std::span<const uint8_t> pushConstant,
      uint32_t workgroupCountX, uint32_t workgroupCountY,
      uint32_t workgroupCountZ, std::span<const DbTensorBinding> bindings,
      uint32_t binary_id,
      memory::optional<memory::string> operation = memory::nullopt,
      memory::optional<memory::string> shader_name = memory::nullopt,
      memory::optional<memory::string> config = memory::nullopt,
      memory::optional<uint64_t> memory_reads = memory::nullopt,
      memory::optional<uint64_t> memory_writes = memory::nullopt,
      memory::optional<uint64_t> flops = memory::nullopt,
      memory::optional<bool> coopmat = memory::nullopt,
      memory::optional<std::span<const uint32_t>> input_bindings =
          memory::nullopt,
      memory::optional<std::span<const uint32_t>> output_bindings =
          memory::nullopt,
      memory::optional<uint32_t> subgroupSize = memory::nullopt);

  bool insert_binary(const SHA256 &srcHash, const SpirvBinary &binary);

  DbShaderBinary queryShaderBinaryById(uint32_t id) const;

  DbComputeDispatch queryComputeDispatchById(uint32_t id) const;

  memory::vector<DbComputeDispatch>
  bulkQueryComputeDispatchById(memory::span<const uint32_t> dispatch_ids) const;

  DbEnv queryEnvById(uint32_t id) const;

  uint32_t queryComputeDispatchCount() const;

  void add_dispatch_benchmark_result(uint32_t dispatch_index,
                                     std::vector<DbSample> samples);

  uint32_t create_bench_environment(
      std::string device, std::string os, std::string driver_version,
      std::string denox_version, std::string denox_commit_hash,
      uint64_t start_timestamp, DbClockMode clockMode,
      uint16_t l2_warmup_iterations, uint16_t jit_warmup_iterations,
      uint16_t measurement_iterations);

  memory::vector<uint32_t> queryAllComputeDispatchIds() const;

  memory::vector<DbDispatchTimingInfo> queryAllDispatchTimingInfos() const;

  bool has_shader_binary(const SHA256 &srcHash) const;
  memory::optional<uint32_t>
  query_shader_binary_id(const SHA256 &srcHash) const;

  memory::hash_map<SHA256, uint32_t> query_in_memory_shader_cache() const;

  DbConvergenceInfo query_convergence_info(uint64_t minSamples,
                                           double maxRelativeError) const;

private:
  void finalize_stmts();
  void create_cached_stmts();

  struct Inner {
    sqlite::Db db;
    std::mutex mutex;

    sqlite::Stmt query_binary_by_hash;
    sqlite::Stmt query_dispatch_latency;

    sqlite::Stmt query_binary_existence;
    sqlite::Stmt insert_binary;

    sqlite::Stmt insert_dispatch_query_dispatch_existance;
    sqlite::Stmt insert_dispatch_insert_dispatch;
    sqlite::Stmt insert_dispatch_insert_bindings;
  };
  explicit Db(std::shared_ptr<Inner> inner) : m_inner(std::move(inner)) {}

  std::shared_ptr<Inner> m_inner;
};

} // namespace denox

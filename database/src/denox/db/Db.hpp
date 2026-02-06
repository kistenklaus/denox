#pragma once

#include "denox/db/DbComputeDispatch.hpp"
#include "denox/db/DbEnv.hpp"
#include "denox/db/DbShaderBinary.hpp"
#include "denox/db/DbTensorBinding.hpp"
#include "denox/io/fs/Path.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/spirv/SpirvBinary.hpp"
#include <chrono>
#include <memory>

namespace denox {

class Db {
public:
  static Db open(const io::Path &path);

  bool atomic_writeback() const;

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
          memory::nullopt);

  bool insert_binary(const SHA256 &srcHash, const SpirvBinary &binary);

  // insert_dispatch, invalidates the span!
  std::span<const DbShaderBinary> binaries() const;

  // insert_dispatch, invalidates the span!
  std::span<const DbComputeDispatch> dispatches() const;

  std::span<const DbEnv> envs() const;

  // Accumulates benchmark results into existing timing statistics.
  // Timing is stored as population mean and standard deviation.
  void add_dispatch_benchmark_result(uint32_t dispatch_index,
                                     std::vector<DbSample> samples);

  uint32_t create_bench_environment(
      std::string device, std::string os, std::string driver_version,
      std::string denox_version, std::string denox_commit_hash,
      uint64_t start_timestamp, DbClockMode clockMode,
      uint16_t l2_warmup_iterations, uint16_t jit_warmup_iterations,
      uint16_t measurement_iterations);

  const io::Path &path() const;

private:
  Db(std::shared_ptr<struct DbMapped> db,
     std::shared_ptr<struct DbIndex> index);

  std::shared_ptr<struct DbMapped> m_db;
  std::shared_ptr<struct DbIndex> m_index;
};

} // namespace denox

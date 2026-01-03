#pragma once

#include "denox/db/DbComputeDispatch.hpp"
#include "denox/db/DbShaderBinary.hpp"
#include "denox/db/DbTensorBinding.hpp"
#include "denox/io/fs/Path.hpp"
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

  bool insert_dispatch(const SHA256 &srcHash,
                       std::span<const uint8_t> pushConstant,
                       uint32_t workgroupCountX, uint32_t workgroupCountY,
                       uint32_t workgroupCountZ,
                       std::span<const DbTensorBinding> bindings,
                       const SpirvBinary &binary);

  bool insert_binary(const SHA256 &srcHash, const SpirvBinary &binary);

  // insert_dispatch, invalidates the span!
  std::span<const DbShaderBinary> binaries() const;

  // insert_dispatch, invalidates the span!
  std::span<const DbComputeDispatch> dispatches() const;

  // Accumulates benchmark results into existing timing statistics.
  // Timing is stored as population mean and standard deviation.
  void add_dispatch_benchmark_result(
      uint32_t dispatch_index, uint32_t samples,
      std::chrono::duration<float, std::milli> latency,
      std::chrono::duration<float, std::milli> std_derivation);

private:
  Db(std::shared_ptr<struct DbMapped> db,
     std::shared_ptr<struct DbIndex> index);

  std::shared_ptr<struct DbMapped> m_db;
  std::shared_ptr<struct DbIndex> m_index;
};

} // namespace denox

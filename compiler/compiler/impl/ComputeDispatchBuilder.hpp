#pragma once

#include "compiler/ir/impl/ComputeDispatch.hpp"
#include "compiler/ir/impl/PushConstant.hpp"
#include "compiler/ir/impl/TensorBinding.hpp"
#include "compiler/ir/impl/TensorId.hpp"
#include "io/fs/Path.hpp"
#include "memory/container/string.hpp"
#include "memory/hypergraph/NodeId.hpp"
#include <memory>

namespace denox::compiler {
class Impl;

class ComputeDispatchBuilder {
public:
  friend class Impl;

  void addBinding(std::uint32_t set, std::uint32_t binding, AccessFlag access,
                  TensorId tensor) {
    self().bindings.push_back(TensorBinding{set, binding, access, tensor});
  }

  void addBinding(std::uint32_t set, std::uint32_t binding, AccessFlag access,
                  memory::NodeId nodeId);

  void addPushConstant(PushConstant pushConstant) {
    self().pushConstants.push_back(pushConstant);
  }

  void setName(memory::string name);

  void setDebugInfo(memory::string debugInfo) {
    if (self().meta == nullptr) {
      self().meta = std::make_unique<ComputeDispatchMeta>();
    }
    self().meta->debug_info = std::move(debugInfo);
  }

  void setSourcePath(io::Path sourcePath) {
    if (self().meta == nullptr) {
      self().meta = std::make_unique<ComputeDispatchMeta>();
    }
    self().meta->sourcePath = std::move(sourcePath);
  }

  void setMemoryReads(Sym reads) {
    if (self().meta == nullptr) {
      self().meta = std::make_unique<ComputeDispatchMeta>();
    }
    self().meta->memory_reads = reads;
  }

  void setMemoryWrites(Sym reads) {
    if (self().meta == nullptr) {
      self().meta = std::make_unique<ComputeDispatchMeta>();
    }
    self().meta->memory_writes = reads;
  }

private:
  ComputeDispatch &self();

  ComputeDispatchBuilder(std::size_t index, Impl *impl)
      : m_index(index), m_impl(impl) {}
  std::size_t m_index;
  Impl *m_impl;
};

} // namespace denox::compiler

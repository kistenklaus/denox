#pragma once

#include "denox/common/Access.hpp"
#include "denox/compiler/implement/ComputeDispatch.hpp"
#include "denox/memory/hypergraph/NodeId.hpp"
#include <cstddef>
namespace denox::compiler {

class OpImpl;

class ComputeDispatchBuilder {
public:
  friend class OpImpl;

  void addBinding(uint32_t set, uint32_t binding, Access access,
                  TensorId tensor);

  void addBinding(uint32_t set, uint32_t binding, Access access,
                  memory::NodeId nodeId);
  void addPushConstant(PushConstant pc) { self().pushConstants.push_back(pc); }
  void setName(memory::string_view name) { self().info.name = name; };
  void setDebugInfo(memory::string_view debugInfo);
  void setSourcePath(const io::Path &path) { self().info.srcPath = path; }
  void setMemoryReads(Sym reads) { self().info.memoryReads = reads; }
  void setMemoryWrites(Sym writes) { self().info.memoryWrites = writes; }

private:
  ComputeDispatchBuilder(size_t index, OpImpl *impl)
      : m_index(index), m_impl(impl) {}
  ComputeDispatch &self();

private:
  size_t m_index;
  OpImpl *m_impl;
};

} // namespace denox::compiler

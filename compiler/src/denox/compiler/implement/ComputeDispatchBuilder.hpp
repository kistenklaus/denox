#pragma once

#include "denox/common/Access.hpp"
#include "denox/compiler/implement/ComputeDispatch.hpp"
#include "denox/compiler/implement/TensorId.hpp"
#include "denox/memory/container/small_vector.hpp"
#include "denox/memory/hypergraph/NodeId.hpp"
#include <cstddef>
#include <vulkan/vulkan.hpp>
namespace denox::compiler {

class OpImpl;

class ComputeDispatchBuilder {
public:
  friend class OpImpl;

  void addBinding(memory::string_view set_macro,
                  memory::string_view binding_macro, Access access,
                  memory::NodeId nodeId);

  void addParamBinding(memory::string_view set_macro,
                       memory::string_view binding_macro, TensorId tensor);

  void addPushConstant(PushConstant pc) { self().pushConstants.push_back(pc); }

  void setName(memory::string_view name) { self().info.name = name; };
  void setConfig(memory::string config) { self().info.config = config; }
  void setOperation(memory::string op) { self().info.operation = op; }
  void usesCoopmat(bool b) { self().info.coopmat = b; }

  void setSourcePath(const io::Path &path) { self().info.srcPath = path; }
  void setMemoryReads(Sym reads) { self().info.memoryReads = reads; }
  void setMemoryWrites(Sym writes) { self().info.memoryWrites = writes; }

  void setFlops(Sym flops) { self().info.flops = flops; }

  ~ComputeDispatchBuilder();

private:
  ComputeDispatchBuilder(size_t index, OpImpl *impl)
      : m_index(index), m_impl(impl) {}
  ComputeDispatchBuilder(const ComputeDispatchBuilder &) = delete;
  ComputeDispatchBuilder &operator=(const ComputeDispatchBuilder &) = delete;
  ComputeDispatchBuilder(ComputeDispatchBuilder &&) = delete;
  ComputeDispatchBuilder &operator=(ComputeDispatchBuilder &&) = delete;

  ComputeDispatch &self();

private:
  enum class BindingType {
    Input,
    Output,
    Param,
    Read,
    Write,
  };

  struct TensorBindingInfo {
    Access access;
    BindingType type;
    int32_t edgeSrc; // -1 means not a source
    bool edgeDst;     // -1 means not a dst
    memory::string set_macro;
    memory::string binding_macro;
    TensorId tensor;
  };

  size_t m_index;
  memory::small_vector<TensorBindingInfo, 4> m_bindingInfos;
  OpImpl *m_impl;
};

} // namespace denox::compiler

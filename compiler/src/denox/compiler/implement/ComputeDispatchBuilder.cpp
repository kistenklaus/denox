#include "denox/compiler/implement/ComputeDispatchBuilder.hpp"
#include "denox/compiler/implement/OpImpl.hpp"
#include "denox/compiler/implement/SuperGraphBuilder.hpp"
#include "denox/diag/invalid_state.hpp"
#include <stdexcept>

namespace denox::compiler {

ComputeDispatch &ComputeDispatchBuilder::self() {
  return m_impl->m_dispatches[m_index];
}

void ComputeDispatchBuilder::addBinding(uint32_t set, uint32_t binding,
                                        Access access, memory::NodeId nodeId) {
  if (*nodeId >= m_impl->m_superBuilder->m_nodeTensorMapping.size()) {
    diag::invalid_state();
  }
  memory::optional<TensorId> tensorId =
      m_impl->m_superBuilder->m_nodeTensorMapping[*nodeId];
  if (!tensorId.has_value()) {
    diag::invalid_state();
  }
  addBinding(set, binding, access, *tensorId);
}

void ComputeDispatchBuilder::addBinding(uint32_t set, uint32_t binding,
                                        Access access, TensorId tensor) {
  self().bindings.push_back(TensorBinding{set, binding, access, tensor});
}
void ComputeDispatchBuilder::setDebugInfo(memory::string_view debugInfo) {
  self().info.debug_info = debugInfo;
}
} // namespace denox::compiler

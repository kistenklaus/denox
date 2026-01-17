#include "denox/compiler/implement/ComputeDispatchBuilder.hpp"
#include "denox/compiler/implement/OpImpl.hpp"
#include "denox/compiler/implement/SuperGraphBuilder.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/memory/container/small_vector.hpp"
#include <algorithm>
#include <iostream>
#include <type_traits>

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

void ComputeDispatchBuilder::addBinding(memory::string_view set_macro,
                                        memory::string_view binding_macro,
                                        Access access, memory::NodeId nodeId) {
  if (*nodeId >= m_impl->m_superBuilder->m_nodeTensorMapping.size()) {
    diag::invalid_state();
  }
  memory::optional<TensorId> tensorId =
      m_impl->m_superBuilder->m_nodeTensorMapping[*nodeId];
  if (!tensorId.has_value()) {
    diag::invalid_state();
  }
  addBinding(set_macro, binding_macro, access, *tensorId);
}

void ComputeDispatchBuilder::addBinding(memory::string_view set_macro,
                                        memory::string_view binding_macro,
                                        Access access, TensorId tensor) {
  const bool isInput =
      access == Access::ReadOnly &&
      std::ranges::find_if(m_impl->m_superBuilder->m_inputs,
                           [&](const memory::NodeId &nid) -> bool {
                             TensorId inputId =
                                 m_impl->m_superBuilder->m_graph.get(nid);
                             return inputId.index == tensor.index;
                           }) != m_impl->m_superBuilder->m_inputs.end();
  const bool isOutput =
      access == Access::WriteOnly &&
      std::ranges::find_if(m_impl->m_superBuilder->m_outputs,
                           [&](const memory::NodeId &nid) -> bool {
                             TensorId outputId =
                                 m_impl->m_superBuilder->m_graph.get(nid);
                             return outputId.index == tensor.index;
                           }) != m_impl->m_superBuilder->m_outputs.end();
  const bool isParam =
      access == Access::ReadOnly &&
      std::ranges::find_if(m_impl->m_parameters,
                           [&](const Parameter &param) -> bool {
                             return param.tensorId.index == tensor.index;
                           });
  const bool isRead = access == Access::ReadOnly;
  const bool isWrite = access == Access::WriteOnly;

  BindingType type;
  if (isInput) {
    type = BindingType::Input;
  } else if (isOutput) {
    type = BindingType::Output;
  } else if (isParam) {
    type = BindingType::Param;
  } else if (isRead) {
    type = BindingType::Read;
  } else if (isWrite) {
    type = BindingType::Write;
  } else {
    std::cerr << fmt::format("Binding doesn't match any descriptor policy.")
              << std::endl;
    diag::invalid_state();
  }
  m_bindingInfos.push_back(TensorBindingInfo{
      .access = access,
      .type = type,
      .set_macro = std::string(set_macro),
      .binding_macro = std::string(binding_macro),
      .tensor = tensor,
  });
}

void ComputeDispatchBuilder::setDebugInfo(memory::string_view debugInfo) {
  self().info.debug_info = debugInfo;
}

ComputeDispatchBuilder::~ComputeDispatchBuilder() {
  memory::small_vector<uint32_t, 4> sets;
  sets.resize(m_bindingInfos.size());
  const auto &policy = m_impl->m_superBuilder->m_descriptorPolicies;
  for (size_t i = 0; i < m_bindingInfos.size(); ++i) {
    const auto &bindingInfo = m_bindingInfos[i];
    switch (bindingInfo.type) {
    case BindingType::Input:
      sets[i] = policy.inputPolicy.set;
      break;
    case BindingType::Output:
      sets[i] = policy.outputPolicy.set;
      break;
    case BindingType::Param:
      sets[i] = policy.paramPolicy.set;
      break;
    case BindingType::Read:
      sets[i] = policy.readPolicy.set;
      break;
    case BindingType::Write:
      sets[i] = policy.writePolicy.set;
      break;
    }
  }

  uint32_t maxSet = 0;
  for (uint32_t set : sets) {
    maxSet = std::max(maxSet, set);
  }
  uint32_t setCount = maxSet + 1;

  std::ranges::sort(
      m_bindingInfos,
      [](const TensorBindingInfo &lhs, const TensorBindingInfo &rhs) -> bool {
        using base_t = std::underlying_type_t<BindingType>;
        return static_cast<base_t>(lhs.type) < static_cast<base_t>(rhs.type);
      });

  memory::small_vector<uint32_t, 4> binding_acc(setCount, 0);

  for (size_t i = 0; i < m_bindingInfos.size(); ++i) {
    const auto &info = m_bindingInfos[i];
    uint32_t set = sets[i];
    uint32_t binding = binding_acc[set]++;
    self().bindings.push_back(TensorBinding{
        .set = set,
        .binding = binding,
        .accessFlag = info.access,
        .tensorId = info.tensor,
    });
    self().glsl.define(info.set_macro, set);
    self().glsl.define(info.binding_macro, binding);
  }
}

} // namespace denox::compiler

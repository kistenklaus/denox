#include "denox/compiler/implement/ComputeDispatchBuilder.hpp"
#include "denox/compiler/implement/OpImpl.hpp"
#include "denox/compiler/implement/SuperGraphBuilder.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/diag/logging.hpp"
#include "denox/memory/container/small_vector.hpp"
#include <algorithm>
#include <iostream>
#include <memory>
#include <type_traits>

namespace denox::compiler {

ComputeDispatch &ComputeDispatchBuilder::self() {
  return m_impl->m_dispatches[m_index];
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
  const bool isInput =
      access == Access::ReadOnly &&
      std::ranges::find_if(m_impl->m_superBuilder->m_inputs,
                           [&](const memory::NodeId &nid) -> bool {
                             TensorId inputId =
                                 m_impl->m_superBuilder->m_graph.get(nid);
                             return inputId.index == tensorId->index;
                           }) != m_impl->m_superBuilder->m_inputs.end();
  const bool isOutput =
      access == Access::WriteOnly &&
      std::ranges::find_if(m_impl->m_superBuilder->m_outputs,
                           [&](const memory::NodeId &nid) -> bool {
                             TensorId outputId =
                                 m_impl->m_superBuilder->m_graph.get(nid);
                             return outputId.index == tensorId->index;
                           }) != m_impl->m_superBuilder->m_outputs.end();
  const bool isRead = access == Access::ReadOnly;
  const bool isWrite = access == Access::WriteOnly;
  BindingType type;
  if (isInput) {
    type = BindingType::Input;
  } else if (isOutput) {
    type = BindingType::Output;
  } else if (isRead) {
    type = BindingType::Read;
  } else if (isWrite) {
    type = BindingType::Write;
  } else {
    std::cerr << fmt::format("Binding doesn't match any descriptor policy.")
              << std::endl;
    diag::invalid_state();
  }

  int32_t edgeSrc = -1;
  for (uint32_t i = 0; i < m_impl->m_inputs.size(); ++i) {
    const auto nid = m_impl->m_inputs[i];
    if (nid == nodeId) {
      edgeSrc = static_cast<int32_t>(i);
      break;
    }
  }
  bool edgeDst = m_impl->m_output == nodeId;


  m_bindingInfos.push_back(TensorBindingInfo{
      .access = access,
      .type = type,
      .edgeSrc = edgeSrc,
      .edgeDst = edgeDst,
      .set_macro = std::string(set_macro),
      .binding_macro = std::string(binding_macro),
      .tensor = *tensorId,
  });
}

void ComputeDispatchBuilder::addParamBinding(memory::string_view set_macro,
                                             memory::string_view binding_macro,
                                             TensorId tensor) {

  BindingType type = BindingType::Param;
  m_bindingInfos.push_back(TensorBindingInfo{
      .access = Access::ReadOnly,
      .type = type,
      .edgeSrc = -1,
      .edgeDst = false,
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
  uint32_t src_count = 0;
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
    src_count = static_cast<uint32_t>(
        std::max(static_cast<int32_t>(src_count), bindingInfo.edgeSrc));
  }
  src_count += 1;

  uint32_t maxSet = 0;
  for (uint32_t set : sets) {
    maxSet = std::max(maxSet, set);
  }
  uint32_t setCount = maxSet + 1;

  std::ranges::stable_sort(
      m_bindingInfos,
      [](const TensorBindingInfo &lhs, const TensorBindingInfo &rhs) -> bool {
        using base_t = std::underlying_type_t<BindingType>;
        return static_cast<base_t>(lhs.type) < static_cast<base_t>(rhs.type);
      });

  memory::small_vector<uint32_t, 4> binding_acc(setCount, 0);

  if (src_count > 0) {
    self().info.input_bindings.emplace();
    self().info.input_bindings->resize(src_count);
  }
  for (size_t i = 0; i < m_bindingInfos.size(); ++i) {
    const auto &info = m_bindingInfos[i];
    uint32_t set = sets[i];
    uint32_t binding = binding_acc[set]++;
    uint32_t id = static_cast<uint32_t>(self().bindings.size());
    self().bindings.push_back(TensorBinding{
        .set = set,
        .binding = binding,
        .accessFlag = info.access,
        .tensorId = info.tensor,
    });
    self().glsl.define(info.set_macro, set);
    self().glsl.define(info.binding_macro, binding);

    if (info.edgeDst) {
      if (!self().info.output_bindings) {
        self().info.output_bindings.emplace();
      } else {
        assert("true outputs seem wrong");
        diag::invalid_state(); // 2 outputs seem wrong!
      }
      self().info.output_bindings->push_back(id);
    }
    if (info.edgeSrc >= 0) {
      (*self().info.input_bindings)[static_cast<uint32_t>(info.edgeSrc)] = id;
    }
  }

}

} // namespace denox::compiler

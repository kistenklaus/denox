#include "compiler/impl/ComputeDispatchBuilder.hpp"
#include "compiler/impl/ImplBuilder.hpp"
#include "diag/unreachable.hpp"
#include <fmt/format.h>

denox::compiler::ComputeDispatch &
denox::compiler::ComputeDispatchBuilder::self() {
  return m_impl->m_impl->dispatches[m_index];
};

void denox::compiler::ComputeDispatchBuilder::addBinding(
    std::uint32_t set, std::uint32_t binding, AccessFlag access,
    memory::NodeId nodeId) {

  assert(static_cast<std::uint64_t>(nodeId) < m_impl->m_nodeTensorMapping.size());
  assert(m_impl->m_nodeTensorMapping[*nodeId].has_value());
  TensorId id = m_impl->m_nodeTensorMapping[*nodeId].value();
  return addBinding(set, binding, access, id);
}

void denox::compiler::ComputeDispatchBuilder::setName(memory::string name) {
  if (self().meta == nullptr) {
    self().meta = std::make_unique<ComputeDispatchMeta>();
  }

  memory::span<ComputeDispatch> dispatches = m_impl->m_impl->dispatches;
  bool duplicatesExist = false;
  memory::optional<std::size_t> original = memory::nullopt;
  memory::string zeroName = fmt::format("{}-0", name);
  for (std::size_t i = 0; i < dispatches.size(); ++i) {
    const auto &dispatch = dispatches[i];
    if (dispatch.meta == nullptr) {
      continue;
    }
    if (!dispatch.meta->name.has_value()) {
      continue;
    }
    if (*dispatch.meta->name == name) {
      original = i;
      break;
    }
    if (*dispatch.meta->name == zeroName) {
      duplicatesExist = true;
      break;
    }
  }

  if (!duplicatesExist && !original.has_value()) {
    self().meta->name = std::move(name);
    return;
  }

  if (original.has_value()) {
    assert(dispatches[*original].meta != nullptr);
    assert(dispatches[*original].meta->name.has_value());
    dispatches[*original].meta->name =
        fmt::format("{}-0", *dispatches[*original].meta->name);
    self().meta->name = fmt::format("{}-1", name);
    return;
  }

  assert(duplicatesExist);
  std::size_t suffix = 2;
  while (true) {
    memory::string newName = fmt::format("{}-{}", name, suffix);
    bool exists = false;
    for (std::size_t i = 0; i < dispatches.size(); ++i) {
      const auto &dispatch = dispatches[i];
      if (dispatch.meta == nullptr) {
        continue;
      }
      if (!dispatch.meta->name.has_value()) {
        continue;
      }
      if (dispatch.meta->name.value() == newName) {
        exists = true;
        break;
      }
    }
    if (!exists) {
      self().meta->name = std::move(newName);
      return;
    }
    ++suffix;
  }
  compiler::diag::unreachable();
}

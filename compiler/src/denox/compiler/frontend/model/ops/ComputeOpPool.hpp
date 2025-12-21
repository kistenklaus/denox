#pragma once

#include "denox/common/PoolFunction.hpp"
#include "denox/memory/container/uvec2.hpp"
#include <fmt/core.h>
#include <memory>
#include <utility>

namespace denox::compiler {

struct ComputeOpPool {
  struct Storage {
    memory::uvec2 kernelSize;
    memory::uvec2 padding;
    memory::uvec2 stride;
    PoolFunction func;
  };

  const Storage &operator*() const { return *m_store; }
  Storage &operator*() { return *m_store; }
  const Storage *operator->() const { return m_store.get(); }
  Storage *operator->() { return m_store.get(); }

  explicit ComputeOpPool(memory::uvec2 kernelSize, memory::uvec2 padding,
                         memory::uvec2 stride, PoolFunction func)
      : m_store(std::make_unique<Storage>(kernelSize, padding, stride, func)) {}

  ComputeOpPool(const ComputeOpPool &o)
      : m_store(std::make_unique<Storage>(*o.m_store)) {}

  ComputeOpPool &operator=(const ComputeOpPool &o) {
    if (this == &o) {
      return *this;
    }
    m_store = std::make_unique<Storage>(*o.m_store);
    return *this;
  }

  ComputeOpPool(ComputeOpPool &&o) : m_store(std::move(o.m_store)) {}

  ComputeOpPool &operator=(ComputeOpPool &&o) {
    if (this == &o) {
      return *this;
    }
    m_store = std::move(o.m_store);
    return *this;
  }

private:
  std::unique_ptr<Storage> m_store;
};

} // namespace denox::compiler

template <> struct fmt::formatter<denox::compiler::ComputeOpPool> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::ComputeOpPool &op,
              FormatContext &ctx) const {
    const auto &s = *op;
    return fmt::format_to(ctx.out(),
                          "{{kernel={}, padding={}, stride={}, func={}}}",
                          s.kernelSize, s.padding, s.stride, s.func);
  }
};

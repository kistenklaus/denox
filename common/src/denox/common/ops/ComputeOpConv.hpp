#pragma once

#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/shared_ptr.hpp"
#include "denox/memory/container/uvec2.hpp"
#include "denox/memory/dtype/dtype.hpp"
#include "denox/memory/tensor/BiasTensor.hpp"
#include "denox/memory/tensor/FilterTensor.hpp"
#include <memory>

namespace denox {

struct ComputeOpConv {
  struct Storage {
    memory::shared_ptr<memory::FilterTensor> W;
    memory::shared_ptr<memory::BiasTensor> B; // <- may be nullptr
    memory::uvec2 padding;
    memory::uvec2 stride;

    memory::optional<memory::Dtype> atype;
  };

  const Storage &operator*() const { return *m_store; }
  Storage &operator*() { return *m_store; }
  const Storage *operator->() const { return m_store.get(); }
  Storage *operator->() { return m_store.get(); }

  ComputeOpConv(memory::FilterTensor W, memory::optional<memory::BiasTensor> B,
                memory::uvec2 padding, memory::uvec2 stride,
                memory::optional<memory::Dtype> atype)
      : m_store(std::make_unique<Storage>(
            std::make_shared<memory::FilterTensor>(std::move(W)),
            B.has_value() ? std::make_shared<memory::BiasTensor>(std::move(*B))
                          : nullptr,
            padding, stride, atype)) {}

  ComputeOpConv(const ComputeOpConv &o)
      : m_store(std::make_unique<Storage>(*o.m_store)) {}

  ComputeOpConv &operator=(const ComputeOpConv &o) {
    if (this == &o) {
      return *this;
    }
    m_store = std::make_unique<Storage>(*o.m_store);
    return *this;
  }

  ComputeOpConv(ComputeOpConv &&o) : m_store(std::move(o.m_store)) {}

  ComputeOpConv &operator=(ComputeOpConv &&o) {
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

template <> struct fmt::formatter<denox::ComputeOpConv> {
  constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::ComputeOpConv &op,
              FormatContext &ctx) const {
    const auto &s = *op;

    // Bias handling
    if (s.B) {
      return fmt::format_to(ctx.out(),
                            "{{W={}, B={}, padding={}, stride={}, atype={}}}",
                            *s.W, *s.B, s.padding, s.stride, s.atype);
    } else {
      return fmt::format_to(ctx.out(),
                            "{{W={}, B=none, padding={}, stride={}, atype={}}}",
                            *s.W, s.padding, s.stride, s.atype);
    }
  }
};

#pragma once

#include "denox/common/PaddingMode.hpp"
#include "denox/symbolic/Sym.hpp"
#include <memory>

namespace denox::compiler {

struct ComputeOpPad {
  struct Storage {
    Sym left;
    Sym right;
    Sym top;
    Sym bottom;
    PaddingMode mode;
  };

  const Storage &operator*() const { return *m_store; }
  Storage &operator*() { return *m_store; }
  const Storage *operator->() const { return m_store.get(); }
  Storage *operator->() { return m_store.get(); }

  ComputeOpPad(Sym left, Sym right, Sym top, Sym bottom, PaddingMode mode)
      : m_store(std::make_unique<Storage>(left, right, top, bottom, mode)) {}

  ComputeOpPad(const ComputeOpPad &o)
      : m_store(std::make_unique<Storage>(*o.m_store)) {}

  ComputeOpPad &operator=(const ComputeOpPad &o) {
    if (this == &o) {
      return *this;
    }
    m_store = std::make_unique<Storage>(*o.m_store);
    return *this;
  }

  ComputeOpPad(ComputeOpPad &&o) : m_store(std::move(o.m_store)) {}

  ComputeOpPad &operator=(ComputeOpPad &&o) {
    if (this == &o) {
      return *this;
    }
    m_store = std::move(o.m_store);
    return *this;
  }

private:
  std::unique_ptr<Storage> m_store;
};

} // namespace vkcnn

template <>
struct fmt::formatter<denox::compiler::ComputeOpPad> {
  constexpr auto parse(fmt::format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const denox::compiler::ComputeOpPad& op,
              FormatContext& ctx) const {
    const auto& s = *op;
    return fmt::format_to(
        ctx.out(),
        "{{left={}, right={}, top={}, bottom={}, mode={}}}",
        s.left,
        s.right,
        s.top,
        s.bottom,
        s.mode);
  }
};

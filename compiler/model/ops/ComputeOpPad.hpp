#pragma once

#include "model/PaddingMode.hpp"
#include "symbolic/Sym.hpp"
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

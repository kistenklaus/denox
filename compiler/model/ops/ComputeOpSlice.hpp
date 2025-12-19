#pragma once

#include "denox/symbolic/Sym.hpp"
#include <memory>
namespace denox::compiler {

struct ComputeOpSlice {
  struct Storage {
    Sym left;
    Sym right;
    Sym top;
    Sym bottom;
  };

  const Storage &operator*() const { return *m_store; }
  Storage &operator*() { return *m_store; }
  const Storage *operator->() const { return m_store.get(); }
  Storage *operator->() { return m_store.get(); }

  ComputeOpSlice(Sym left, Sym right, Sym top, Sym bottom)
      : m_store(std::make_unique<Storage>(left, right, top, bottom)) {}

  ComputeOpSlice(const ComputeOpSlice &o)
      : m_store(std::make_unique<Storage>(*o.m_store)) {}
  ComputeOpSlice &operator=(const ComputeOpSlice &o) {
    if (this == &o) {
      return *this;
    }
    m_store = std::make_unique<Storage>(*o.m_store);
    return *this;
  }

  ComputeOpSlice(ComputeOpSlice &&o) : m_store(std::move(o.m_store)) {}
  ComputeOpSlice &operator=(ComputeOpSlice &&o) {
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

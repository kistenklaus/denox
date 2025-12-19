#pragma once

#include "denox/memory/container/uvec2.hpp"
#include "model/PoolFunction.hpp"
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

} // namespace vkcnn

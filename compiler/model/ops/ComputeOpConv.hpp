#pragma once

#include "memory/container/optional.hpp"
#include "memory/container/shared_ptr.hpp"
#include "memory/container/uvec2.hpp"
#include "memory/dtype/dtype.hpp"
#include "memory/tensor/BiasTensor.hpp"
#include "memory/tensor/FilterTensor.hpp"
#include <memory>

namespace denox::compiler {

struct ComputeOpConv {
  struct Storage {
    memory::shared_ptr<memory::FilterTensor> W;
    memory::shared_ptr<memory::BiasTensor> B;
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
    // NOTE: I know this looks bad, but we actually have to deep copy this.
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

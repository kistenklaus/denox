#pragma once

#include "vkcnn/common/tensor/BiasHostTensor.hpp"
#include "vkcnn/common/tensor/FilterHostTensor.hpp"
#include <glm/vec2.hpp>
#include <memory>
#include <optional>

namespace vkcnn {

struct ComputeOpConv {
  struct Storage {
    std::shared_ptr<FilterHostTensor> W;
    std::shared_ptr<BiasHostTensor> B;
    glm::uvec2 padding;
    glm::uvec2 stride;

    std::optional<FloatType> atype;
  };

  const Storage &operator*() const { return *m_store; }
  Storage &operator*() { return *m_store; }
  const Storage *operator->() const { return m_store.get(); }
  Storage *operator->() { return m_store.get(); }

  ComputeOpConv(FilterHostTensor W, std::optional<BiasHostTensor> B,
                glm::uvec2 padding, glm::uvec2 stride,
                std::optional<FloatType> atype)
      : m_store(std::make_unique<Storage>(
            std::make_shared<FilterHostTensor>(std::move(W)),
            B.has_value() ? std::make_shared<BiasHostTensor>(std::move(*B))
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

} // namespace vkcnn

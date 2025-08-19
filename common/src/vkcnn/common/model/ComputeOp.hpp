#pragma once
#include "vkcnn/common/ActivationFunction.hpp"
#include "vkcnn/common/FilterMode.hpp"
#include "vkcnn/common/PoolFunction.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
#include <cassert>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <memory>
#include <stdexcept>
#include <utility>

namespace vkcnn {

class Model;

struct ComputeOpConv {
  struct Storage {
    glm::uvec2 kernelSize;
    unsigned int K;
    bool bias;
    glm::uvec2 padding;
    glm::uvec3 stride;

    std::optional<FloatType> atype;
  };

  const Storage &operator*() const { return *m_store; }
  Storage &operator*() { return *m_store; }
  const Storage *operator->() const { return m_store.get(); }
  Storage *operator->() { return m_store.get(); }

  ComputeOpConv(glm::uvec2 kernelSize, unsigned int K,
                bool bias, glm::uvec2 padding, glm::uvec2 stride,
                std::optional<FloatType> atype)
      : m_store(std::make_shared<Storage>(kernelSize, K, bias, padding,
                                          stride, atype)) {}

private:
  std::shared_ptr<Storage> m_store;
};

struct ComputeOpActivation {
  ActivationFunction func;
};

struct ComputeOpUpsample {
  unsigned int scalingFactor;
  FilterMode mode;
};

struct ComputeOpPool {
  struct Storage {
    glm::uvec2 kernelSize;
    glm::uvec2 padding;
    glm::uvec2 stride;
    PoolFunction func;
  };

  const Storage &operator*() const { return *m_store; }
  Storage &operator*() { return *m_store; }
  const Storage *operator->() const { return m_store.get(); }
  Storage *operator->() { return m_store.get(); }

  ComputeOpPool(glm::uvec2 kernelSize, glm::uvec2 padding, glm::uvec2 stride,
                PoolFunction func)
      : m_store(std::make_shared<Storage>(kernelSize, padding, stride, func)) {}

private:
  std::shared_ptr<Storage> m_store;
};

struct ComputeOpConcat {};

enum class ComputeOpTag {
  Conv,
  Activation,
  Upsample,
  Pool,
  Concat,
};

class ComputeOp {
public:
  ComputeOp(ComputeOpConv conv)
      : m_tag(ComputeOpTag::Conv), m_uni(std::move(conv)) {}

  ComputeOp(ComputeOpActivation activation)
      : m_tag(ComputeOpTag::Activation), m_uni(std::move(activation)) {}

  ComputeOp(ComputeOpUpsample upsample)
      : m_tag(ComputeOpTag::Upsample), m_uni(std::move(upsample)) {}

  ComputeOp(ComputeOpPool pool)
      : m_tag(ComputeOpTag::Pool), m_uni(std::move(pool)) {}

  ComputeOp(ComputeOpConcat concat)
      : m_tag(ComputeOpTag::Concat), m_uni(std::move(concat)) {}

  ~ComputeOp() {
    switch (m_tag) {
    case ComputeOpTag::Conv:
      std::destroy_at(&m_uni.concat);
      break;
    case ComputeOpTag::Activation:
      std::destroy_at(&m_uni.activation);
      break;
    case ComputeOpTag::Upsample:
      std::destroy_at(&m_uni.upsample);
      break;
    case ComputeOpTag::Pool:
      std::destroy_at(&m_uni.pool);
      break;
    case ComputeOpTag::Concat:
      std::destroy_at(&m_uni.concat);
      break;
    default:
#ifndef NDEBUG
      throw std::runtime_error("Not implemented");
#else
      std::unreachable();
#endif
    }
  }

  ComputeOpTag tag() const { return m_tag; }

  const ComputeOpConv &conv() const {
    assert(m_tag == ComputeOpTag::Conv);
    return m_uni.conv;
  }
  ComputeOpConv &conv() {
    assert(m_tag == ComputeOpTag::Conv);
    return m_uni.conv;
  }

  const ComputeOpActivation &activation() const {
    assert(m_tag == ComputeOpTag::Activation);
    return m_uni.activation;
  }
  ComputeOpActivation &activation() {
    assert(m_tag == ComputeOpTag::Activation);
    return m_uni.activation;
  }

  const ComputeOpUpsample &upsample() const {
    assert(m_tag == ComputeOpTag::Upsample);
    return m_uni.upsample;
  }
  ComputeOpUpsample &upsample() {
    assert(m_tag == ComputeOpTag::Upsample);
    return m_uni.upsample;
  }

  const ComputeOpPool &pool() const {
    assert(m_tag == ComputeOpTag::Pool);
    return m_uni.pool;
  }

  ComputeOpPool &pool() {
    assert(m_tag == ComputeOpTag::Pool);
    return m_uni.pool;
  }

  const ComputeOpConcat &concat() const {
    assert(m_tag == ComputeOpTag::Concat);
    return m_uni.concat;
  }

  ComputeOpConcat &concat() {
    assert(m_tag == ComputeOpTag::Concat);
    return m_uni.concat;
  }

private:
  friend Model;
  union Uni {
    ComputeOpConv conv;
    ComputeOpActivation activation;
    ComputeOpUpsample upsample;
    ComputeOpPool pool;
    ComputeOpConcat concat;

    Uni(ComputeOpConv conv) : conv(std::move(conv)) {}
    Uni(ComputeOpActivation acti) : activation(std::move(acti)) {}
    Uni(ComputeOpUpsample upsample) : upsample(std::move(upsample)) {}
    Uni(ComputeOpPool pool) : pool(std::move(pool)) {}
    Uni(ComputeOpConcat concat) : concat(std::move(concat)) {}

    ~Uni() {}
  };
  ComputeOpTag m_tag;
  Uni m_uni;
};

} // namespace vkcnn::graph

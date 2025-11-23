#pragma once

#include "model/ops/ComputeOpActivation.hpp"
#include "model/ops/ComputeOpConcat.hpp"
#include "model/ops/ComputeOpConv.hpp"
#include "model/ops/ComputeOpPad.hpp"
#include "model/ops/ComputeOpPool.hpp"
#include "model/ops/ComputeOpSlice.hpp"
#include "model/ops/ComputeOpUpsample.hpp"

namespace denox::compiler {

enum class ComputeOpTag {
  None,
  Conv,
  Activation,
  Upsample,
  Pool,
  Concat,
  Pad,
  Slice,
};

class Model;

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

  ComputeOp(ComputeOpPad pad)
      : m_tag(ComputeOpTag::Pad), m_uni(std::move(pad)) {}

  ComputeOp(ComputeOpSlice slice)
      : m_tag(ComputeOpTag::Slice), m_uni(std::move(slice)) {}

  ComputeOp(const ComputeOp &o) : m_tag(o.m_tag) {
    switch (o.m_tag) {
    case ComputeOpTag::Conv:
      new (&m_uni.conv) ComputeOpConv(o.m_uni.conv);
      break;
    case ComputeOpTag::Activation:
      new (&m_uni.activation) ComputeOpActivation(o.m_uni.activation);
      break;
    case ComputeOpTag::Upsample:
      new (&m_uni.upsample) ComputeOpUpsample(o.m_uni.upsample);
      break;
    case ComputeOpTag::Pool:
      new (&m_uni.pool) ComputeOpPool(o.m_uni.pool);
      break;
    case ComputeOpTag::Concat:
      new (&m_uni.concat) ComputeOpConcat(o.m_uni.concat);
      break;
    case ComputeOpTag::Pad:
      new (&m_uni.pad) ComputeOpPad(o.m_uni.pad);
      break;
    case ComputeOpTag::Slice:
      new (&m_uni.slice) ComputeOpSlice(o.m_uni.slice);
      break;
    case ComputeOpTag::None:
      break;
    }
  }
  ComputeOp &operator=(const ComputeOp &o) {
    if (this == &o) {
      return *this;
    }
    m_tag = o.m_tag;
    switch (m_tag) {
    case ComputeOpTag::Conv:
      m_uni.conv = o.m_uni.conv;
      break;
    case ComputeOpTag::Activation:
      m_uni.activation = o.m_uni.activation;
      break;
    case ComputeOpTag::Upsample:
      m_uni.upsample = o.m_uni.upsample;
      break;
    case ComputeOpTag::Pool:
      m_uni.pool = o.m_uni.pool;
      break;
    case ComputeOpTag::Concat:
      m_uni.concat = o.m_uni.concat;
      break;
    case ComputeOpTag::Pad:
      m_uni.pad = o.m_uni.pad;
      break;
    case ComputeOpTag::Slice:
      m_uni.slice = o.m_uni.slice;
      break;
    case ComputeOpTag::None:
      break;
    }
    return *this;
  }

  ComputeOp(ComputeOp &&o) : m_tag(o.m_tag) {
    switch (o.m_tag) {
    case ComputeOpTag::Conv:
      new (&m_uni.conv) ComputeOpConv(std::move(o.m_uni.conv));
      break;
    case ComputeOpTag::Activation:
      new (&m_uni.activation)
          ComputeOpActivation(std::move(o.m_uni.activation));
      break;
    case ComputeOpTag::Upsample:
      new (&m_uni.upsample) ComputeOpUpsample(std::move(o.m_uni.upsample));
      break;
    case ComputeOpTag::Pool:
      new (&m_uni.pool) ComputeOpPool(std::move(o.m_uni.pool));
      break;
    case ComputeOpTag::Concat:
      new (&m_uni.concat) ComputeOpConcat(std::move(o.m_uni.concat));
      break;
    case ComputeOpTag::Pad:
      new (&m_uni.pad) ComputeOpPad(std::move(o.m_uni.pad));
      break;
    case ComputeOpTag::Slice:
      new (&m_uni.slice) ComputeOpSlice(std::move(o.m_uni.slice));
      break;
    case ComputeOpTag::None:
      break;
    }
  }
  ComputeOp &operator=(ComputeOp &&o) {
    if (this == &o) {
      return *this;
    }
    release();
    m_tag = o.m_tag;
    switch (m_tag) {
    case ComputeOpTag::Conv:
      m_uni.conv = std::move(o.m_uni.conv);
      break;
    case ComputeOpTag::Activation:
      m_uni.activation = std::move(o.m_uni.activation);
      break;
    case ComputeOpTag::Upsample:
      m_uni.upsample = std::move(o.m_uni.upsample);
      break;
    case ComputeOpTag::Pool:
      m_uni.pool = std::move(o.m_uni.pool);
      break;
    case ComputeOpTag::Concat:
      m_uni.concat = std::move(o.m_uni.concat);
      break;
    case ComputeOpTag::Pad:
      m_uni.pad = std::move(o.m_uni.pad);
      break;
    case ComputeOpTag::Slice:
      m_uni.slice = std::move(o.m_uni.slice);
      break;
    case ComputeOpTag::None:
      break;
    }
    return *this;
  }

  ~ComputeOp() { release(); }

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

  const ComputeOpPad &pad() const {
    assert(m_tag == ComputeOpTag::Pad);
    return m_uni.pad;
  }

  ComputeOpPad &pad() {
    assert(m_tag == ComputeOpTag::Pad);
    return m_uni.pad;
  }

  const ComputeOpSlice &slice() const {
    assert(m_tag == ComputeOpTag::Slice);
    return m_uni.slice;
  }

  ComputeOpSlice &slice() {
    assert(m_tag == ComputeOpTag::Slice);
    return m_uni.slice;
  }

private:
  void release() {
    switch (m_tag) {
    case ComputeOpTag::Conv:
      std::destroy_at(&m_uni.conv);
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
    case ComputeOpTag::Pad:
      std::destroy_at(&m_uni.pad);
      break;
    case ComputeOpTag::Slice:
      std::destroy_at(&m_uni.slice);
      break;
    case ComputeOpTag::None:
      break;
    }
    m_tag = ComputeOpTag::None;
  }

  friend Model;
  union Uni {
    char m_raw = 0;
    ComputeOpConv conv;
    ComputeOpActivation activation;
    ComputeOpUpsample upsample;
    ComputeOpPool pool;
    ComputeOpConcat concat;
    ComputeOpPad pad;
    ComputeOpSlice slice;

    Uni(ComputeOpConv conv) : conv(std::move(conv)) {}
    Uni(ComputeOpActivation acti) : activation(std::move(acti)) {}
    Uni(ComputeOpUpsample upsample) : upsample(std::move(upsample)) {}
    Uni(ComputeOpPool pool) : pool(std::move(pool)) {}
    Uni(ComputeOpConcat concat) : concat(std::move(concat)) {}
    Uni(ComputeOpPad pad) : pad(std::move(pad)) {}
    Uni(ComputeOpSlice slice) : slice(std::move(slice)) {}

    Uni() : m_raw(0) {}

    ~Uni() {}
  };
  ComputeOpTag m_tag;
  Uni m_uni;
};

} // namespace denox::compiler

#pragma once

#include "memory/container/optional.hpp"
#include "memory/dtype/dtype.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include "symbolic/sym_vec2.hpp"

namespace denox::compiler {

class Model;
class Tensor;

class ComputeTensor {
public:
  friend Model;
  friend Tensor;
  explicit ComputeTensor(sym_vec2 extent, unsigned int channels,
                         memory::optional<memory::ActivationLayout> layout,
                         memory::optional<memory::Dtype> type)
      : m_extent(extent), m_channels(channels), m_layout(layout), m_type(type) {
  }

  unsigned int channels() const { return m_channels; }
  memory::optional<memory::ActivationLayout> layout() const { return m_layout; }
  memory::optional<memory::Dtype> type() const { return m_type; }

private:
  sym_vec2 m_extent;
  unsigned int m_channels;
  memory::optional<memory::ActivationLayout> m_layout;
  memory::optional<memory::Dtype> m_type;
};

} // namespace denox::compiler

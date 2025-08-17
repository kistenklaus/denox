#pragma once

#include "vkcnn/common/tensor/ActivationLayout.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"

#include <optional>

namespace vkcnn::graph {

class ComputeGraph;

class ComputeTensor {
public:
  friend ComputeGraph;
  explicit ComputeTensor(unsigned int channels,
                       std::optional<ActivationLayout> layout,
                       std::optional<FloatType> type)
      : m_channels(channels), m_layout(layout), m_type(type) {}

  unsigned int channels() const { return m_channels; }
  std::optional<ActivationLayout> layout() const { return m_layout; }
  std::optional<FloatType> type() const { return m_type; }

private:
  unsigned int m_channels;
  std::optional<ActivationLayout> m_layout;
  std::optional<FloatType> m_type;
};

} // namespace vkcnn::graph

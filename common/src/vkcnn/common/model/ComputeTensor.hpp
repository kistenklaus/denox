#pragma once

#include "vkcnn/common/model/SymTensorExtent.hpp"
#include "vkcnn/common/tensor/ActivationLayout.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"

#include <optional>

namespace vkcnn {

class Model;
class Tensor;

class ComputeTensor {
public:
  friend Model;
  friend Tensor;
  explicit ComputeTensor(SymTensorExtent extent, unsigned int channels,
                         std::optional<ActivationLayout> layout,
                         std::optional<FloatType> type)
      : m_extent(extent), m_channels(channels), m_layout(layout), m_type(type) {
  }

  unsigned int channels() const { return m_channels; }
  std::optional<ActivationLayout> layout() const { return m_layout; }
  std::optional<FloatType> type() const { return m_type; }

private:
  SymTensorExtent m_extent;
  unsigned int m_channels;
  std::optional<ActivationLayout> m_layout;
  std::optional<FloatType> m_type;
};

} // namespace vkcnn

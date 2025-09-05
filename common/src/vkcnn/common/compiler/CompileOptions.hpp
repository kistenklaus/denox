#pragma once

#include "vkcnn/common/tensor/ActivationLayout.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
#include <vector>
namespace vkcnn {

class CompileOptions {
  public:

  CompileOptions()
      : m_dtypes({FloatType::F16}),
        m_layouts({ActivationLayout::CHW, ActivationLayout::HWC,
                   ActivationLayout::CHWC8}) {}

  std::span<const FloatType> dtypes() const { return m_dtypes; }
  std::span<const ActivationLayout> layouts() const { return m_layouts; }

private:
  // The first dtype will be uses as a parameter.
  // dtypes not present here will not be generated and lead to an exception.
  std::vector<FloatType> m_dtypes;
  // All allowed layouts, layouts not present here, will not be
  // used for intermediate tensors.
  std::vector<ActivationLayout> m_layouts;
};

} // namespace vkcnn

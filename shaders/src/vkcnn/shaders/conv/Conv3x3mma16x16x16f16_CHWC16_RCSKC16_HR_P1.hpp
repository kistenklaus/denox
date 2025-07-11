#pragma once

#include "vkcnn/common/ops/OpConv.hpp"
#include "vkcnn/common/shader/conv/ConvShaderSource.hpp"
#include "vkcnn/shaders/conv/ConvTemplate.hpp"
#include <fmt/base.h>

namespace vkcnn::shaders {

class Conv3x3mma16x16x16_CHWC16_RCSKC16_HR_P1 final : public ConvTemplate {
public:
  Conv3x3mma16x16x16_CHWC16_RCSKC16_HR_P1();

  bool supports(const OpConv &op) const final override;

  ConvShaderSource do_specialize(const OpConv &op) const final override;

  std::string_view name() const final override {
    return "conv3x3mma16x16x16f16_CHWC16_RSCKC16_HR_P1";
  };

private:
  std::string m_source;
};

} // namespace vkcnn::shaders

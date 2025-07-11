#pragma once

#include "vkcnn/common/ops/OpConv.hpp"
#include "vkcnn/common/shader/conv/ConvShaderSource.hpp"
#include "vkcnn/shaders/conv/ConvTemplate.hpp"
#include <fmt/base.h>

namespace vkcnn::shaders {

class Conv3x3mma16x8x8_CHWC8_RSCKC8_NR_P2 final : public ConvTemplate {
public:
  Conv3x3mma16x8x8_CHWC8_RSCKC8_NR_P2();

  bool supports(const OpConv &op) const final override;

  ConvShaderSource do_specialize(const OpConv &op) const final override;

  std::string_view name() const final override {
    return "conv3x3mma16x8x8f16_CHWC8_RSCKC8_NR_P2";
  };

private:
  std::string m_source;
};

} // namespace vkcnn::shaders

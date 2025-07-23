#pragma once

#include "vkcnn/common/ops/OpConv.hpp"
#include "vkcnn/common/shader/conv/ConvShaderSource.hpp"
#include "vkcnn/shaders/conv/ConvTemplate.hpp"
#include <fmt/base.h>

#include <glm/vec3.hpp>

namespace vkcnn::shaders {

class Conv3x3mmaVectorized final : public ConvTemplate {
public:
  Conv3x3mmaVectorized(glm::uvec3 mmaShape);

  bool supports(const OpConv &op) const final override;

  ConvShaderSource do_specialize(const OpConv &op) const final override;

  std::string_view name() const final override {
    return "conv3x3mmaVectorized";
  };

private:
  std::string m_source;
  glm::uvec3 m_mmaShape;
};

} // namespace vkcnn::shaders

#pragma once

#include "vkcnn/common/ops/OpConv.hpp"
#include "vkcnn/common/shader/conv/ConvShaderSource.hpp"
#include <fmt/base.h>
#include <glm/ext/vector_uint2.hpp>
namespace vkcnn::shaders {

class ConvTemplate {
public:
  virtual ~ConvTemplate() = default;

  virtual bool supports(const OpConv &op) const = 0;

  std::optional<ConvShaderSource> specialize(const OpConv &op) const;

  virtual ConvShaderSource do_specialize(const OpConv &op) const = 0;

  virtual std::string_view name() const = 0;
};

} // namespace vkcnn::shaders

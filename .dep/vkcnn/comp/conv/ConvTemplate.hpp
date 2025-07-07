#pragma once

#include "vkcnn/common/shader/conv/ConvShaderSource.hpp"
#include "vkcnn/comp/conv/OpConv.hpp"
#include <fmt/base.h>
#include <glm/ext/vector_uint2.hpp>
namespace vkcnn::comp {

class ConvTemplate {
public:
  virtual ~ConvTemplate() = default;

  virtual bool supports(const OpConv &op) const = 0;
  std::optional<ConvShaderSource> specialize(const OpConv &op) const {
    if (!this->supports(op)) {
      return std::nullopt;
    } else {
      return do_specialize(op);
    }
  }

  virtual ConvShaderSource do_specialize(const OpConv &op) const = 0;
};

} // namespace vkcnn::comp

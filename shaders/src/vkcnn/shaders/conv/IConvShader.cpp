#include "./IConvShader.hpp"

namespace vkcnn::shaders {

std::optional<ConvShaderSource>
IConvShader::specialize(const OpConv &op) const {
  if (!this->supports(op)) {
    return std::nullopt;
  } else {
    return do_specialize(op);
  }
}

} // namespace vkcnn::shaders

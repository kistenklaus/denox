#include "./ICopyShader.hpp"

namespace vkcnn::shaders {

std::optional<CopyShaderSource>
ICopyShader::specialize(const OpCopy &op) const {
  if (!this->supports(op)) {
    return std::nullopt;
  } else {
    return do_specialize(op);
  }
}

} // namespace vkcnn::shaders

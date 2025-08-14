#include "./IActivationShader.hpp"

namespace vkcnn::shaders {

std::optional<ActivationShaderSource>
IActivationShader::specialize(const OpActivation &op) const {
  if (!this->supports(op)) {
    return std::nullopt;
  } else {
    return do_specialize(op);
  }
}

} // namespace vkcnn::shaders

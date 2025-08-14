#include "./IUpsampleShader.hpp"

namespace vkcnn::shaders {

std::optional<UpsampleShaderSource>
IUpsampleShader::specialize(const OpUpsample &op) const {
  if (!this->supports(op)) {
    return std::nullopt;
  } else {
    return do_specialize(op);
  }
}
} // namespace vkcnn::shaders

#include "./IPoolShader.hpp"

namespace vkcnn::shaders {

std::optional<PoolShaderSource>
IPoolShader::specialize(const OpPool &op) const {
  if (!this->supports(op)) {
    return std::nullopt;
  } else {
    return do_specialize(op);
  }
}

} // namespace vkcnn::shader

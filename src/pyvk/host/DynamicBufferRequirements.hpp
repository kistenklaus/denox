#pragma once

#include "pyvk/host/Lifetime.hpp"
#include <functional>
#include <glm/ext/vector_uint2.hpp>
#include <memory>
namespace pyvk {

class DynamicBufferRequirements {
public:
  DynamicBufferRequirements(
      unsigned int channels,
      std::shared_ptr<std::function<glm::uvec2(glm::uvec2)>> imageSizeFunc) {}

  std::size_t resolveByteRequirements(glm::uvec2 inputImageSize) {
    glm::uvec2 wh = (*m_imageSizeFunc)(inputImageSize);
    return static_cast<std::size_t>(wh.x) * static_cast<std::size_t>(wh.y) *
           static_cast<std::size_t>(channels);
  }

private:
  Lifetime m_lifetime;
  std::shared_ptr<std::function<glm::uvec2(glm::uvec2)>> m_imageSizeFunc;
  unsigned int channels;
};

} // namespace pyvk

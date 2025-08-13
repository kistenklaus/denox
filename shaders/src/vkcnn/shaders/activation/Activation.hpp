#pragma once

#include <glm/vec3.hpp>
#include <string>
#include <vector>
namespace vkcnn::shaders {

class Activation {
public:
private:
  glm::uvec3 tileSize;
  std::vector<std::byte> m_source;
  std::string m_name;
};

} // namespace vkcnn::shaders

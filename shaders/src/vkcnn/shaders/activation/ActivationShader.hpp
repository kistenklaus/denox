#pragma once

#include "vkcnn/common/ops/OpActivation.hpp"
#include "vkcnn/common/shader/ActivationShaderSource.hpp"
#include "vkcnn/shaders/activation/IActivationShader.hpp"
#include <glm/vec3.hpp>
#include <optional>
#include <string>
#include <vector>
namespace vkcnn::shaders {

class ActivationShader : public IActivationShader {
public:
  ActivationShader(glm::uvec3 iTile, glm::uvec3 wTile);
  ActivationShader();

  bool supports(const OpActivation &op) const final override;

  ActivationShaderSource
  do_specialize(const OpActivation &op) const final override;

  std::string_view name() const final override;

private:
  struct TileSizes {
    glm::uvec3 iTile;
    glm::uvec3 wTile;
  };
  std::optional<TileSizes> m_tileSizes;
  std::vector<std::byte> m_source;
  std::string m_name;
};

} // namespace vkcnn::shaders

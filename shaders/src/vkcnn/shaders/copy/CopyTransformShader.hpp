#pragma once
#include "vkcnn/common/ops/OpCopy.hpp"
#include "vkcnn/common/shader/CopyShaderSource.hpp"
#include "vkcnn/shaders/copy/ICopyShader.hpp"
#include <glm/vec3.hpp>

namespace vkcnn::shaders {

class CopyTransformShader : public ICopyShader {
public:
  explicit CopyTransformShader(glm::uvec3 wTile, glm::uvec3 iTile);

  /// NOTE: Autotuning constructor.
  CopyTransformShader();

  bool supports(const OpCopy &op) const final override;

  CopyShaderSource do_specialize(const OpCopy &op) const final override;

  std::string_view name() const final override;

private:
  struct TileSizes {
    glm::uvec3 wTile;
    glm::uvec3 iTile;
  };
  std::optional<TileSizes> m_tileSizes;
  std::vector<std::byte> m_source;
  std::string m_name;
};
} // namespace vkcnn::shaders

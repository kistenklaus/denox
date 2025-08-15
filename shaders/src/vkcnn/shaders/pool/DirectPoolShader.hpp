#pragma once

#include "vkcnn/common/ops/OpPool.hpp"
#include "vkcnn/common/shader/PoolShaderSource.hpp"
#include "vkcnn/shaders/pool/IPoolShader.hpp"
namespace vkcnn::shaders {

class DirectPoolShader : public IPoolShader {
public:
  DirectPoolShader(glm::uvec3 iTile, glm::uvec3 wTile);
  DirectPoolShader();

  bool supports(const OpPool &op) const final override;

  PoolShaderSource do_specialize(const OpPool &op) const final override;

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

#pragma once

#include "vkcnn/common/ops/OpUpsample.hpp"
#include "vkcnn/common/shader/UpsampleShaderSource.hpp"
#include "vkcnn/shaders/upsample/IUpsampleShader.hpp"
namespace vkcnn::shaders {

class DirectUpsampleShader : public IUpsampleShader {
public:
  explicit DirectUpsampleShader(glm::uvec3 iTile, glm::uvec3 wTile);
  DirectUpsampleShader();

  bool supports(const OpUpsample &op) const final override;

  UpsampleShaderSource do_specialize(const OpUpsample &op) const final override;

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

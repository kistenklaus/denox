#pragma once

#include "vkcnn/common/ops/OpConv.hpp"
#include "vkcnn/common/shader/ConvShaderSource.hpp"
#include "vkcnn/shaders/conv/ConvTemplate.hpp"
#include <fmt/base.h>

#include <glm/vec3.hpp>

namespace vkcnn::shaders {

class ConvGEMM final : public ConvTemplate {
public:
  ConvGEMM(glm::uvec3 cmShape, glm::uvec3 sgTile, glm::uvec2 wgTile, bool asyncRead = true);

  bool supports(const OpConv &op) const final override;

  ConvShaderSource do_specialize(const OpConv &op) const final override;

  std::string_view name() const final override;

private:
  std::string m_source;
  glm::uvec3 m_cmShape;
  glm::uvec3 m_sgTile;
  glm::uvec2 m_wgTile;
  bool m_asyncRead;
  std::string m_name;
};

} // namespace vkcnn::shaders

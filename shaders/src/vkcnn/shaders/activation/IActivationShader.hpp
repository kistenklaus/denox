#pragma once

#include "vkcnn/common/ops/OpActivation.hpp"
#include "vkcnn/common/shader/ActivationShaderSource.hpp"
#include <fmt/base.h>
#include <glm/ext/vector_uint2.hpp>
namespace vkcnn::shaders {

class IActivationShader {
public:
  virtual ~IActivationShader() = default;

  virtual bool supports(const OpActivation &op) const = 0;

  std::optional<ActivationShaderSource>
  specialize(const OpActivation &op) const;

  virtual ActivationShaderSource
  do_specialize(const OpActivation &op) const = 0;

  virtual std::string_view name() const = 0;
};

} // namespace vkcnn::shaders

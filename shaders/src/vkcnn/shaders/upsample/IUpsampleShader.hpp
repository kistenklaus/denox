#pragma once

#include "vkcnn/common/ops/OpUpsample.hpp"
#include "vkcnn/common/shader/UpsampleShaderSource.hpp"
namespace vkcnn::shaders {

class IUpsampleShader {
public:
  virtual ~IUpsampleShader() = default;

  virtual bool supports(const OpUpsample &op) const = 0;

  std::optional<UpsampleShaderSource> specialize(const OpUpsample &op) const;

  virtual UpsampleShaderSource do_specialize(const OpUpsample &op) const = 0;

  virtual std::string_view name() const = 0;
};

} // namespace vkcnn::shaders

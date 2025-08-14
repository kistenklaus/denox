#pragma once

#include "vkcnn/common/ops/OpCopy.hpp"
#include "vkcnn/common/shader/CopyShaderSource.hpp"
namespace vkcnn::shaders {

class ICopyShader {
public:
  virtual ~ICopyShader() = default;

  virtual bool supports(const OpCopy &op) const = 0;

  std::optional<CopyShaderSource> specialize(const OpCopy &op) const;

  virtual CopyShaderSource do_specialize(const OpCopy &op) const = 0;

  virtual std::string_view name() const = 0;
};

} // namespace vkcnn::shaders

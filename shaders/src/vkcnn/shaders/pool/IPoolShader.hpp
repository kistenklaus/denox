#pragma once

#include "vkcnn/common/ops/OpPool.hpp"
#include "vkcnn/common/shader/PoolShaderSource.hpp"

namespace vkcnn::shaders {

class IPoolShader {
public:
  virtual ~IPoolShader() = default;

  virtual bool supports(const OpPool &op) const = 0;

  std::optional<PoolShaderSource> specialize(const OpPool &op) const;

  virtual PoolShaderSource do_specialize(const OpPool &op) const = 0;

  virtual std::string_view name() const = 0;
};

} // namespace vkcnn::shaders

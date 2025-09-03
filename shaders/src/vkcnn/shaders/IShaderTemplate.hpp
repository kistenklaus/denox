#pragma once

#include "vkcnn/common/model/ComputeOp.hpp"
#include <span>
#include <vector>
namespace vkcnn {

class ShaderOp {
  unsigned int opBegin;
  unsigned int opEnd;
};

class IShaderTemplate {
  public:
    ~IShaderTemplate() = default;

    virtual std::vector<ShaderOp> capabilities(std::span<const ComputeOp*> ops) = 0;

  private:
};


}

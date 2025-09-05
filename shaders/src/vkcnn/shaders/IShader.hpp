#pragma once

#include "vkcnn/common/hypergraph/ConstGraph.hpp"
#include "vkcnn/common/model/ComputeOp.hpp"
#include <span>
#include <vector>
namespace vkcnn {

class ShaderOp {
  unsigned int opBegin;
  unsigned int opEnd;
};

class IShader {
  public:
    ~IShader() = default;

  private:
};


}

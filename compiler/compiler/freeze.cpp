#include "compiler/freeze.hpp"

namespace denox::compiler {

ConstModel freeze(const AdjModel &adjModel) {
  return ConstModel {
    .graph = ConstModel::Graph(adjModel.graph),
    .input = adjModel.input,
    .output = adjModel.output
  };
}

} // namespace denox::compiler

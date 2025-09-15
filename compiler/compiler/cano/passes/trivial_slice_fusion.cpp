#include "compiler/cano/passes/passes.hpp"
#include "compiler/ir/LinkedModel.hpp"
#include "diag/logging.hpp"
#include "memory/container/vector.hpp"

namespace denox::compiler::cano {

void trivial_slice_fusion(LinkedModel &model) {
  DENOX_TRACE("Hello");
  using Graph = LinkedModel::Graph;
  using NodeHandle = Graph::NodeHandle;

  memory::vector<Graph::EdgeIt> stack;
  memory::vector<NodeHandle> fusion;

  { // initalize
    auto outgoing = model.input->outgoing();
    auto it = outgoing.begin();
    auto end = outgoing.end();
    while (it++ != end) {
      stack.push_back(it);
    }
  }
}

} // namespace denox::compiler::cano

#pragma once

#include "denox/memory/hypergraph/NodeId.hpp"
#include <string>

namespace denox::compiler {

struct ModelInterfaceDescriptor {
  memory::NodeId nodeId;
  std::string name;
};

} // namespace denox::compiler

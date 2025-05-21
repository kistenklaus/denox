#pragma once
#include "pyvk/host/DispatchOp.hpp"
#include "pyvk/host/NetworkDescription.hpp"
#include <vector>

namespace pyvk {

std::vector<DispatchOp> fuseLayers(const NetworkDescription& network);

}

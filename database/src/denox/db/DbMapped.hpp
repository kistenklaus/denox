#pragma once

#include "denox/db/DbComputeDispatch.hpp"
#include "denox/db/DbEnv.hpp"
#include "denox/db/DbShaderBinary.hpp"
#include "denox/io/fs/Path.hpp"

namespace denox {

struct DbMapped {
  io::Path m_path;
  std::vector<DbEnv> environments;
  std::vector<DbShaderBinary> binaries;
  std::vector<DbComputeDispatch> dispatches;
};

} // namespace denox

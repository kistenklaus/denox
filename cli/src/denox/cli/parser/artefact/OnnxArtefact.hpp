#include "denox/cli/io/IOEndpoint.hpp"

struct OnnxArtefact {
  IOEndpoint endpoint; // must exist
  std::vector<std::byte> data;
};

#include "denox/cli/io/IOEndpoint.hpp"

struct DnxArtefact {
  IOEndpoint endpoint; // must exist.
  std::vector<std::byte> data;
};

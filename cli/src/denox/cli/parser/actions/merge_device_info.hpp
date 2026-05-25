#pragma once

#include "denox/cli/io/IOEndpoint.hpp"
#include "denox/memory/container/small_vector.hpp"
struct MergeDeviceInfo {
  denox::memory::small_vector<IOEndpoint, 2> device_infos;
  IOEndpoint output;
};

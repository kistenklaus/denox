#pragma once

#include "denox/cli/io/IOEndpoint.hpp"
#include "denox/device_info/ApiVersion.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/string.hpp"

struct QueryDeviceInfo {
  denox::memory::optional<denox::memory::string> deviceName;
  denox::memory::optional<IOEndpoint> output;
  denox::ApiVersion apiVersion = denox::ApiVersion::VULKAN_1_4;
};

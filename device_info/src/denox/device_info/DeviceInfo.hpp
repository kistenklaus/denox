#pragma once

#include "denox/device_info/ApiVersion.hpp"
#include "denox/device_info/CoopmatProperties.hpp"
#include "denox/device_info/LayoutRules.hpp"
#include "denox/device_info/MemoryModel.hpp"
#include "denox/device_info/ResourceLimits.hpp"
#include "denox/device_info/SubgroupProperties.hpp"

namespace denox {

struct DeviceInfo {
  ApiVersion apiVersion;
  memory::string name;
  ResourceLimits limits;
  SubgroupProperties subgroup;
  LayoutRules layouts;
  MemoryModelProperties memoryModel;
  CoopmatProperties coopmat;
};

} // namespace denox::compiler

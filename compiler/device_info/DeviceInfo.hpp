#pragma once

#include "device_info/ApiVersion.hpp"
#include "device_info/CoopmatProperties.hpp"
#include "device_info/LayoutRules.hpp"
#include "device_info/MemoryModel.hpp"
#include "device_info/ResourceLimits.hpp"
#include "device_info/SubgroupProperties.hpp"

namespace denox::compiler {

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

#pragma once

#include "device_info/CoopmatProperties.hpp"
#include "device_info/LayoutRules.hpp"
#include "device_info/MemoryModel.hpp"
#include "device_info/ResourceLimits.hpp"
#include "device_info/SubgroupProperties.hpp"
#include <cstdint>
namespace denox::compiler {

struct DeviceInfo {
  std::uint32_t apiVersion;
  ResourceLimits limits;
  SubgroupProperties subgroup;
  LayoutRules layouts;
  MemoryModelProperties memoryModel;
  CoopmatProperties coopmat;
};

} // namespace denox::compiler

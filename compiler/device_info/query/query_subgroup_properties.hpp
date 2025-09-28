#pragma once

#include "device_info/SubgroupProperties.hpp"
#include "vulkan/vulkan.hpp"

namespace denox::compiler::device_info::query {

SubgroupProperties query_subgroup_properties(vk::Instance instance,
                                             vk::PhysicalDevice physicalDevice);
}

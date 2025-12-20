#pragma once

#include "denox/device_info/SubgroupProperties.hpp"
#include <vulkan/vulkan.hpp>

namespace denox {

SubgroupProperties query_subgroup_properties(vk::Instance instance,
                                             vk::PhysicalDevice physicalDevice);
}

#pragma once

#include "denox/device_info/LayoutRules.hpp"
#include <vulkan/vulkan.hpp>

namespace denox {

LayoutRules query_layout_rules(vk::Instance instance,
                               vk::PhysicalDevice physicalDevice);
}

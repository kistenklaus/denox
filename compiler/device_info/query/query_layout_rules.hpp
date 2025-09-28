#pragma once

#include "device_info/LayoutRules.hpp"
#include <vulkan/vulkan.hpp>

namespace denox::compiler::device_info::query {

LayoutRules query_layout_rules(vk::Instance instance,
                               vk::PhysicalDevice physicalDevice);
}

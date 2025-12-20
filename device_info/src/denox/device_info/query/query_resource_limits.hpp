#pragma once

#include "denox/device_info/ResourceLimits.hpp"

#include <vulkan/vulkan.hpp>

namespace denox {

ResourceLimits query_resource_limits(vk::Instance instance,
                                     vk::PhysicalDevice physicalDevice);

} // namespace denox::compiler::device_info::query

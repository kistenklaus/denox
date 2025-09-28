#pragma once

#include "device_info/CoopmatProperties.hpp"
#include <vulkan/vulkan.hpp>

namespace denox::compiler::device_info::query {

CoopmatProperties query_coopmat_properties(vk::Instance instance,
                                          vk::PhysicalDevice physicalDevice);
}

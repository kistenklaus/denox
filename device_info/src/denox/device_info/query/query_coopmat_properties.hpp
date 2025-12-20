#pragma once

#include "denox/device_info/CoopmatProperties.hpp"
#include <vulkan/vulkan.hpp>

namespace denox {

CoopmatProperties query_coopmat_properties(vk::Instance instance,
                                          vk::PhysicalDevice physicalDevice);
}

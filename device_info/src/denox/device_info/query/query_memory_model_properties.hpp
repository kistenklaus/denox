#pragma once

#include "denox/device_info/ApiVersion.hpp"
#include "denox/device_info/MemoryModel.hpp"
#include <vulkan/vulkan.hpp>

namespace denox {

MemoryModelProperties
query_memory_model_properties(vk::Instance instance,
                             vk::PhysicalDevice physicalDevice, ApiVersion apiVersion);
}

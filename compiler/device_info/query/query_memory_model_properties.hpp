#pragma once

#include "device_info/MemoryModel.hpp"
#include <vulkan/vulkan.hpp>

namespace denox::compiler::device_info::query {

MemoryModelProperties
query_model_model_properties(vk::Instance instance,
                             vk::PhysicalDevice physicalDevice);
}

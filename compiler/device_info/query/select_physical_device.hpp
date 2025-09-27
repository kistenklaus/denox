#pragma once

#include "memory/container/string.hpp"
#include "vulkan/vulkan.hpp"

namespace denox::compiler::device_info::query {

vk::PhysicalDevice select_physical_device(vk::Instance instance, const memory::string& deviceName);

}

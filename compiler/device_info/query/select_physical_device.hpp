#pragma once

#include "memory/container/string.hpp"
#include "memory/container/optional.hpp"
#include "vulkan/vulkan.hpp"

namespace denox::compiler::device_info::query {

vk::PhysicalDevice select_physical_device(vk::Instance instance, const memory::optional<memory::string>& deviceName);

}

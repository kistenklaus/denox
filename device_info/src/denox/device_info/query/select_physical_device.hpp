#pragma once

#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/string.hpp"
#include <vulkan/vulkan.hpp>

namespace denox {

vk::PhysicalDevice
select_physical_device(vk::Instance instance,
                       const memory::optional<memory::string> &deviceName);

}

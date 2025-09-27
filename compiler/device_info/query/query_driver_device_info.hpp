#pragma once

#include "vulkan/vulkan.hpp"
#include "device_info/DeviceInfo.hpp"

namespace denox::compiler::device_info {

DeviceInfo query_driver_device_info();

DeviceInfo query_driver_device_info(vk::Instance instance);

DeviceInfo query_driver_device_info(vk::Instance instance, vk::PhysicalDevice physicalDevice);

}


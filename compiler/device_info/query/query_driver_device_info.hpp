#pragma once

#include "device_info/ApiVersion.hpp"
#include "device_info/DeviceInfo.hpp"
#include "vulkan/vulkan.hpp"

namespace denox::compiler::device_info {

DeviceInfo query_driver_device_info(ApiVersion apiVersion,
                                    const memory::string &deviceName);

DeviceInfo query_driver_device_info(vk::Instance instance,
                                    const memory::string &deviceName, ApiVersion apiVersion);

DeviceInfo query_driver_device_info(vk::Instance instance,
                                    vk::PhysicalDevice physicalDevice, ApiVersion apiVersion);

} // namespace denox::compiler::device_info

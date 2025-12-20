#pragma once

#include "denox/device_info/ApiVersion.hpp"
#include "denox/device_info/DeviceInfo.hpp"
#include "denox/memory/container/optional.hpp"
#include <vulkan/vulkan.hpp>

namespace denox {

DeviceInfo
query_driver_device_info(ApiVersion& apiVersion,
                         const memory::optional<memory::string> &deviceName);

DeviceInfo
query_driver_device_info(vk::Instance instance,
                         const memory::optional<memory::string> &deviceName,
                         ApiVersion apiVersion);

DeviceInfo query_driver_device_info(vk::Instance instance,
                                    vk::PhysicalDevice physicalDevice,
                                    ApiVersion apiVersion);

} // namespace denox::compiler::device_info

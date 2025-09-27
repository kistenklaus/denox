#include "device_info/query/query_driver_device_info.hpp"
#include "device_info/DeviceInfo.hpp"
#include "device_info/query/create_query_instance.hpp"
#include "device_info/query/select_physical_device.hpp"
#include "fmt/format.h"

namespace denox::compiler::device_info {

DeviceInfo query_driver_device_info() {

  vk::Instance instance = query::create_query_instance(ApiVersion::VULKAN_1_4);

  return query_driver_device_info(instance);
}

DeviceInfo query_driver_device_info(vk::Instance instance) {
  vk::PhysicalDevice physicalDevice =
      query::select_physical_device(instance, "*RTX*");
  fmt::println("Selected {}", physicalDevice.getProperties().deviceName.data());
  return query_driver_device_info(instance, physicalDevice);
}

DeviceInfo query_driver_device_info(vk::Instance instance,
                                    vk::PhysicalDevice physicalDevice) {

  return DeviceInfo{};
}

} // namespace denox::compiler::device_info

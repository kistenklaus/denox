#include "device_info/query/query_driver_device_info.hpp"
#include "device_info/CoopmatProperties.hpp"
#include "device_info/DeviceInfo.hpp"
#include "device_info/MemoryModel.hpp"
#include "device_info/ResourceLimits.hpp"
#include "device_info/query/create_query_instance.hpp"
#include "device_info/query/query_coopmat_properties.hpp"
#include "device_info/query/query_layout_rules.hpp"
#include "device_info/query/query_memory_model_properties.hpp"
#include "device_info/query/query_resource_limits.hpp"
#include "device_info/query/query_subgroup_properties.hpp"
#include "device_info/query/select_physical_device.hpp"

namespace denox::compiler::device_info {

DeviceInfo
query_driver_device_info(ApiVersion& apiVersion,
                         const memory::optional<memory::string> &deviceName) {

  vk::Instance instance = query::create_query_instance(apiVersion);

  return query_driver_device_info(instance, deviceName, apiVersion);
}

DeviceInfo
query_driver_device_info(vk::Instance instance,
                         const memory::optional<memory::string> &deviceName,
                         ApiVersion apiVersion) {
  vk::PhysicalDevice physicalDevice =
      query::select_physical_device(instance, deviceName);
  return query_driver_device_info(instance, physicalDevice, apiVersion);
}

DeviceInfo query_driver_device_info(vk::Instance instance,
                                    vk::PhysicalDevice physicalDevice,
                                    ApiVersion apiVersion) {
  memory::string deviceName = physicalDevice.getProperties().deviceName;
  ResourceLimits limits =
      query::query_resource_limits(instance, physicalDevice);
  SubgroupProperties subgroup =
      query::query_subgroup_properties(instance, physicalDevice);
  LayoutRules layouts = query::query_layout_rules(instance, physicalDevice);
  MemoryModelProperties memoryModel =
      query::query_model_model_properties(instance, physicalDevice);
  CoopmatProperties coopmat =
      query::query_coopmat_properties(instance, physicalDevice);

  return DeviceInfo{
      .apiVersion = apiVersion,
      .name = std::move(deviceName),
      .limits = std::move(limits),
      .subgroup = std::move(subgroup),
      .layouts = std::move(layouts),
      .memoryModel = std::move(memoryModel),
      .coopmat = std::move(coopmat),
  };
}

} // namespace denox::compiler::device_info

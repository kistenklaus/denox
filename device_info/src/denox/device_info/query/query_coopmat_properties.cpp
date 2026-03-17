#include "denox/device_info/query/query_coopmat_properties.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/dtype/dtype.hpp"
#include <fmt/format.h>
#include <unistd.h>
#include <vulkan/vulkan.hpp>

namespace denox {

CoopmatProperties query_coopmat_properties(vk::Instance instance,
                                           vk::PhysicalDevice physicalDevice) {
  CoopmatProperties props{};

  bool hasExtension = false;
  for (const auto &ext : physicalDevice.enumerateDeviceExtensionProperties()) {
    if (std::strcmp(ext.extensionName,
                    VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) {
      hasExtension = true;
      break;
    }
  }

  if (!hasExtension) {
    props.supported = false;
    return props;
  }

  props.supported = true;

  auto mapCompType =
      [](vk::ComponentTypeKHR type) -> memory::optional<memory::Dtype> {
    switch (type) {
    case vk::ComponentTypeKHR::eFloat16:
      return memory::Dtype::F16;
    case vk::ComponentTypeKHR::eFloat32:
      return memory::Dtype::F32;
    case vk::ComponentTypeKHR::eFloat64:
      return memory::Dtype::F64;
    default:
      return memory::nullopt;
    }
  };

  auto fp =
      reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
          instance.getProcAddr(
              "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));

  if (!fp) {
    props.supported = false;
    return props;
  }

  uint32_t count = 0;
  VkResult res = fp(physicalDevice, &count, nullptr);
  if (res != VK_SUCCESS) {
    throw std::runtime_error(
        "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(count) failed");
  }

  std::vector<VkCooperativeMatrixPropertiesKHR> raw(count);
  for (auto &r : raw) {
    r.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    r.pNext = nullptr;
  }

  res = fp(physicalDevice, &count, raw.data());
  if (res != VK_SUCCESS && res != VK_INCOMPLETE) {
    throw std::runtime_error(
        "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(data) failed");
  }

  raw.resize(count);

  for (const auto &p : raw) {
    auto atype = mapCompType(static_cast<vk::ComponentTypeKHR>(p.AType));
    auto btype = mapCompType(static_cast<vk::ComponentTypeKHR>(p.BType));
    auto ctype = mapCompType(static_cast<vk::ComponentTypeKHR>(p.CType));
    auto resultType =
        mapCompType(static_cast<vk::ComponentTypeKHR>(p.ResultType));

    if (atype && btype && ctype && resultType) {
      props.shapes.push_back(CoopmatShape{
          .M = p.MSize,
          .N = p.NSize,
          .K = p.KSize,
          .atype = *atype,
          .btype = *btype,
          .ctype = *ctype,
          .acctype = *resultType,
          .saturatingAccumulation = static_cast<bool>(p.saturatingAccumulation),
          .subgroupScope = (p.scope == VK_SCOPE_SUBGROUP_KHR),
      });
    }
  }

  return props;
}

} // namespace denox

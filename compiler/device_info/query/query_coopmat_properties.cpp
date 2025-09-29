#include "device_info/query/query_coopmat_properties.hpp"
#include "diag/unreachable.hpp"
#include "memory/container/optional.hpp"
#include "memory/dtype/dtype.hpp"
#include <fmt/format.h>
#include <unistd.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>

namespace denox::compiler::device_info::query {

CoopmatProperties
query_coopmat_properties([[maybe_unused]] vk::Instance instance,
                         vk::PhysicalDevice physicalDevice) {

  CoopmatProperties props{};
  props.supported = false;
#if defined(VK_KHR_cooperative_matrix) && VK_KHR_cooperative_matrix

  for (auto &ext : physicalDevice.enumerateDeviceExtensionProperties()) {
    if (strcmp(ext.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) ==
        0) {
      props.supported = true;
      break;
    }
  }

  const auto mapCompType =
      [](vk::ComponentTypeKHR type) -> memory::optional<memory::Dtype> {
    switch (type) {
    case vk::ComponentTypeKHR::eFloat16:
      return memory::Dtype::F16;
    case vk::ComponentTypeKHR::eFloat32:
      return memory::Dtype::F32;
    case vk::ComponentTypeKHR::eFloat64:
      return memory::Dtype::F64;
    case vk::ComponentTypeKHR::eSint8:
    case vk::ComponentTypeKHR::eSint16:
    case vk::ComponentTypeKHR::eSint32:
    case vk::ComponentTypeKHR::eSint64:
    case vk::ComponentTypeKHR::eUint8:
    case vk::ComponentTypeKHR::eUint16:
    case vk::ComponentTypeKHR::eUint32:
    case vk::ComponentTypeKHR::eUint64:
#if defined(VK_KHR_shader_bfloat16) && VK_KHR_shader_bfloat16
    case vk::ComponentTypeKHR::eBfloat16:
#endif
#if defined(VK_NV_cooperative_vector) && VK_NV_cooperative_vector
    case vk::ComponentTypeKHR::eSint8PackedNV:
    case vk::ComponentTypeKHR::eUint8PackedNV:
#endif
#if defined(VK_EXT_shader_float8) && VK_EXT_shader_float8
    case vk::ComponentTypeKHR::eFloat8E4M3EXT:
    case vk::ComponentTypeKHR::eFloat8E5M2EXT:
#endif
      return memory::nullopt;
    default:
      compiler::diag::unreachable();
    }
  };

  if (props.supported) {
    // Load the function pointer manually
    auto fpGetCoopMatProps =
        reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            instance.getProcAddr(
                "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));

    if (!fpGetCoopMatProps) {
      throw std::runtime_error(
          "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR not found");
    }

    // First query how many properties there are
    uint32_t count = 0;
    fpGetCoopMatProps(physicalDevice, &count, nullptr);

    std::vector<VkCooperativeMatrixPropertiesKHR> raw(count);
    for (auto &r : raw) {
      r.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
      r.pNext = nullptr;
    }

    fpGetCoopMatProps(physicalDevice, &count, raw.data());

    for (auto &p : raw) {
      fmt::println("coopmat {}", p.KSize);
      auto atype = mapCompType(static_cast<vk::ComponentTypeKHR>(p.AType));
      auto btype = mapCompType(static_cast<vk::ComponentTypeKHR>(p.BType));
      auto ctype = mapCompType(static_cast<vk::ComponentTypeKHR>(p.CType));
      auto acctype =
          mapCompType(static_cast<vk::ComponentTypeKHR>(p.ResultType));

      if (atype && btype && ctype && acctype) {
        CoopmatShape s{
            .M = p.MSize,
            .N = p.NSize,
            .K = p.KSize,
            .atype = *atype,
            .btype = *btype,
            .ctype = *ctype,
            .acctype = *acctype,
            .saturatingAccumulation =
                static_cast<bool>(p.saturatingAccumulation),
            .subgroupScope = (p.scope == VK_SCOPE_SUBGROUP_KHR),
            .workgroupScope = (p.scope == VK_SCOPE_WORKGROUP_KHR),
        };
        props.shapes.push_back(s);
      }
    }
  }
#endif
  return props;
}

} // namespace denox::compiler::device_info::query

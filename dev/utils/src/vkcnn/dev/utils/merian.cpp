#include "./merian.hpp"

#include "merian/vk/extension/extension.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/extension/extension_vk_float_atomics.hpp"
#include "merian/vk/extension/extension_vk_push_descriptor.hpp"
#include "vulkan/vulkan_core.h"

namespace vkcnn::merian {

class ExtensionCoopMat : public ::merian::Extension {
public:
  ExtensionCoopMat() : ::merian::Extension("CooperativeMatrix") {}
  ~ExtensionCoopMat() override = default;

  std::vector<const char *>
  required_device_extension_names(const vk::PhysicalDevice &) const override {
    return {"VK_KHR_cooperative_matrix"};
  }

  void *pnext_device_create_info(void *const p_next) override {
    coopMatFeat.pNext = p_next;
    coopMatFeat.cooperativeMatrix = true;
    coopMatFeat.cooperativeMatrixRobustBufferAccess = false;
    return &coopMatFeat;
  }

private:
  vk::PhysicalDeviceCooperativeMatrixFeaturesKHR coopMatFeat;
};

::merian::ContextHandle createContext(std::string_view appName) {
  // Setup logging
#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
  spdlog::set_level(spdlog::level::debug);
#endif

  // Setup Vulkan context
  const auto core =
      std::make_shared<::merian::ExtensionVkCore>(std::set<std::string>{
          "vk12/vulkanMemoryModel", "vk12/vulkanMemoryModelDeviceScope",
          "vk12/shaderBufferInt64Atomics", "vk12/shaderSubgroupExtendedTypes"});

  const auto floatAtomics =
      std::make_shared<::merian::ExtensionVkFloatAtomics>(std::set<std::string>{
          "shaderBufferFloat32Atomics",
          "shaderBufferFloat32AtomicAdd",
      });

  const auto coopMat = std::make_shared<ExtensionCoopMat>();

  const auto debug_utils =
      std::make_shared<::merian::ExtensionVkDebugUtils>(true);
  const auto resources = std::make_shared<::merian::ExtensionResources>();
  const auto push_descriptor =
      std::make_shared<::merian::ExtensionVkPushDescriptor>();

  const std::vector<std::shared_ptr<::merian::Extension>> extensions = {
      core, floatAtomics, resources, debug_utils, push_descriptor, coopMat};

  std::string appNameStr{appName};
  const ::merian::ContextHandle context = ::merian::Context::create(
      extensions, appNameStr, VK_MAKE_VERSION(1, 0, 0), 1, VK_API_VERSION_1_3,
      false);

  if (!context) {
    throw std::runtime_error("Failed to create context!!!");
  }
  return context;
}
} // namespace vkcnn::merian

#pragma once

#include "merian/vk/extension/extension.hpp"
#include "vulkan/vulkan_core.h"
class ExtensionCoopMat : public merian::Extension {
public:
  ExtensionCoopMat() : merian::Extension("CooperativeMatrix") {}
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

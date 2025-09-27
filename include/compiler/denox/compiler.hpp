#pragma once

#include <cstddef>

#ifdef DENOX_EXTERNALLY_MANAGED_VULKAN_CONTEXT
#include <vulkan/vulkan_core.h>
#endif

namespace denox {

enum class SrcType {
  Auto, // <- infer from file extention.
  Onnx, // <- force onnx frontend.
};

enum FeatureState {
  Require,
  Enable,
  Disable,
};

struct Features {
  FeatureState fusion = Enable;
  FeatureState memory_concat = Enable;
  FeatureState coopmat = Enable;
};

enum class Heuristic {
  MemoryBandwidth,
};

enum class Storage {
  StorageBuffer,
};

enum class Layout {
  HWC,
  CHWC8,
};

enum DataType {
  Float16,
};

struct BufferDescription {
  Storage storage;
  Layout layout;
  DataType dtype;
};

#ifdef DENOX_EXTERNALLY_MANAGED_VULKAN_CONTEXT
struct ExternallyManagedVulkanContext {
  VkInstance instance;
  VkPhysicalDevice physicalDevice;
};
#endif

struct CompileOptions {
  unsigned int dnxVersion = 0; // <- 0 "auto" picks stable version.
  SrcType srcType = SrcType::Auto;
  Features features;
  Heuristic heuristic = Heuristic::MemoryBandwidth;
  BufferDescription inputDescription;
  BufferDescription outputDescription;
  const char *cwd = nullptr;

  bool externally_managed_glslang_runtime = false;
#ifdef DENOX_EXTERNALLY_MANAGED_VULKAN_CONTEXT
  ExternallyManagedVulkanContext *externally_managed_vulkan_context = nullptr;
#endif
};

void compile(const char *path, const CompileOptions &options);
void compile(void *data, std::size_t n, const CompileOptions &options);

} // namespace denox

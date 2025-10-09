#pragma once

#include <cstddef>
#include <string>

#ifdef DENOX_EXTERNALLY_MANAGED_VULKAN_CONTEXT
#include <vulkan/vulkan.h>
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
  CHW,
  CHWC8,
};

enum DataType {
  Auto,
  Float16,
  Float32,
  Uint8,
  Int8,
};

struct Extent {
  const char *name;
  unsigned int value;
};

struct Shape {
  Extent height;
  Extent width;
  Extent channels;
};

struct BufferDescription {
  Shape shape;
  Storage storage;
  Layout layout;
  DataType dtype;
};

enum class VulkanApiVersion {
  Vulkan_1_0,
  Vulkan_1_1,
  Vulkan_1_2,
  Vulkan_1_3,
  Vulkan_1_4,
};

struct SpirvOptions {
  bool debugInfo = false;
  bool nonSemanticDebugInfo = false;
  bool optimize = true;
  bool skipCompilation;
};

struct Device {
  const char *deviceName; // <- allows of patterns like *AMD*
  VulkanApiVersion apiVersion = VulkanApiVersion::Vulkan_1_3;
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
  Device device;
  SpirvOptions spirvOptions;
  const char *cwd = nullptr;
  bool verbose;
  bool quite;
  bool summarize;
  bool externally_managed_glslang_runtime = false;
#ifdef DENOX_EXTERNALLY_MANAGED_VULKAN_CONTEXT
  ExternallyManagedVulkanContext *externally_managed_vulkan_context = nullptr;
#endif
};

struct CompilationResult {
  const void *dnx;
  size_t dnxSize;
  const char *message;
};

int compile(const char *path, const CompileOptions *options,
            CompilationResult *result);

void destroy_compilation_result(CompilationResult* result); 

} // namespace denox

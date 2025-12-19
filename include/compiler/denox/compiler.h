#ifndef DENOX_COMPILER_H
#define DENOX_COMPILER_H

#include <cstdint>

typedef enum DenoxResult { DENOX_SUCCESS = 0, DENOX_FAILURE = 1 } DenoxResult;

typedef enum DenoxDnxVersion {
  DENOX_DNX_VERSION_0_0,
} DenoxDnxVersion;

typedef enum DenoxDataType {
  DENOX_DATA_TYPE_AUTO,
  DENOX_DATA_TYPE_F16,
  DENOX_DATA_TYPE_F32,
  DENOX_DATA_TYPE_F64,
  DENOX_DATA_TYPE_U8,
  DENOX_DATA_TYPE_I8,
  DENOX_DATA_TYPE_U16,
  DENOX_DATA_TYPE_I16,
  DENOX_DATA_TYPE_U32,
  DENOX_DATA_TYPE_I32,
} DenoxDataType;

typedef enum DenoxDeviceTargetEnv {
  DENOX_TARGET_ENV_VULKAN_1_0,
  DENOX_TARGET_ENV_VULKAN_1_1,
  DENOX_TARGET_ENV_VULKAN_1_2,
  DENOX_TARGET_ENV_VULKAN_1_3,
  DENOX_TARGET_ENV_VULKAN_1_4,
} DenoxDeviceTargetEnv;

/// DenoxDeviceResourceLimits can be directly queries from
/// a VkPhysicalDeviceLimits struct.
typedef struct DenoxDeviceResourceLimits {
  uint32_t maxComputeWorkGroupCount[3];
  uint32_t maxComputeWorkGroupSize[3];
  uint32_t maxComputeWorkGroupInvocations;
  uint32_t maxComputeSharedMemory;

  uint32_t maxPerStageResources;
  uint32_t maxPerStageUniformBuffers;
  uint32_t maxPerStageStorageBuffers;
  uint32_t maxPerStageSampledImages;
  uint32_t maxPerStageStorageImages;
  uint32_t maxPerStageSamplers;

  uint32_t maxPushConstantSize;
} DenoxDeviceResourceLimits;

/// DenoxDeviceResourceLimits can be directly queries from
/// a VkPhysicalDeviceSubgroupProperties struct.
typedef enum DenoxDeviceSubgroupFeatureFlagBits {
  DENOX_DEVICE_SUBGROUP_FEATURE_BASIC_BIT = 1,
  DENOX_DEVICE_SUBGROUP_FEATURE_VOTE_BIT = 2,
  DENOX_DEVICE_SUBGROUP_FEATURE_ARITHMETIC_BIT = 4,
  DENOX_DEVICE_SUBGROUP_FEATURE_BALLOT_BIT = 8,
  DENOX_DEVICE_SUBGROUP_FEATURE_SHUFFLE_BIT = 16,
  DENOX_DEVICE_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT = 32,
} DenoxDeviceSubgroupFeatureFlagBits;

typedef uint32_t DenoxSubgroupFeatureFlags;

typedef struct DenoxDeviceSubgroupProperties {
  /// The subgroup size has to be constant across all dispatches,
  /// for machines with dynamic subgroup sizes, the runtime much guarnatee
  /// that the subgroup size is constant across all denox dispatches.
  uint32_t subgroupSize;
  /// All subgroup operations specified here, may be used by denox.
  /// The runtime has to guarantee that those features are available
  /// and enabled on the logical vulkan device.
  DenoxSubgroupFeatureFlags subgroupOperations;
} DenoxDeviceSubgroupProperties;

typedef uint32_t DenoxDeviceLayoutFeatureFlags;
typedef uint32_t DenoxBool32;

typedef enum DenoxDeviceScopeBits {
  DENOX_DEVICE_SCOPE_SUBGROUP_SCOPE_BIT = 1,
} DenoxDeviceScopeBits;

typedef uint32_t DenoxDeviceScope;

typedef struct DenoxDeviceCooperativeMatrixShape {
  //    A    *    B    =   C
  // (m x k) * (k x n) = (m x n)
  uint32_t mSize;
  uint32_t kSize;
  uint32_t nSize;
  DenoxDataType aType;
  DenoxDataType bType;
  DenoxDataType cType;
  DenoxDataType accType;
  DenoxBool32 saturatingAccumulation;
  DenoxDeviceScope scope;
} DenoxDeviceCooperativeMatrixShape;

typedef struct DenoxDeviceInfo {
  DenoxDeviceTargetEnv targetEnv;
  const char* deviceName;
  DenoxDeviceResourceLimits limits;
  DenoxDeviceSubgroupProperties subgroupProperties;

  /// If not null, denox expects all listed cooperative matrix shapes to be
  /// supported and enabled on the logical device.
  ///
  /// List of all supported cooperative matrix shapes.
  /// Denox will assume that this list is sorted by fastest
  /// shape to slowest shape.
  uint32_t coopmatShapeCount;
  DenoxDeviceCooperativeMatrixShape *pCoopmatShapes;
} DenoxDeviceInfo;

typedef struct DenoxSpirvOptions {
  // ============== Compiler-Flags ================

  /// add debug info to generated SPIR-V shaders.
  DenoxBool32 debugInfo;
  /// Add non semantic debug info to generated SPIR-V shaders.
  /// This might be interessting for profiling with third party tools,
  /// like NVIDIA NSight.
  DenoxBool32 nonSemanticDebugInfo;
  /// optimize all spirv-shaders (spirv-opt).
  DenoxBool32 optimize;
  /// validate all spirv-shaders (spirv-val).
  DenoxBool32 validate;

  // ============= Descriptor policies ============
  // allows specifing specific descriptor set to use for certain tasks,
  // this can be interessting for renderers which have set=0, globally
  // bound and the generated SPIR-V would collide with this.
  // For more complex things we advise using
  // https://github.com/kistenklaus/spirv-rebind to rebind descriptors at
  // runtime or during code generation.

  /// selects descriptor set, which is used for network inputs.
  uint32_t input_descriptor_set;
  /// selects descriptor set, which is used for network outputs.
  uint32_t output_descriptor_set;
  /// selects descriptor set, which is used for internal
  /// readonly descriptors.
  uint32_t readonly_descriptor_set;
  /// selects descriptor set, which is used for internal
  /// writeonly descriptors.
  uint32_t writeonly_descriptor_set;
  /// selects descriptor set, which is used for parameters
  /// (e.g. convolution weights or bias).
  uint32_t param_descriptor_set;
} DenoxSpirvOptions;

typedef struct DenoxOptimizationOptions {
  DenoxBool32 disableLayerFusion;
  DenoxBool32 disableMemoryConcat;
} DenoxOptimizationOptions;

typedef enum DenoxTensorStorage {
  DENOX_TENSOR_STORAGE_OPTIMAL,                // glsl: denox decides itself.
  DENOX_TENSOR_STORAGE_STORAGE_BUFFER,         // glsl: buffer
  DENOX_TENSOR_STORAGE_STORAGE_IMAGE,          // glsl: image2D
  DENOX_TENSOR_STORAGE_SAMPLER,                // glsl: sampler
  DENOX_TENSOR_STORAGE_COMBINED_IMAGE_SAMPLER, // glsl: sampler2D
  DENOX_TENSOR_STORAGE_SAMPLED_IMAGE,          // glsl: texture2D
} DenoxTensorStorage;

typedef enum DenoxTensorLayout {
  DENOX_TENSOR_LAYOUT_OPTIMAL, // denox will pick the best layout
  DENOX_TENSOR_LAYOUT_HWC,
  DENOX_TENSOR_LAYOUT_CHW,
  DENOX_TENSOR_LAYOUT_CHWC8,    // channel count must be a multiple of 8.
  DENOX_TENSOR_LAYOUT_TEX_RGBA, // pad upto rgba
  DENOX_TENSOR_LAYOUT_TEX_RGB,  // pad upto rgb
  DENOX_TENSOR_LAYOUT_TEX_RG,   // pad upto rg
  DENOX_TENSOR_LAYOUT_TEX_R,    // pad upto r
} DenoxTensorLayout;

typedef struct DenoxTensorShape {
  const char *widthValueName;
  uint64_t width;
  const char *heightValueName;
  uint64_t height;
  const char *channelValueName;
  uint64_t channels;
} DenoxTensorShape;

typedef struct DenoxIODescriptor {
  const char *tensorName;
  DenoxTensorStorage storage;
  DenoxTensorLayout layout;
  DenoxTensorShape shape;
  DenoxDataType dataType;
} DenoxIODescriptor;

typedef struct {
  DenoxDnxVersion version;
  DenoxDeviceInfo *pDeviceInfo;
  DenoxSpirvOptions *pSpvOptions;
  DenoxOptimizationOptions *pOptimizationOptions;
  uint32_t ioDescriptorCount;
  DenoxIODescriptor *pIODescriptors;
} DenoxCompileOptions;

typedef void *DenoxDNX;

typedef void *DenoxONNX;

typedef void *DenoxDB;

DenoxResult denox_query_device_info(const char *device_name,
                                    DenoxDeviceInfo *pDeviceInfo);
void denox_destroy_device_info(DenoxDeviceInfo *pDeviceInfo);

DenoxResult denox_open_database(const char *path, DenoxDB *db);
DenoxResult denox_close_database(DenoxDB db);

DenoxResult denox_compile_onnx(DenoxONNX *pOnnx, DenoxCompileOptions *pOptions,
                               DenoxDB db, DenoxDNX *pDNX);
void denox_get_dnx_blob(DenoxDNX dnx, uint64_t *pCount, void *blob);
void denox_destroy_dnx(DenoxDNX dnx);

#endif

#include "api_helpers.hpp"
#include "denox/compiler.h"
#include "device_info/ApiVersion.hpp"
#include "device_info/query/query_driver_device_info.hpp"

DenoxResult denox_query_device_info(const char *device_name,
                                    DenoxDeviceInfo *pDeviceInfo) {
  assert(pDeviceInfo != nullptr);
  denox::compiler::DeviceInfo deviceInfo;
  denox::compiler::ApiVersion optimalTargetEnv =
      denox::compiler::ApiVersion::VULKAN_1_4;
  try {
    if (device_name == nullptr) {
      deviceInfo = denox::compiler::device_info::query_driver_device_info(
          optimalTargetEnv, std::nullopt);
    } else {
      deviceInfo = denox::compiler::device_info::query_driver_device_info(
          optimalTargetEnv, device_name);
    }
  } catch (const std::exception &) {
    return DENOX_FAILURE;
  }

  switch (deviceInfo.apiVersion) {
  case denox::compiler::ApiVersion::VULKAN_1_0:
    pDeviceInfo->targetEnv = DENOX_TARGET_ENV_VULKAN_1_0;
    break;
  case denox::compiler::ApiVersion::VULKAN_1_1:
    pDeviceInfo->targetEnv = DENOX_TARGET_ENV_VULKAN_1_1;
    break;
  case denox::compiler::ApiVersion::VULKAN_1_2:
    pDeviceInfo->targetEnv = DENOX_TARGET_ENV_VULKAN_1_2;
    break;
  case denox::compiler::ApiVersion::VULKAN_1_3:
    pDeviceInfo->targetEnv = DENOX_TARGET_ENV_VULKAN_1_3;
    break;
  case denox::compiler::ApiVersion::VULKAN_1_4:
    pDeviceInfo->targetEnv = DENOX_TARGET_ENV_VULKAN_1_4;
    break;
  }

  std::memcpy(pDeviceInfo->limits.maxComputeWorkGroupCount,
              deviceInfo.limits.maxComputeWorkGroupCount.data(),
              sizeof(uint32_t) * 3);

  std::memcpy(pDeviceInfo->limits.maxComputeWorkGroupSize,
              deviceInfo.limits.maxComputeWorkGroupSize.data(),
              sizeof(uint32_t) * 3);

  pDeviceInfo->limits.maxComputeWorkGroupInvocations =
      deviceInfo.limits.maxComputeWorkGroupInvocations;
  pDeviceInfo->limits.maxComputeSharedMemory =
      deviceInfo.limits.maxComputeSharedMemory;
  pDeviceInfo->limits.maxPerStageResources =
      deviceInfo.limits.maxPerStageResources;
  pDeviceInfo->limits.maxPerStageUniformBuffers =
      deviceInfo.limits.maxPerStageUniformBuffers;
  pDeviceInfo->limits.maxPerStageStorageBuffers =
      deviceInfo.limits.maxPerStageStorageBuffers;
  pDeviceInfo->limits.maxPerStageSampledImages =
      deviceInfo.limits.maxPerStageSampledImages;
  pDeviceInfo->limits.maxPerStageStorageImages =
      deviceInfo.limits.maxPerStageStorageImages;
  pDeviceInfo->limits.maxPerStageSamplers =
      deviceInfo.limits.maxPerStageSamplers;
  pDeviceInfo->limits.maxPushConstantSize =
      deviceInfo.limits.maxPushConstantSize;

  pDeviceInfo->subgroupProperties.subgroupSize =
      deviceInfo.subgroup.subgroupSize;
  pDeviceInfo->subgroupProperties.subgroupOperations = 0;
  if (deviceInfo.subgroup.supportsBasicOps) {
    pDeviceInfo->subgroupProperties.subgroupOperations |=
        DENOX_DEVICE_SUBGROUP_FEATURE_BASIC_BIT;
  }
  if (deviceInfo.subgroup.supportsVoteOps) {
    pDeviceInfo->subgroupProperties.subgroupOperations |=
        DENOX_DEVICE_SUBGROUP_FEATURE_VOTE_BIT;
  }
  if (deviceInfo.subgroup.supportsArithmeticOps) {
    pDeviceInfo->subgroupProperties.subgroupOperations |=
        DENOX_DEVICE_SUBGROUP_FEATURE_ARITHMETIC_BIT;
  }
  if (deviceInfo.subgroup.supportsBallotOps) {
    pDeviceInfo->subgroupProperties.subgroupOperations |=
        DENOX_DEVICE_SUBGROUP_FEATURE_BALLOT_BIT;
  }
  if (deviceInfo.subgroup.supportsShuffleOps) {
    pDeviceInfo->subgroupProperties.subgroupOperations |=
        DENOX_DEVICE_SUBGROUP_FEATURE_SHUFFLE_BIT;
  }
  if (deviceInfo.subgroup.supportsShuffleRelativeOps) {
    pDeviceInfo->subgroupProperties.subgroupOperations |=
        DENOX_DEVICE_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
  }
  if (deviceInfo.coopmat.supported) {
    pDeviceInfo->coopmatShapeCount =
        static_cast<uint32_t>(deviceInfo.coopmat.shapes.size());
    pDeviceInfo->pCoopmatShapes =
        static_cast<DenoxDeviceCooperativeMatrixShape *>(
            malloc(sizeof(DenoxDeviceCooperativeMatrixShape) *
                   deviceInfo.coopmat.shapes.size()));

    for (size_t i = 0; i < deviceInfo.coopmat.shapes.size(); ++i) {
      const auto &shape = deviceInfo.coopmat.shapes[i];
      pDeviceInfo->pCoopmatShapes[i].mSize = shape.M;

      pDeviceInfo->pCoopmatShapes[i].kSize = shape.K;
      pDeviceInfo->pCoopmatShapes[i].nSize = shape.N;
      pDeviceInfo->pCoopmatShapes[i].aType =
          denox::api::Dtype_to_DenoxDataType(shape.atype);
      pDeviceInfo->pCoopmatShapes[i].bType =
          denox::api::Dtype_to_DenoxDataType(shape.btype);
      pDeviceInfo->pCoopmatShapes[i].cType =
          denox::api::Dtype_to_DenoxDataType(shape.ctype);
      pDeviceInfo->pCoopmatShapes[i].accType =
          denox::api::Dtype_to_DenoxDataType(shape.acctype);
      pDeviceInfo->pCoopmatShapes[i].saturatingAccumulation =
          shape.saturatingAccumulation ? 1 : 0;
      pDeviceInfo->pCoopmatShapes[i].scope = 0;
      if (shape.subgroupScope) {
        pDeviceInfo->pCoopmatShapes[i].scope |=
            DENOX_DEVICE_SCOPE_SUBGROUP_SCOPE_BIT;
      }
    }
  } else {
    pDeviceInfo->coopmatShapeCount = 0;
    pDeviceInfo->pCoopmatShapes = nullptr;
  }

  if (deviceInfo.name.empty()) {
    pDeviceInfo->deviceName = nullptr;
  } else {
    char *name = static_cast<char *>(malloc(deviceInfo.name.size() + 1));
    std::memset(name, 0, deviceInfo.name.size() + 1);
    std::memcpy(name, deviceInfo.name.data(), deviceInfo.name.size());
    pDeviceInfo->deviceName = name;
  }

  return DENOX_SUCCESS;
}

void denox_destroy_device_info(DenoxDeviceInfo *pDeviceInfo) {
  free(pDeviceInfo->pCoopmatShapes);
  pDeviceInfo->coopmatShapeCount = 0;
  pDeviceInfo->pCoopmatShapes = nullptr;

  free(const_cast<char *>(pDeviceInfo->deviceName));
  pDeviceInfo->deviceName = nullptr;
}

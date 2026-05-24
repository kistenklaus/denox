#include "denox/cli/merge_device_info.hpp"
#include "absl/strings/str_format.h"
#include "denox/cli/device_yml/device_yml.hpp"
#include "denox/cli/io/InputStream.hpp"
#include "denox/cli/io/OutputStream.hpp"
#include "denox/device_info/CoopmatProperties.hpp"
#include "denox/device_info/DeviceInfo.hpp"
#include "denox/memory/container/vector.hpp"
#include <algorithm>

static denox::DeviceInfo min_device_info(const denox::DeviceInfo &lhs,
                                         const denox::DeviceInfo &rhs) {
  denox::DeviceInfo out{};

  out.apiVersion = static_cast<denox::ApiVersion>(
      std::min(static_cast<uint32_t>(lhs.apiVersion),
               static_cast<uint32_t>(rhs.apiVersion)));

  out.name = fmt::format("{}|{}", lhs.name, rhs.name);

  for (std::size_t i = 0; i < 3; ++i) {
    out.limits.maxComputeWorkGroupCount[i] =
        std::min(lhs.limits.maxComputeWorkGroupCount[i],
                 rhs.limits.maxComputeWorkGroupCount[i]);

    out.limits.maxComputeWorkGroupSize[i] =
        std::min(lhs.limits.maxComputeWorkGroupSize[i],
                 rhs.limits.maxComputeWorkGroupSize[i]);
  }

  out.limits.maxComputeWorkGroupInvocations =
      std::min(lhs.limits.maxComputeWorkGroupInvocations,
               rhs.limits.maxComputeWorkGroupInvocations);

  out.limits.maxComputeSharedMemory = std::min(
      lhs.limits.maxComputeSharedMemory, rhs.limits.maxComputeSharedMemory);

  out.limits.maxPushConstantSize =
      std::min(lhs.limits.maxPushConstantSize, rhs.limits.maxPushConstantSize);

  const auto rhs_supports_subgroup_size = [&](uint32_t size) -> bool {
    if (rhs.subgroup.controlProperties.supported) {
      return std::ranges::find(
                 rhs.subgroup.controlProperties.supportedSubgroupSizes, size) !=
             rhs.subgroup.controlProperties.supportedSubgroupSizes.end();
    }

    return rhs.subgroup.subgroupSize == size;
  };

  auto add_common_subgroup_size = [&](uint32_t size) {
    if (!rhs_supports_subgroup_size(size)) {
      return;
    }

    const bool alreadyAdded =
        std::ranges::find(out.subgroup.controlProperties.supportedSubgroupSizes,
                          size) !=
        out.subgroup.controlProperties.supportedSubgroupSizes.end();

    if (!alreadyAdded) {
      out.subgroup.controlProperties.supportedSubgroupSizes.emplace_back(size);
    }
  };

  if (lhs.subgroup.controlProperties.supported) {
    for (uint32_t size :
         lhs.subgroup.controlProperties.supportedSubgroupSizes) {
      add_common_subgroup_size(size);
    }
  } else {
    add_common_subgroup_size(lhs.subgroup.subgroupSize);
  }

  if (out.subgroup.controlProperties.supportedSubgroupSizes.empty()) {
    throw std::runtime_error(
        "Failed to merge device infos, incompatible subgroup size support");
  }

  // Arbitrary, but choose an actually common size.
  out.subgroup.subgroupSize =
      out.subgroup.controlProperties.supportedSubgroupSizes.front();

  if (lhs.subgroup.controlProperties.supported &&
      rhs.subgroup.controlProperties.supported) {
    out.subgroup.controlProperties.supported = true;

    out.subgroup.controlProperties.maxComputeWorkgroupSubgroups =
        std::min(lhs.subgroup.controlProperties.maxComputeWorkgroupSubgroups,
                 rhs.subgroup.controlProperties.maxComputeWorkgroupSubgroups);
  } else {
    // The merged device info must not claim subgroup-size control unless both
    // devices support it. In this case, the common capability is just the fixed
    // subgroup size chosen above.
    out.subgroup.controlProperties.supported = false;
    out.subgroup.controlProperties.supportedSubgroupSizes.clear();
  }

  out.subgroup.supportsBasicOps =
      lhs.subgroup.supportsBasicOps && rhs.subgroup.supportsBasicOps;

  out.subgroup.supportsVoteOps =
      lhs.subgroup.supportsVoteOps && rhs.subgroup.supportsVoteOps;

  out.subgroup.supportsArithmeticOps =
      lhs.subgroup.supportsArithmeticOps && rhs.subgroup.supportsArithmeticOps;

  out.subgroup.supportsBallotOps =
      lhs.subgroup.supportsBallotOps && rhs.subgroup.supportsBallotOps;

  out.subgroup.supportsShuffleOps =
      lhs.subgroup.supportsShuffleOps && rhs.subgroup.supportsShuffleOps;

  out.subgroup.supportsShuffleRelativeOps =
      lhs.subgroup.supportsShuffleRelativeOps &&
      rhs.subgroup.supportsShuffleRelativeOps;

  out.memoryModel.vmm = lhs.memoryModel.vmm && rhs.memoryModel.vmm;

  out.memoryModel.vmmDeviceScope =
      lhs.memoryModel.vmmDeviceScope && rhs.memoryModel.vmmDeviceScope;

  const auto same_coopmat_shape = [](const denox::CoopmatShape &a,
                                     const denox::CoopmatShape &b) -> bool {
    return a.N == b.N && a.K == b.K && a.M == b.M && a.atype == b.atype &&
           a.btype == b.btype && a.ctype == b.ctype && a.acctype == b.acctype &&
           a.saturatingAccumulation == b.saturatingAccumulation &&
           a.subgroupScope == b.subgroupScope;
  };

  if (lhs.coopmat.supported && rhs.coopmat.supported) {
    for (const denox::CoopmatShape &lhsShape : lhs.coopmat.shapes) {
      const bool rhsSupportsShape =
          std::ranges::find_if(rhs.coopmat.shapes,
                               [&](const denox::CoopmatShape &rhsShape) {
                                 return same_coopmat_shape(lhsShape, rhsShape);
                               }) != rhs.coopmat.shapes.end();

      const bool alreadyAdded =
          std::ranges::find_if(out.coopmat.shapes,
                               [&](const denox::CoopmatShape &outShape) {
                                 return same_coopmat_shape(lhsShape, outShape);
                               }) != out.coopmat.shapes.end();

      if (rhsSupportsShape && !alreadyAdded) {
        out.coopmat.shapes.emplace_back(lhsShape);
      }
    }
  }

  out.coopmat.supported = !out.coopmat.shapes.empty();

  return out;
}

void merge_device_info(MergeDeviceInfo &action) {
  assert(action.device_infos.size() >= 2);

  static constexpr size_t MAX_YAML_FILE_SIZE = 1 << 20; // 1MB
  denox::memory::vector<std::byte> yamlCache(MAX_YAML_FILE_SIZE);

  denox::memory::vector<denox::DeviceInfo> deviceInfos;
  deviceInfos.resize(action.device_infos.size());
  for (size_t i = 0; i < deviceInfos.size(); ++i) {
    InputStream istream{action.device_infos[i]};
    size_t sz = istream.read(yamlCache);
    denox::memory::span yaml{yamlCache.begin(), sz};
    deviceInfos[i] = deserialize_device_yml(yaml);
  }

  denox::DeviceInfo min = deviceInfos[0];
  for (uint32_t i = 1; i < deviceInfos.size(); ++i) {
    min = min_device_info(min, deviceInfos[i]);
  }

  denox::memory::vector<std::byte> minYaml = serialize_device_yml(min);
  OutputStream ostream{action.output};
  ostream.write_exact(minYaml);
}

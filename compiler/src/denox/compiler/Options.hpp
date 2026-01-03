#pragma once

#include "denox/common/TensorDataType.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/common/TensorStorage.hpp"
#include "denox/device_info/DeviceInfo.hpp"
#include "denox/diag/logging.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/spirv/ShaderDebugInfoLevel.hpp"

namespace denox::compiler {

struct Features {
  bool coopmat = true;
  bool enableImplicitConcat = true;
  bool enableConvReluFusion = true;
};

struct DescriptorPolicy {
  // sets should be selected in order.
  uint32_t set;
};

struct DescriptorPolicies {
  // policy is selected in order, i.e. first check input then output and so on.
  DescriptorPolicy inputPolicy;
  DescriptorPolicy outputPolicy;
  DescriptorPolicy paramPolicy;
  DescriptorPolicy readPolicy;
  DescriptorPolicy writePolicy;
};

struct InterfaceTensorDescriptor {
  std::string name;
  TensorFormat format;
  TensorStorage storage;
  TensorDataType dtype;
  std::optional<std::string> heightValueName;
  std::optional<uint32_t> height;
  std::optional<std::string> widthValueName;
  std::optional<uint32_t> width;
  std::optional<std::string> channelValueName;
  std::optional<uint32_t> channels;
};

struct SpirvOptions {
  spirv::SpirvDebugInfoLevel debugInfo;
  bool optimize;
};

struct OptimizationAssumption {
  std::string valueName;
  uint64_t value;
};

struct OptimizationAssumptions {
  std::vector<OptimizationAssumption> valueAssumptions;
};

enum class DebugInfo {
  Strip,
  Enable,
};

struct CompileOptions {
  unsigned int dnxVersion;
  Features features;
  SpirvOptions spirv;
  DeviceInfo deviceInfo;
  std::vector<InterfaceTensorDescriptor> interfaceDescriptors;
  DescriptorPolicies descriptorPolicies;
  diag::LogLevel loglevel;
  OptimizationAssumptions assumptions;
  DebugInfo debugInfo;
};

} // namespace denox::compiler

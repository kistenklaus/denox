#pragma once

#include "denox/common/TensorDataType.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/common/TensorStorage.hpp"
#include "denox/common/ValueSpec.hpp"
#include "denox/device_info/DeviceInfo.hpp"
#include "denox/diag/logging.hpp"
#include "denox/spirv/ShaderDebugInfoLevel.hpp"

namespace denox::compiler {

struct Features {
  bool coopmat = true;
  bool enableImplicitConcat = true;
  bool enableConvReluFusion = true;
  bool enableConcatConvFusion = true;
};

struct DescriptorPolicy {
  // sets should be selected in order.
  uint32_t set = 0;
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
  TensorFormat format = TensorFormat::Optimal;
  TensorStorage storage = TensorStorage::Optimal;
  TensorDataType dtype = TensorDataType::Auto;
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

struct OptimizationAssumptions {
  std::vector<ValueSpec> valueAssumptions;
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
  uint32_t optimizationLevel;
};

} // namespace denox::compiler

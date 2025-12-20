#pragma once

#include "denox/common/TensorDataType.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/common/TensorStorage.hpp"
#include "denox/device_info/DeviceInfo.hpp"
#include "denox/diag/logging.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/spirv/ShaderDebugInfoLevel.hpp"

namespace denox::compiler {

struct FusionRules {
  bool enableSliceSliceFusion = true;
  bool enableImplicitConcat = true;
  bool enableConvReluFusion = true;
};

struct TensorShapeExtent {
  memory::optional<std::string> name;
  memory::optional<unsigned int> value;
};

struct TensorShapeDesc {
  TensorShapeExtent channels;
  TensorShapeExtent height;
  TensorShapeExtent width;
};

struct Features {
  bool coopmat = true;
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

struct TensorDescriptor {
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


struct Options {
  unsigned int dnxVersion;
  Features features;
  FusionRules fusion;
  SpirvOptions spirv;
  DeviceInfo deviceInfo;
  std::vector<TensorDescriptor> interfaceDescriptors;
  DescriptorPolicies descriptorPolicies;
  diag::LogLevel loglevel;
};

} // namespace denox::compiler

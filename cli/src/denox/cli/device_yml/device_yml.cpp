#include "denox/cli/device_yml/device_yml.hpp"
#include "denox/device_info/DeviceInfo.hpp"
#include "denox/device_info/MemoryModel.hpp"
#include "denox/device_info/ResourceLimits.hpp"
#include "denox/memory/container/span.hpp"
#include <yaml-cpp/yaml.h>
#include <stdexcept>

denox::memory::vector<std::byte>
serialize_device_yml(const denox::DeviceInfo &deviceInfo) {
  YAML::Emitter out;
  { // serialize device-info
    out << YAML::BeginMap;
    {
      out << YAML::Key << "api-version";
      out << YAML::Value;
      switch (deviceInfo.apiVersion) {
      case denox::ApiVersion::VULKAN_1_0:
        out << "VULKAN_1_0";
        break;
      case denox::ApiVersion::VULKAN_1_1:
        out << "VULKAN_1_1";
        break;
      case denox::ApiVersion::VULKAN_1_2:
        out << "VULKAN_1_2";
        break;
      case denox::ApiVersion::VULKAN_1_3:
        out << "VULKAN_1_3";
        break;
      case denox::ApiVersion::VULKAN_1_4:
        out << "VULKAN_1_4";
        break;
      }
    }
    {
      out << YAML::Key << "device-name";
      out << YAML::Value << deviceInfo.name;
    }
    {
      out << YAML::Key << "limits";
      out << YAML::Value;
      out << YAML::BeginMap;
      {
        out << YAML::Key << "max-compute-workgroup-count-x";
        out << YAML::Value << deviceInfo.limits.maxComputeWorkGroupCount[0];
      }
      {
        out << YAML::Key << "max-compute-workgroup-count-y";
        out << YAML::Value << deviceInfo.limits.maxComputeWorkGroupCount[1];
      }
      {
        out << YAML::Key << "max-compute-workgroup-count-z";
        out << YAML::Value << deviceInfo.limits.maxComputeWorkGroupCount[2];
      }
      {
        out << YAML::Key << "max-compute-workgroup-size-x";
        out << YAML::Value << deviceInfo.limits.maxComputeWorkGroupSize[0];
      }
      {
        out << YAML::Key << "max-compute-workgroup-size-y";
        out << YAML::Value << deviceInfo.limits.maxComputeWorkGroupSize[1];
      }
      {
        out << YAML::Key << "max-compute-workgroup-size-z";
        out << YAML::Value << deviceInfo.limits.maxComputeWorkGroupSize[2];
      }
      {
        out << YAML::Key << "max-compute-workgroup-invocations";
        out << YAML::Value << deviceInfo.limits.maxComputeWorkGroupInvocations;
      }
      {
        out << YAML::Key << "max-compute-shared-memory";
        out << YAML::Value << deviceInfo.limits.maxComputeSharedMemory;
      }
      {
        out << YAML::Key << "max-push-constant-size";
        out << YAML::Value << deviceInfo.limits.maxPushConstantSize;
      }
      out << YAML::EndMap;
    }
    {
      out << YAML::Key << "subgroup-properties";
      out << YAML::Value << YAML::BeginMap;
      {
        out << YAML::Key << "subgroup-size";
        out << YAML::Value << deviceInfo.subgroup.subgroupSize;
      }
      {
        out << YAML::Key << "supports-basic-ops";
        out << YAML::Value << deviceInfo.subgroup.supportsBasicOps;
      }
      {
        out << YAML::Key << "supports-vote-ops";
        out << YAML::Value << deviceInfo.subgroup.supportsVoteOps;
      }
      {
        out << YAML::Key << "supports-arithmetic-ops";
        out << YAML::Value << deviceInfo.subgroup.supportsArithmeticOps;
      }
      {
        out << YAML::Key << "supports-ballot-ops";
        out << YAML::Value << deviceInfo.subgroup.supportsBallotOps;
      }
      {
        out << YAML::Key << "supports-shuffle-ops";
        out << YAML::Value << deviceInfo.subgroup.supportsShuffleOps;
      }
      {
        out << YAML::Key << "supports-shuffle-relative-ops";
        out << YAML::Value << deviceInfo.subgroup.supportsShuffleRelativeOps;
      }
      {
        out << YAML::Key << "subgroup-control";
        if (deviceInfo.subgroup.controlProperties.supported) {
          out << YAML::Value << YAML::BeginMap;
          {
            out << YAML::Key << "max-compute-workgroup-subgroups";
            out << YAML::Value
                << deviceInfo.subgroup.controlProperties
                       .maxComputeWorkgroupSubgroups;
          }
          {
            out << YAML::Key << "supported-subgroup-sizes";
            out << YAML::Value << YAML::Flow << YAML::BeginSeq;
            for (const uint32_t sg_size :
                 deviceInfo.subgroup.controlProperties.supportedSubgroupSizes) {
              out << sg_size;
            }
            out << YAML::EndSeq;
          }
          out << YAML::EndMap;
        } else {
          out << YAML::Value << YAML::Null;
        }
      }

      out << YAML::EndMap;
    }
    {
      out << YAML::Key << "memory-model";
      out << YAML::Value << YAML::BeginMap;
      {
        out << YAML::Key << "vulkan-memory-model";
        out << YAML::Value << deviceInfo.memoryModel.vmm;
      }
      {
        out << YAML::Key << "vulkan-memory-model-device-scope";
        out << YAML::Value << deviceInfo.memoryModel.vmmDeviceScope;
      }
      out << YAML::EndMap;
    }
    {
      out << YAML::Key << "coopmat-shapes";
      if (deviceInfo.coopmat.supported) {
        out << YAML::Value << YAML::BeginSeq;
        for (const denox::CoopmatShape &shape : deviceInfo.coopmat.shapes) {
          out << YAML::BeginMap;
          {
            out << YAML::Key << "N";
            out << YAML::Value << shape.N;
          }
          {
            out << YAML::Key << "K";
            out << YAML::Value << shape.K;
          }
          {
            out << YAML::Key << "M";
            out << YAML::Value << shape.M;
          }
          {
            out << YAML::Key << "atype";
            out << YAML::Value << fmt::format("{}", shape.atype);
          }
          {
            out << YAML::Key << "btype";
            out << YAML::Value << fmt::format("{}", shape.btype);
          }
          {
            out << YAML::Key << "ctype";
            out << YAML::Value << fmt::format("{}", shape.ctype);
          }
          {
            out << YAML::Key << "acctype";
            out << YAML::Value << fmt::format("{}", shape.acctype);
          }
          {
            out << YAML::Key << "saturating-accumulation";
            out << YAML::Value << shape.saturatingAccumulation;
          }
          {
            out << YAML::Key << "subgroup-scope";
            out << YAML::Value << shape.subgroupScope;
          }
          out << YAML::EndMap;
        }

        out << YAML::EndSeq;
      } else {
        out << YAML::Value << YAML::Flow << YAML::BeginSeq << YAML::EndSeq;
      }
    }

    out << YAML::EndMap;
  }
  static_assert(sizeof(std::byte) == sizeof(char));
  denox::memory::span<const std::byte> span{
      reinterpret_cast<const std::byte *>(out.c_str()),
      out.size(),
  };
  return denox::memory::vector<std::byte>{span.begin(), span.end()};
}

static denox::ApiVersion deserialize_api_version(const YAML::Node &node) {
  const std::string &str = node.as<std::string>();
  if (str == "VULKAN_1_4") {
    return denox::ApiVersion::VULKAN_1_4;
  } else if (str == "VULKAN_1_3") {
    return denox::ApiVersion::VULKAN_1_3;
  } else if (str == "VULKAN_1_2") {
    return denox::ApiVersion::VULKAN_1_2;
  } else if (str == "VULKAN_1_1") {
    return denox::ApiVersion::VULKAN_1_1;
  } else if (str == "VULKAN_1_0") {
    return denox::ApiVersion::VULKAN_1_0;
  } else {
    throw std::runtime_error(fmt::format("Invalid api-version: \"{}\"", str));
  }
}

static std::string deserialize_device_name(const YAML::Node &node) {
  return node.as<std::string>();
}

static denox::ResourceLimits
deserialize_resource_limits(const YAML::Node &node) {
  denox::ResourceLimits limits;
  limits.maxComputeWorkGroupCount[0] =
      node["max-compute-workgroup-count-x"].as<uint32_t>();
  limits.maxComputeWorkGroupCount[1] =
      node["max-compute-workgroup-count-y"].as<uint32_t>();
  limits.maxComputeWorkGroupCount[2] =
      node["max-compute-workgroup-count-z"].as<uint32_t>();
  limits.maxComputeWorkGroupSize[0] =
      node["max-compute-workgroup-size-x"].as<uint32_t>();
  limits.maxComputeWorkGroupSize[1] =
      node["max-compute-workgroup-size-y"].as<uint32_t>();
  limits.maxComputeWorkGroupSize[2] =
      node["max-compute-workgroup-size-z"].as<uint32_t>();
  limits.maxComputeWorkGroupInvocations =
      node["max-compute-workgroup-invocations"].as<uint32_t>();
  limits.maxComputeSharedMemory =
      node["max-compute-shared-memory"].as<uint32_t>();
  limits.maxPushConstantSize = node["max-push-constant-size"].as<uint32_t>();
  return limits;
}

denox::SubgroupProperties
deserialize_subgroup_properties(const YAML::Node &node) {
  denox::SubgroupProperties props;
  props.subgroupSize = node["subgroup-size"].as<uint32_t>();
  props.supportsBasicOps = node["supports-basic-ops"].as<bool>();
  props.supportsVoteOps = node["supports-vote-ops"].as<bool>();
  props.supportsArithmeticOps = node["supports-arithmetic-ops"].as<bool>();
  props.supportsBallotOps = node["supports-ballot-ops"].as<bool>();
  props.supportsShuffleOps = node["supports-shuffle-ops"].as<bool>();
  props.supportsShuffleRelativeOps =
      node["supports-shuffle-relative-ops"].as<bool>();
  const YAML::Node &control = node["subgroup-control"];
  if (control.IsNull()) {
    props.controlProperties.supported = false;
  } else {
    props.controlProperties.supported = true;
    props.controlProperties.maxComputeWorkgroupSubgroups =
        control["max-compute-workgroup-subgroups"].as<uint32_t>();
    auto seq = control["supported-subgroup-sizes"].as<std::vector<uint32_t>>();
    props.controlProperties.supportedSubgroupSizes.assign(seq.begin(),
                                                          seq.end());
  }
  return props;
}

static denox::MemoryModelProperties
deserialize_memory_model(const YAML::Node &node) {
  denox::MemoryModelProperties props;
  props.vmm = node["vulkan-memory-model"].as<bool>();
  props.vmmDeviceScope = node["vulkan-memory-model-device-scope"].as<bool>();
  return props;
}

static denox::memory::Dtype deserialize_dtype(const YAML::Node &node) {
  std::string str = node.as<std::string>();
  if (str == "float16" || str == "f16") {
    return denox::memory::Dtype::F16;
  } else if (str == "float32" || str == "f32" || str == "float") {
    return denox::memory::Dtype::F32;
  } else if (str == "float64" || str == "f64" || str == "double") {
    return denox::memory::Dtype::F64;
  } else if (str == "uint32" || str == "u32") {
    return denox::memory::Dtype::U32;
  } else if (str == "uint64" || str == "u64") {
    return denox::memory::Dtype::U64;
  } else if (str == "int32" || str == "i32") {
    return denox::memory::Dtype::U32;
  } else if (str == "int64" || str == "i64") {
    return denox::memory::Dtype::I64;
  } else {
    throw std::runtime_error(fmt::format("Invalid datatype {}", str));
  }
}

static denox::CoopmatShape deserialize_coopmat_shape(const YAML::Node &node) {
  denox::CoopmatShape shape;
  shape.N = node["N"].as<uint32_t>();
  shape.K = node["K"].as<uint32_t>();
  shape.M = node["M"].as<uint32_t>();
  shape.atype = deserialize_dtype(node["atype"]);
  shape.btype = deserialize_dtype(node["btype"]);
  shape.ctype = deserialize_dtype(node["ctype"]);
  shape.acctype = deserialize_dtype(node["acctype"]);
  shape.saturatingAccumulation = node["saturating-accumulation"].as<bool>();
  shape.subgroupScope = node["subgroup-scope"].as<bool>();
  return shape;
}

static denox::CoopmatProperties
deserialize_coopmat_shapes(const YAML::Node &node) {
  assert(node.IsSequence());
  size_t count = node.size();
  denox::CoopmatProperties props;
  for (size_t i = 0; i < count; ++i) {
    props.shapes.emplace_back(deserialize_coopmat_shape(node[i]));
  }
  props.supported = !props.shapes.empty();
  return props;
}

denox::DeviceInfo
deserialize_device_yml(std::span<const std::byte> device_yml) {
  const char *device_yml_cstr =
      reinterpret_cast<const char *>(device_yml.data());
  const std::string device_yml_str{device_yml_cstr, device_yml.size()};
  const YAML::Node yaml = YAML::Load(device_yml_str);
  denox::DeviceInfo deviceInfo;
  deviceInfo.apiVersion = deserialize_api_version(yaml["api-version"]);
  deviceInfo.name = deserialize_device_name(yaml["device-name"]);
  deviceInfo.limits = deserialize_resource_limits(yaml["limits"]);
  deviceInfo.subgroup =
      deserialize_subgroup_properties(yaml["subgroup-properties"]);
  deviceInfo.memoryModel = deserialize_memory_model(yaml["memory-model"]);
  deviceInfo.coopmat = deserialize_coopmat_shapes(yaml["coopmat-shapes"]);
  return deviceInfo;
}

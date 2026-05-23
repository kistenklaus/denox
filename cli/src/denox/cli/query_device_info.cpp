#include "query_device_info.hpp"
#include "denox/cli/io/OutputStream.hpp"
#include "denox/device_info/CoopmatProperties.hpp"
#include "denox/device_info/query/query_driver_device_info.hpp"
#include "denox/io/fs/Path.hpp"
#include "yaml-cpp/emittermanip.h"
#include "yaml-cpp/null.h"
#include <span>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

void query_device_info(QueryDeviceInfo &action) {

  auto apiVersion = denox::ApiVersion::VULKAN_1_4;
  denox::DeviceInfo deviceInfo =
      denox::query_driver_device_info(apiVersion, action.deviceName);

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

  IOEndpoint output = [&]() {
    if (action.output.has_value()) {
      return *action.output;
    } else {
      return IOEndpoint{
          denox::io::Path{deviceInfo.name}.with_extension("yaml")};
    }
  }();

  OutputStream outstream{output};
  const char *out_cstr = out.c_str();
  fmt::println("{}", out_cstr);
  static_assert(sizeof(char) == sizeof(std::byte));
  outstream.write_exact(
      std::span{reinterpret_cast<const std::byte *>(out_cstr), out.size()});
}

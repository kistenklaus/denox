#include "denox/compiler/recover_options/recover_options.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/symbolic/Sym.hpp"
#include <algorithm>
#include <cassert>
#include <dnx.h>
#include <stdexcept>

namespace denox::compiler {

CompileOptions recover_options(memory::span<const std::byte> dnxBuf) {
  CompileOptions options;

  const auto *dnx = dnx::GetModel(dnxBuf.data());
  const auto *model_info = dnx->info();
  if (model_info == nullptr) {
    throw std::runtime_error("Failed to recover options from dnx artefact. "
                             "Missing debug information.");
  }
  const auto *compilation_info = model_info->compilation_info();
  if (compilation_info == nullptr) {
    throw std::runtime_error("Failed to recover options from dnx artefact. "
                             "Missing debug information.");
  }

  // recover descriptor policies
  {
    const auto *descriptor_policies = compilation_info->descriptor_policy();
    if (descriptor_policies == nullptr) {
      throw std::runtime_error("Failed to recover options from dnx artefact. "
                               "Missing debug information.");
    }
    options.descriptorPolicies.inputPolicy.set =
        descriptor_policies->input_policy();
    options.descriptorPolicies.outputPolicy.set =
        descriptor_policies->output_policy();
    options.descriptorPolicies.paramPolicy.set =
        descriptor_policies->param_policy();
    options.descriptorPolicies.readPolicy.set =
        descriptor_policies->read_policy();
    options.descriptorPolicies.writePolicy.set =
        descriptor_policies->write_policy();
  }
  // recover features
  {
    const auto *features = compilation_info->features();
    options.features.coopmat = features->coopmat();
    options.features.enableImplicitConcat = features->implicit_concat();
    options.features.enableConvReluFusion = features->conv_relu_fusion();
    options.features.enableConcatConvFusion = features->concat_conv_fusion();
    options.features.enableUpsampleConvFusion =
        features->upsample_conv_fusion();
    options.features.enableConvMaxPoolFusion = features->conv_pool_fusion();
  }
  // recover device info.
  {
    const auto *device_info = compilation_info->device_info();
    options.deviceInfo.name = *device_info->device_name();
    switch (device_info->api_version()) {
    case dnx::VulkanApiVersion_VULKAN_1_0:
      options.deviceInfo.apiVersion = ApiVersion::VULKAN_1_0;
      break;
    case dnx::VulkanApiVersion_VULKAN_1_1:
      options.deviceInfo.apiVersion = ApiVersion::VULKAN_1_1;
      break;
    case dnx::VulkanApiVersion_VULKAN_1_2:
      options.deviceInfo.apiVersion = ApiVersion::VULKAN_1_2;
      break;
    case dnx::VulkanApiVersion_VULKAN_1_3:
      options.deviceInfo.apiVersion = ApiVersion::VULKAN_1_3;
      break;
    case dnx::VulkanApiVersion_VULKAN_1_4:
      options.deviceInfo.apiVersion = ApiVersion::VULKAN_1_4;
      break;
    }
    options.deviceInfo.limits.maxComputeWorkGroupCount[0] =
        device_info->max_compute_workgroup_count_x();
    options.deviceInfo.limits.maxComputeWorkGroupCount[1] =
        device_info->max_compute_workgroup_count_y();
    options.deviceInfo.limits.maxComputeWorkGroupCount[2] =
        device_info->max_compute_workgroup_count_z();

    options.deviceInfo.limits.maxComputeWorkGroupSize[0] =
        device_info->max_compute_workgroup_size_x();
    options.deviceInfo.limits.maxComputeWorkGroupSize[1] =
        device_info->max_compute_workgroup_size_y();
    options.deviceInfo.limits.maxComputeWorkGroupSize[2] =
        device_info->max_compute_workgroup_size_z();

    options.deviceInfo.limits.maxComputeWorkGroupInvocations =
        device_info->max_compute_workgroup_invocations();
    options.deviceInfo.subgroup.controlProperties.maxComputeWorkgroupSubgroups =
        device_info->max_compute_workgroup_subgroups();
    options.deviceInfo.limits.maxComputeSharedMemory =
        device_info->max_compute_shared_memory();

    options.deviceInfo.limits.maxPushConstantSize =
        device_info->max_push_constant_size();

    if (device_info->supported_subgroup_sizes()->size() == 1) {
      options.deviceInfo.subgroup.subgroupSize =
          device_info->supported_subgroup_sizes()->Get(0);
      options.deviceInfo.subgroup.controlProperties.supported = false;
    } else {
      options.deviceInfo.subgroup.subgroupSize = 0;
      options.deviceInfo.subgroup.controlProperties.supported = true;
      options.deviceInfo.subgroup.controlProperties.supportedSubgroupSizes
          .assign(device_info->supported_subgroup_sizes()->begin(),
                  device_info->supported_subgroup_sizes()->end());
    }
    options.deviceInfo.subgroup.supportsBasicOps =
        device_info->subgroup_basic_ops();
    options.deviceInfo.subgroup.supportsVoteOps =
        device_info->subgroup_vote_ops();
    options.deviceInfo.subgroup.supportsArithmeticOps =
        device_info->subgroup_arithmetic_ops();
    options.deviceInfo.subgroup.supportsBallotOps =
        device_info->subgroup_ballot_ops();
    options.deviceInfo.subgroup.supportsShuffleOps =
        device_info->subgroup_shuffle_ops();
    options.deviceInfo.subgroup.supportsShuffleRelativeOps =
        device_info->subgroup_shuffle_relative_ops();

    options.deviceInfo.memoryModel.vmm = device_info->vulkan_memory_model();
    options.deviceInfo.memoryModel.vmmDeviceScope =
        device_info->vulkan_memory_model_device_scope();

    if (device_info->supported_coopmat_shapes()->empty()) {
      options.deviceInfo.coopmat.supported = false;
    } else {
      options.deviceInfo.coopmat.supported = true;

      const auto deserialize_type =
          [](denox::dnx::ScalarType type) -> memory::Dtype {
        switch (type) {
        case dnx::ScalarType_I16:
        case dnx::ScalarType_U16:
          diag::invalid_state();
        case dnx::ScalarType_I32:
          return memory::Dtype::I32;
        case dnx::ScalarType_U32:
          return memory::Dtype::U32;
        case dnx::ScalarType_I64:
          return memory::Dtype::I64;
        case dnx::ScalarType_U64:
          return memory::Dtype::U64;
        case dnx::ScalarType_F16:
          return memory::Dtype::F16;
        case dnx::ScalarType_F32:
          return memory::Dtype::F32;
        case dnx::ScalarType_F64:
          return memory::Dtype::F64;
        default:
          diag::unreachable();
        }
      };

      for (const dnx::CoopmatShape *shape :
           *device_info->supported_coopmat_shapes()) {
        options.deviceInfo.coopmat.shapes.push_back(CoopmatShape{
            .M = shape->m(),
            .N = shape->n(),
            .K = shape->k(),
            .atype = deserialize_type(shape->atype()),
            .btype = deserialize_type(shape->atype()),
            .ctype = deserialize_type(shape->ctype()),
            .acctype = deserialize_type(shape->acctype()),
            .saturatingAccumulation = shape->saturating(),
            .subgroupScope = true, // dnx only stores subgroup scope shapes.
        });
      }
    }
  }

  // quick helper to parse dnx value values into something usable.
  const auto *value_names = dnx->value_names();
  memory::vector<std::pair<std::string, Sym>> valueNames;
  if (value_names != nullptr) {
    for (const dnx::ValueName *value_name : *value_names) {
      memory::string name{value_name->name()->begin(),
                          value_name->name()->end()};
      Sym value;
      if (value_name->value_type() == dnx::ScalarSource_literal) {
        const auto *literal = value_name->value_as_literal();
        switch (literal->dtype()) {
        case dnx::ScalarType_I16: {
          int16_t v;
          std::memcpy(&v, literal->bytes()->data(), sizeof(int16_t));
          value = Sym::Const(static_cast<int64_t>(v));
          break;
        }
        case dnx::ScalarType_U16: {
          uint16_t v;
          std::memcpy(&v, literal->bytes()->data(), sizeof(uint16_t));
          value = Sym::Const(static_cast<int64_t>(v));
          break;
        }
        case dnx::ScalarType_I32: {
          int32_t v;
          std::memcpy(&v, literal->bytes()->data(), sizeof(int32_t));
          value = Sym::Const(static_cast<int64_t>(v));
          break;
        }
        case dnx::ScalarType_U32: {
          uint32_t v;
          std::memcpy(&v, literal->bytes()->data(), sizeof(uint32_t));
          value = Sym::Const(static_cast<int64_t>(v));
          break;
        }
        case dnx::ScalarType_I64: {
          int64_t v;
          std::memcpy(&v, literal->bytes()->data(), sizeof(int64_t));
          value = Sym::Const(v);
          break;
        }
        case dnx::ScalarType_U64: {
          uint64_t v;
          std::memcpy(&v, literal->bytes()->data(), sizeof(uint64_t));
          value = Sym::Const(static_cast<int64_t>(v));
          break;
        }
        case dnx::ScalarType_F16:
        case dnx::ScalarType_F32:
        case dnx::ScalarType_F64:
          diag::invalid_state();
        default:
          diag::unreachable();
        }
      } else {
        auto *symRef = value_name->value_as_symbolic();
        value = Sym::Symbol(static_cast<uint64_t>(symRef->sid()));
      }
      valueNames.push_back(std::make_pair(name, value));
    }
  }

  { // partially recover interface descriptors.
    assert(dnx->inputs() != nullptr);
    assert(dnx->tensors() != nullptr);
    for (uint32_t tid : *dnx->inputs()) {
      const auto *input = dnx->tensors()->Get(tid);
      if (input->info() == nullptr || input->info()->name() == nullptr) {
        throw std::runtime_error("Failed to recover options from dnx artefact. "
                                 "Missing debug information.");
      }
      const auto *info = input->info();
      TensorFormat format;
      switch (info->format()) {
      case dnx::TensorFormat_UNKNOWN:
        diag::invalid_state();
      case dnx::TensorFormat_SSBO_HWC:
        format = TensorFormat::SSBO_HWC;
        break;
      case dnx::TensorFormat_SSBO_CHW:
        format = TensorFormat::SSBO_CHW;
        break;
      case dnx::TensorFormat_SSBO_CHWC8:
        format = TensorFormat::SSBO_CHWC8;
        break;
      case dnx::TensorFormat_TEX_RGBA:
        format = TensorFormat::TEX_RGBA;
        break;
      case dnx::TensorFormat_TEX_RGB:
        format = TensorFormat::TEX_RGB;
        break;
      case dnx::TensorFormat_TEX_RG:
        format = TensorFormat::TEX_RG;
        break;
      case dnx::TensorFormat_TEX_R:
        format = TensorFormat::TEX_R;
        break;
      default:
        diag::unreachable();
      }

      TensorStorage storage;
      switch (info->storage()) {
      case dnx::TensorStorage_StorageBuffer:
        storage = TensorStorage::StorageBuffer;
        break;
      case dnx::TensorStorage_StorageImage:
        storage = TensorStorage::StorageBuffer;
        break;
      case dnx::TensorStorage_SampledStorageImage:
        storage = TensorStorage::StorageBuffer;
        break;
      default:
        diag::unreachable();
      }

      TensorDataType dtype;
      switch (info->type()) {
      case dnx::ScalarType_I16:
      case dnx::ScalarType_U16:
        diag::invalid_state();
      case dnx::ScalarType_I32:
      case dnx::ScalarType_U32:
      case dnx::ScalarType_I64:
      case dnx::ScalarType_U64:
      case dnx::ScalarType_F16:
        dtype = TensorDataType::Float16;
        break;
      case dnx::ScalarType_F32:
        dtype = TensorDataType::Float32;
        break;
      case dnx::ScalarType_F64:
        dtype = TensorDataType::Float64;
        break;
      default:
        diag::unreachable();
      }

      memory::string name{info->name()->begin(), info->name()->end()};
      options.interfaceDescriptors.push_back(InterfaceTensorDescriptor{
          .name = std::move(name),
          .format = format,
          .storage = storage,
          .dtype = dtype,
          .heightValueName = memory::nullopt,
          .height = memory::nullopt,
          .widthValueName = memory::nullopt,
          .width = memory::nullopt,
          .channelValueName = memory::nullopt,
          .channels = memory::nullopt,
      });
    }
  }

  // partially recover assumptions:
  {
    const auto *assumptions = compilation_info->assumptions();
    if (assumptions == nullptr) {
      throw std::runtime_error("Failed to recover options from dnx artefact. "
                               "Missing debug information.");
    }
    for (const dnx::Assumption *assumption : *assumptions) {
      Sym::symbol sid = static_cast<Sym::symbol>(assumption->sid());
      uint32_t value = assumption->value();

      // Find any input / output, which has sid as a channel, width or height.
      bool found = false;
      // 1. Check inputs
      for (const uint32_t inputTensorId : *dnx->inputs()) {
        const auto *tensor = dnx->tensors()->Get(inputTensorId);
        auto *info = tensor->info();
        if (info == nullptr || info->name() == nullptr) {
          throw std::runtime_error(
              "Failed to recover options from dnx artefact. "
              "Missing debug information.");
        }
        memory::string name{info->name()->begin(), info->name()->end()};
        auto it = std::ranges::find_if(
            options.interfaceDescriptors,
            [&](const auto &desc) -> bool { return desc.name == name; });
        assert(it != options.interfaceDescriptors.end());
        InterfaceTensorDescriptor &desc = *it;
        if (info->channels_type() == dnx::ScalarSource_symbolic) {
          const uint32_t csid = info->channels_as_symbolic()->sid();
          if (csid == sid) {
            desc.channels = value;
            found = true;
            break;
          }
        }
        if (info->width_type() == dnx::ScalarSource_symbolic) {
          const uint32_t wsid = info->width_as_symbolic()->sid();
          if (wsid == sid) {
            desc.width = value;
            found = true;
            break;
          }
        }
        if (info->height_type() == dnx::ScalarSource_symbolic) {
          const uint32_t hsid = info->height_as_symbolic()->sid();
          if (hsid == sid) {
            desc.height = value;
            found = true;
            break;
          }
        }
      }

      if (!found) {
        for (const uint32_t outputTensorId : *dnx->outputs()) {
          const auto *tensor = dnx->tensors()->Get(outputTensorId);
          auto *info = tensor->info();
          if (info == nullptr || info->name() == nullptr) {
            throw std::runtime_error(
                "Failed to recover options from dnx artefact. "
                "Missing debug information.");
          }
          memory::string name{info->name()->begin(), info->name()->end()};
          auto it = std::ranges::find_if(
              options.interfaceDescriptors,
              [&](const auto &desc) -> bool { return desc.name == name; });
          assert(it != options.interfaceDescriptors.end());
          InterfaceTensorDescriptor &desc = *it;
          if (info->channels_type() == dnx::ScalarSource_symbolic) {
            const uint32_t csid = info->channels_as_symbolic()->sid();
            if (csid == sid) {
              desc.channels = value;
              break;
            }
          }
          if (info->width_type() == dnx::ScalarSource_symbolic) {
            const uint32_t wsid = info->width_as_symbolic()->sid();
            if (wsid == sid) {
              desc.width = value;
              break;
            }
          }
          if (info->height_type() == dnx::ScalarSource_symbolic) {
            const uint32_t hsid = info->height_as_symbolic()->sid();
            if (hsid == sid) {
              desc.height = value;
              break;
            }
          }
        }
      }
    }
  }
  // recover optimization level
  options.optimizationLevel = compilation_info->optimization_level();

  return options;
}

} // namespace denox::compiler

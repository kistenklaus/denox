#include "denox/spirv/SpirvTools.hpp"
#include "denox/diag/unreachable.hpp"
#include <memory>
#include <spirv-tools/libspirv.h>
#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>

static spvtools::ValidatorOptions default_validator_options() {
  spvtools::ValidatorOptions valOptions;
  // Only if you want extra guardrails; otherwise omit these:
  valOptions.SetUniversalLimit(spv_validator_limit_max_struct_members, 1024);
  valOptions.SetUniversalLimit(spv_validator_limit_max_struct_depth, 64);
  valOptions.SetUniversalLimit(spv_validator_limit_max_local_variables, 8192);
  valOptions.SetUniversalLimit(spv_validator_limit_max_global_variables, 4096);
  valOptions.SetUniversalLimit(spv_validator_limit_max_switch_branches, 16384);
  valOptions.SetUniversalLimit(spv_validator_limit_max_function_args, 64);
  valOptions.SetUniversalLimit(
      spv_validator_limit_max_control_flow_nesting_depth, 128);
  valOptions.SetUniversalLimit(spv_validator_limit_max_access_chain_indexes,
                               255);
  // Be careful with this one if you also run optimizers:
  valOptions.SetUniversalLimit(spv_validator_limit_max_id_bound, 1u << 24);

  valOptions.SetRelaxBlockLayout(true);
  valOptions.SetRelaxStructStore(false);
  valOptions.SetUniformBufferStandardLayout(false);
  valOptions.SetScalarBlockLayout(false);
  valOptions.SetWorkgroupScalarBlockLayout(false);
  valOptions.SetSkipBlockLayout(false);
  valOptions.SetAllowLocalSizeId(false); // <- spec constants.
  valOptions.SetAllowOffsetTextureOperand(false);
  valOptions.SetAllowVulkan32BitBitwise(false);
  valOptions.SetRelaxLogicalPointer(false);
  valOptions.SetBeforeHlslLegalization(false);
  valOptions.SetFriendlyNames(true);
  return valOptions;
}

denox::spirv::SpirvTools::SpirvTools(const DeviceInfo &deviceInfo)
    : m_logger(std::make_shared<AsyncLogger>()) {

  switch (deviceInfo.apiVersion) {
  case ApiVersion::VULKAN_1_0:
    m_targetEnv = SPV_ENV_VULKAN_1_0;
    break;
  case ApiVersion::VULKAN_1_1:
    m_targetEnv = SPV_ENV_VULKAN_1_1;
    break;
  case ApiVersion::VULKAN_1_2:
    m_targetEnv = SPV_ENV_VULKAN_1_2;
    break;
  case ApiVersion::VULKAN_1_3:
    m_targetEnv = SPV_ENV_VULKAN_1_3;
    break;
  case ApiVersion::VULKAN_1_4:
    m_targetEnv = SPV_ENV_VULKAN_1_4;
    break;
  default:
    fmt::println("IAM unreachable");
    diag::unreachable();
  }
}

bool denox::spirv::SpirvTools::validate(const SpirvBinary &binary) {
  auto state = thread_local_state();
  state->stage = "spirv-val";
  return state->tools.Validate(binary.spv.data(), binary.spv.size(),
                        default_validator_options());
}

bool denox::spirv::SpirvTools::optimize(SpirvBinary &binary) {
  auto state = thread_local_state();

  spvtools::OptimizerOptions optOptions;
  optOptions.set_run_validator(false);
  optOptions.set_validator_options(default_validator_options());
  optOptions.set_preserve_bindings(true);
  optOptions.set_preserve_spec_constants(true);

  std::vector<uint32_t> original = binary.spv;
  binary.spv.clear();
  state->stage = "spirv-opt";
  return state->optimizer.Run(original.data(), original.size(), &binary.spv);
}


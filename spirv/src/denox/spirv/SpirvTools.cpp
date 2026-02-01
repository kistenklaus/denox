#include "denox/spirv/SpirvTools.hpp"
#include "denox/diag/unreachable.hpp"
#include <memory>
#include <spirv-tools/libspirv.h>
#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>
#include <utility>

static constexpr uint16_t OpDecorate = 71;
static constexpr uint32_t MagicNumber = 0x07230203;
static constexpr uint32_t DecorationDescriptorSet = 34;
static constexpr uint32_t DecorationBinding = 33;

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
    : m_tools(malloc(sizeof(spvtools::SpirvTools))),
      m_optimizer(malloc(sizeof(spvtools::Optimizer))) {
  auto tools = static_cast<spvtools::SpirvTools *>(m_tools);
  auto optimizer = static_cast<spvtools::Optimizer *>(m_optimizer);

  spv_target_env target_env;
  switch (deviceInfo.apiVersion) {
  case ApiVersion::VULKAN_1_0:
    target_env = SPV_ENV_VULKAN_1_0;
    break;
  case ApiVersion::VULKAN_1_1:
    target_env = SPV_ENV_VULKAN_1_1;
    break;
  case ApiVersion::VULKAN_1_2:
    target_env = SPV_ENV_VULKAN_1_2;
    break;
  case ApiVersion::VULKAN_1_3:
    target_env = SPV_ENV_VULKAN_1_3;
    break;
  case ApiVersion::VULKAN_1_4:
    target_env = SPV_ENV_VULKAN_1_4;
    break;
  default:
    fmt::println("IAM unreachable");
    diag::unreachable();
  }

  std::construct_at(tools, target_env);

  auto logger = [this](spv_message_level_t level, const char *source,
                       const spv_position_t &p, const char *msg) {
    static const bool use_color = true;
    const char *R = use_color ? "\x1b[0m" : "";
    const char *B = use_color ? "\x1b[1m" : "";
    const char *RED = use_color ? "\x1b[31m" : "";
    const char *YEL = use_color ? "\x1b[33m" : "";
    const char *CYN = use_color ? "\x1b[36m" : "";
    const char *MAG = use_color ? "\x1b[35m" : "";
    const char *GRY = use_color ? "\x1b[90m" : "";

    // Map level → label/color
    const char *label = "message";
    const char *col = GRY;
    switch (level) {
    case SPV_MSG_FATAL:
      label = "fatal error";
      col = RED;
      break;
    case SPV_MSG_INTERNAL_ERROR:
      label = "internal error";
      col = MAG;
      break;
    case SPV_MSG_ERROR:
      label = "error";
      col = RED;
      break;
    case SPV_MSG_WARNING:
      label = "warning";
      col = YEL;
      break;
    case SPV_MSG_INFO:
      label = "info";
      col = CYN;
      break;
    case SPV_MSG_DEBUG:
      label = "debug";
      col = GRY;
      break;
    default:
      label = "message";
      col = GRY;
      break;
    }

    if (!source)
      source = m_current_stage;
    if (!msg)
      msg = "(no message)";

    // Trim trailing whitespace/newlines (keep it compact like compilers)
    std::string_view m{msg};
    while (!m.empty() && (m.back() == '\n' || m.back() == '\r' ||
                          m.back() == ' ' || m.back() == '\t'))
      m.remove_suffix(1);

    // <source>:<line>:<col>: <severity>: <message>
    fmt::format_to(std::back_inserter(m_log), "{}{}{}", B, source, R);
    if (p.line || p.column)
      fmt::format_to(std::back_inserter(m_log), ":{}:{}", p.line ? p.line : 0,
                     p.column ? p.column : 0);
    fmt::format_to(std::back_inserter(m_log), ": {}{}{}: {}\n", col, label, R,
                   m);

    // Optional extra context (word index) in dim text
    if (p.index)
      fmt::format_to(std::back_inserter(m_log), "{}  note: word-index {}{}\n",
                     GRY, p.index, R);
  };
  tools->SetMessageConsumer(logger);

  std::construct_at(optimizer, target_env);
  optimizer->SetMessageConsumer(logger);
  optimizer->SetTargetEnv(target_env);
  optimizer->SetValidateAfterAll(true);
  optimizer->RegisterPerformancePasses();
}

denox::spirv::SpirvTools::SpirvTools(SpirvTools &o)
    : m_tools(std::exchange(o.m_tools, nullptr)),
      m_optimizer(std::exchange(o.m_optimizer, nullptr)) {}

denox::spirv::SpirvTools &denox::spirv::SpirvTools::operator=(SpirvTools &o) {
  if (this == &o) {
    return *this;
  }
  if (m_tools != nullptr) {
    free(m_tools);
    free(m_optimizer);
  }
  m_tools = std::exchange(o.m_tools, nullptr);
  m_optimizer = std::exchange(o.m_optimizer, nullptr);
  return *this;
}

bool denox::spirv::SpirvTools::validate(const SpirvBinary &binary) {
  auto tools = static_cast<spvtools::SpirvTools *>(m_tools);
  m_log.clear();
  m_current_stage = "spirv-val";
  return tools->Validate(binary.spv.data(), binary.spv.size(),
                         default_validator_options());
}
denox::spirv::SpirvTools::~SpirvTools() {
  if (m_tools != nullptr) {
    auto *tools = static_cast<spvtools::SpirvTools *>(m_tools);
    std::destroy_at(tools);
    free(m_tools);
    auto *optimizer = static_cast<spvtools::Optimizer *>(m_optimizer);
    std::destroy_at(optimizer);
    free(m_optimizer);
    m_tools = nullptr;
    m_optimizer = nullptr;
  }
}

bool denox::spirv::SpirvTools::optimize(SpirvBinary &binary) {
  auto *optimizer = static_cast<spvtools::Optimizer *>(m_optimizer);

  spvtools::OptimizerOptions optOptions;
  optOptions.set_run_validator(false);
  optOptions.set_validator_options(default_validator_options());
  optOptions.set_preserve_bindings(true);
  optOptions.set_preserve_spec_constants(true);

  std::vector<uint32_t> original = binary.spv;
  binary.spv.clear();
  m_log.clear();
  m_current_stage = "spirv-opt";
  return optimizer->Run(original.data(), original.size(), &binary.spv);
}

bool denox::spirv::SpirvTools::rebind(
    SpirvBinary &binary, memory::span<const SpirvDescriptorRebind> rebinds) {
  m_log.clear();
  m_current_stage = "spirv-rebind";

  std::span<uint32_t> spirv = binary.spv;
  assert(spirv.size() > 5);

  [[maybe_unused]] const uint32_t magic = spirv[0];
  assert(magic == MagicNumber);

  uint32_t pc = 5;
  struct RebindLoc {
    uint32_t id;
    uint32_t pc_set;
    uint32_t pc_binding;
  };
  std::vector<RebindLoc> locs;
  locs.reserve(10);

  const auto try_rebind_set = [&](uint32_t id, uint32_t pc_set,
                                  uint32_t pc_binding) {
    assert(pc_set != 0 || pc_binding != 0);
    auto loc = std::ranges::find_if(
        locs, [id](const RebindLoc &loc) { return loc.id == id; });
    if (loc == locs.end()) {
      locs.emplace_back(id, pc_set, pc_binding);
      return;
    }
    loc->pc_binding |= pc_binding;
    loc->pc_set |= pc_set;
    if (loc->pc_set != 0 && loc->pc_binding != 0) {
      const uint32_t set = spirv[loc->pc_set];
      const uint32_t binding = spirv[loc->pc_binding];
      auto rebind = std::ranges::find_if(
          rebinds, [set, binding](const SpirvDescriptorRebind &rebind) {
            return rebind.src_set == set && rebind.src_binding == binding;
          });
      if (rebind != rebinds.end()) {
        spirv[loc->pc_set] = rebind->dst_set;
        spirv[loc->pc_binding] = rebind->dst_binding;
      }
      std::swap(*loc, locs.back());
      locs.pop_back();
    }
  };

  while (pc < spirv.size()) {
    uint32_t op = spirv[pc];
    uint32_t size = op >> 16;
    uint32_t opcode = op & 0xFFFF;
    assert(size > 0);
    assert(pc + size <= spirv.size());
    if (opcode == OpDecorate) {
      assert(size >= 3);
      const uint32_t id = spirv[pc + 1];
      const uint32_t decoration = spirv[pc + 2];
      if (decoration == DecorationDescriptorSet) {
        assert(size >= 4);
        try_rebind_set(id, pc + 3, 0);
      } else if (decoration == DecorationBinding) {
        assert(size >= 4);
        try_rebind_set(id, 0, pc + 3);
      }
    }
    pc += size;
  }
  return true;
}

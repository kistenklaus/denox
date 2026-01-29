#include "model.hpp"
#include "context.hpp"
#include "denox/common/PushConstant.hpp"
#include "denox/common/SHA256.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/diag/invalid_state.hpp"
#include "denox/diag/logging.hpp"
#include "denox/diag/not_implemented.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/symbolic/SymIR.hpp"
#include "dnx.h"
#include <algorithm>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <string_view>

namespace denox::runtime {

static Sym parse_sym(dnx::ScalarSource type, const void *ptr) {
  switch (type) {
  case dnx::ScalarSource_NONE:
    diag::invalid_state();
  case dnx::ScalarSource_literal: {
    const auto *literal = static_cast<const dnx::ScalarLiteral *>(ptr);
    int64_t value;
    switch (literal->dtype()) {
    case dnx::ScalarType_I16: {
      int16_t x;
      std::memcpy(&x, literal->bytes()->data(), 2);
      value = x;
      break;
    }
    case dnx::ScalarType_U16: {
      uint16_t x;
      std::memcpy(&x, literal->bytes()->data(), 2);
      value = x;
      break;
    }
    case dnx::ScalarType_I32: {
      int32_t x;
      std::memcpy(&x, literal->bytes()->data(), 4);
      value = x;
      break;
    }
    case dnx::ScalarType_U32: {
      uint32_t x;
      std::memcpy(&x, literal->bytes()->data(), 4);
      value = x;
      break;
    }
    case dnx::ScalarType_I64: {
      int64_t x;
      std::memcpy(&x, literal->bytes()->data(), 8);
      value = x;
      break;
    }
    case dnx::ScalarType_U64: {
      uint64_t x;
      std::memcpy(&x, literal->bytes()->data(), 8);
      value = static_cast<int64_t>(x);
      break;
    }
    case dnx::ScalarType_F16:
    case dnx::ScalarType_F32:
    case dnx::ScalarType_F64:
      diag::invalid_state();
    default:
      diag::unreachable();
    }
    return Sym::Const(value);
  }
  case dnx::ScalarSource_symbolic: {
    const auto *symref = reinterpret_cast<const dnx::SymRef *>(ptr);
    return Sym::Symbol(static_cast<Sym::symbol>(symref->sid()));
  }
  }
  diag::unreachable();
}

static ModelDispatch
create_model_dispatch(const runtime::ContextHandle &ctx, const dnx::Model *dnx,
                      const dnx::ComputeDispatch *dispatch,
                      memory::span<const ModelTensor> tensors) {

  std::size_t descriptorSetCount = dispatch->bindings()->size();
  std::vector<VkDescriptorSetLayout> descriptorSetLayouts(descriptorSetCount);
  std::vector<runtime::ModelDescriptorSet> descriptorSets(descriptorSetCount);

  std::vector<VkDescriptorSetLayoutBinding> bindings;
  bindings.reserve(8);
  for (std::size_t s = 0; s < dispatch->bindings()->size(); ++s) {
    const dnx::DescriptorSetBinding *set =
        dispatch->bindings()->Get(static_cast<unsigned int>(s));
    bindings.clear();

    memory::vector<ModelDescriptorBinding> modelBindings(
        set->bindings()->size());
    for (std::size_t b = 0; b < set->bindings()->size(); ++b) {
      const dnx::DescriptorBinding *binding =
          set->bindings()->Get(static_cast<unsigned int>(b));
      VkDescriptorSetLayoutBinding descriptorSetLayoutBinding;
      descriptorSetLayoutBinding.binding = binding->binding();
      descriptorSetLayoutBinding.descriptorCount = 1;
      descriptorSetLayoutBinding.descriptorType =
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptorSetLayoutBinding.pImmutableSamplers = nullptr;
      descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bindings.push_back(descriptorSetLayoutBinding);

      Access access;
      switch (binding->access()) {
      case dnx::Access_ReadOnly:
        access = Access::ReadOnly;
        break;
      case dnx::Access_WriteOnly:
        access = Access::WriteOnly;
        break;
      case dnx::Access_ReadWrite:
        DENOX_WARN("readwrite is not really supported, but we will let it pass "
                   "for now");
        access = Access::ReadWrite;
        break;
      }

      modelBindings[b] = ModelDescriptorBinding{
          .binding = binding->binding(),
          .access = access,
          .tensor = binding->tensor(),
      };
    }
    VkDescriptorSetLayout setLayout = ctx->createDescriptorSetLayout(bindings);
    descriptorSetLayouts[s] = setLayout;
    descriptorSets[s].descriptorSetLayout = setLayout;
    descriptorSets[s].set = set->set();
    descriptorSets[s].bindings = std::move(modelBindings);
  }

  std::uint32_t pushConstantRange = dispatch->push_constant()->size();
  VkPipelineLayout pipelineLayout =
      ctx->createPipelineLayout(descriptorSetLayouts, pushConstantRange);

  std::uint32_t binaryId = dispatch->binary_id();
  const dnx::ShaderBinary *binary = dnx->shader_binaries()->Get(binaryId);

  std::span<const std::uint32_t> spirv(binary->spirv()->data(),
                                       binary->spirv()->size());
  VkPipeline pipeline = ctx->createComputePipeline(
      pipelineLayout, spirv, dispatch->entry_point()->c_str());

  Sym workgroupCountX = parse_sym(dispatch->workgroup_count_x_type(),
                                  dispatch->workgroup_count_x());
  Sym workgroupCountY = parse_sym(dispatch->workgroup_count_y_type(),
                                  dispatch->workgroup_count_y());
  Sym workgroupCountZ = parse_sym(dispatch->workgroup_count_z_type(),
                                  dispatch->workgroup_count_z());

  const auto *pc = dispatch->push_constant();
  memory::vector<std::pair<uint16_t, PushConstant>> pushConstants;
  uint16_t pcRange = 0;
  if (pc) {
    pcRange = pc->size();
    const auto *fields = pc->fields();
    uint32_t fieldCount = fields->size();
    for (uint32_t f = 0; f < fieldCount; ++f) {
      const auto *field = fields->Get(f);
      Sym value = parse_sym(field->source_type(), field->source());
      memory::Dtype type;
      switch (field->dtype()) {
      case dnx::ScalarType_I16:
      case dnx::ScalarType_U16:
        diag::not_implemented();
      case dnx::ScalarType_I32:
        type = memory::Dtype::I32;
        break;
      case dnx::ScalarType_U32:
        type = memory::Dtype::U32;
        break;
      case dnx::ScalarType_I64:
        type = memory::Dtype::I64;
        break;
      case dnx::ScalarType_U64:
        type = memory::Dtype::U64;
        break;
      case dnx::ScalarType_F16:
      case dnx::ScalarType_F32:
      case dnx::ScalarType_F64:
        diag::not_implemented();
      default:
        diag::unreachable();
      }
      pushConstants.emplace_back(field->offset(), PushConstant(value, type));
    }
  }

  memory::optional<memory::string> name;
  memory::optional<memory::string> debugInfo;
  memory::optional<memory::string> srcPath;
  memory::optional<Sym> memoryReads;
  memory::optional<Sym> memoryWrites;
  if (dispatch->info()) {
    const dnx::DispatchInfo *info = dispatch->info();
    if (info->name()) {
      name = info->name()->str();
    }
    if (info->debug_info()) {
      debugInfo = info->debug_info()->str();
    }
    if (info->src_path()) {
      srcPath = info->src_path()->str();
    }
    if (info->memory_reads()) {
      memoryReads = parse_sym(info->memory_reads_type(), info->memory_reads());
    }
    if (info->memory_writes()) {
      memoryWrites =
          parse_sym(info->memory_writes_type(), info->memory_writes());
    }
  }

  return runtime::ModelDispatch{
      .descriptorSets = std::move(descriptorSets),
      .workgroupCountX = workgroupCountX,
      .workgroupCountY = workgroupCountY,
      .workgroupCountZ = workgroupCountZ,
      .pushConstantRange = pcRange,
      .pushConstants = std::move(pushConstants),
      .pipelineLayout = pipelineLayout,
      .pipeline = pipeline,
      .name = name,
      .debugInfo = debugInfo,
      .srcPath = srcPath,
      .memoryReads = memoryReads,
      .memoryWrites = memoryWrites,
  };
}

enum class TensorState {
  Undefined,
  ComputeWrite,
  ComputeRead,
};

static std::optional<runtime::ModelBarrier>
generate_pipeline_barrier([[maybe_unused]] const dnx::Model *dnx,
                          std::vector<TensorState> &tensorStates,
                          const dnx::ComputeDispatch *dispatch) {

  enum HazardType : int { None = 0, RAW = 1, WAW = 2, WAR = 4 };

  std::vector<runtime::ModelBufferBarrier> bufferBarriers;
  for (std::size_t s = 0; s < dispatch->bindings()->size(); ++s) {
    const dnx::DescriptorSetBinding *set =
        dispatch->bindings()->Get(static_cast<unsigned int>(s));
    for (std::size_t b = 0; b < set->bindings()->size(); ++b) {
      const dnx::DescriptorBinding *binding =
          set->bindings()->Get(static_cast<unsigned int>(b));
      TensorState currentState = tensorStates[binding->tensor()];
      dnx::Access access = binding->access();
      HazardType hazard = None;
      TensorState nextState;
      switch (currentState) {
      case TensorState::Undefined:
        switch (access) {
        case dnx::Access_ReadOnly:
          nextState = TensorState::ComputeRead;
          break;
        case dnx::Access_WriteOnly:
        case dnx::Access_ReadWrite:
          nextState = TensorState::ComputeWrite;
          break;
        }
        break;
      case TensorState::ComputeWrite:
        switch (access) {
        case dnx::Access_ReadOnly:
          hazard = static_cast<HazardType>(hazard | RAW);
          nextState = TensorState::ComputeRead;
          break;
        case dnx::Access_WriteOnly:
          hazard = static_cast<HazardType>(hazard | WAW);
          nextState = TensorState::ComputeWrite;
          break;
        case dnx::Access_ReadWrite:
          hazard = static_cast<HazardType>(hazard | RAW);
          hazard = static_cast<HazardType>(hazard | WAW);
          nextState = TensorState::ComputeWrite;
          break;
        }
        break;
      case TensorState::ComputeRead:
        switch (access) {
        case dnx::Access_ReadOnly:
          nextState = TensorState::ComputeRead;
          break;
        case dnx::Access_WriteOnly:
          hazard = static_cast<HazardType>(hazard | WAR);
          nextState = TensorState::ComputeWrite;
          break;
        case dnx::Access_ReadWrite:
          hazard = static_cast<HazardType>(hazard | WAR);
          hazard = static_cast<HazardType>(hazard | WAW);
          nextState = TensorState::ComputeWrite;
          break;
        }
        break;
      }
      if (hazard != None) {
        runtime::ModelBufferBarrier barrier;
        barrier.srcStage = VK_SHADER_STAGE_COMPUTE_BIT;
        barrier.dstStage = VK_SHADER_STAGE_COMPUTE_BIT;
        barrier.srcAccess = 0;
        barrier.dstAccess = 0;
        if (hazard & RAW) {
          barrier.srcAccess |= VK_ACCESS_SHADER_WRITE_BIT;
          barrier.dstAccess |= VK_ACCESS_SHADER_READ_BIT;
        }
        if (hazard & WAR) {
          barrier.srcAccess |= VK_ACCESS_SHADER_READ_BIT;
          barrier.dstAccess |= VK_ACCESS_SHADER_WRITE_BIT;
        }
        if (hazard & WAW) {
          barrier.srcAccess |= VK_ACCESS_SHADER_WRITE_BIT;
          barrier.dstAccess |= VK_ACCESS_SHADER_WRITE_BIT;
        }
        barrier.tensorId = binding->tensor();
        bufferBarriers.push_back(barrier);
      }
      tensorStates[binding->tensor()] = nextState;
    }
  }
  if (bufferBarriers.empty()) {
    return std::nullopt;
  } else {
    return runtime::ModelBarrier{std::move(bufferBarriers), {}};
  }
}

static void collect_descriptor_pool_requirements(
    const dnx::ComputeDispatch *dispatch,
    ModelDescriptorPoolRequirements &requirements) {

  for (std::size_t s = 0; s < dispatch->bindings()->size(); ++s) {
    const dnx::DescriptorSetBinding *setBinding =
        dispatch->bindings()->Get(static_cast<unsigned int>(s));
    for (std::size_t b = 0; b < setBinding->bindings()->size(); ++b) {
      // const dnx::DescriptorBinding *binding =
      // setBinding->bindings()->Get(static_cast<unsigned int>(b));
      const auto it = std::find_if(
          requirements.poolSizes.begin(), requirements.poolSizes.end(),
          [&](const VkDescriptorPoolSize &poolSize) {
            return poolSize.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          });
      if (it == requirements.poolSizes.end()) {
        VkDescriptorPoolSize poolSize;
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 1;
        requirements.poolSizes.push_back(poolSize);
      } else {
        it->descriptorCount += 1;
      }
    }
  }
  requirements.maxSets += dispatch->bindings()->size();
}

static std::pair<ModelDescriptorPoolRequirements, memory::vector<ModelCmd>>
parse_cmds(const ContextHandle &context, const dnx::Model *dnx,
           memory::span<const ModelTensor> tensors) {

  ModelDescriptorPoolRequirements descriptorPoolRequirements{};
  memory::vector<ModelCmd> cmds;
  memory::vector<TensorState> tensorStates(dnx->tensors()->size(),
                                           TensorState::Undefined);
  for (size_t d = 0; d < dnx->dispatches()->size(); ++d) {
    const dnx::ComputeDispatch *dispatch =
        dnx->dispatches()->Get(static_cast<unsigned int>(d));
    ModelDispatch modelDispatch =
        create_model_dispatch(context, dnx, dispatch, tensors);

    if (std::optional<ModelBarrier> barrier =
            generate_pipeline_barrier(dnx, tensorStates, dispatch)) {
      cmds.emplace_back(*barrier);
    }
    cmds.emplace_back(modelDispatch);
    collect_descriptor_pool_requirements(dispatch, descriptorPoolRequirements);
  }
  return {descriptorPoolRequirements, cmds};
}

static SymIR parse_symir(const dnx::Model *dnx) {
  uint32_t varCount = dnx->sym_ir()->var_count();
  uint32_t opCount = dnx->sym_ir()->ops()->size();

  memory::vector<SymIROp> ops;
  ops.reserve(opCount);
  for (uint32_t i = 0; i < opCount; ++i) {
    const auto &op = dnx->sym_ir()->ops()->Get(i);
    const bool rhsc = op->opcode() & denox::dnx::SymIROpCode_RHSC;
    const bool lhsc = op->opcode() & denox::dnx::SymIROpCode_LHSC;
    const auto opcode =
        static_cast<std::underlying_type_t<denox::dnx::SymIROpCode>>(
            static_cast<int32_t>(op->opcode()) &
            ~(static_cast<int32_t>(denox::dnx::SymIROpCode_LHSC) |
              static_cast<int32_t>(denox::dnx::SymIROpCode_RHSC)));

    switch (opcode) {
    case denox::dnx::SymIROpCode_NOP:
      diag::invalid_state();
    case denox::dnx::SymIROpCode_ADD:
      if (!rhsc && !lhsc) {
        ops.emplace_back(denox::SymIROpCode::Add_SS, op->lhs(), op->rhs());
      } else if (rhsc) {
        ops.emplace_back(denox::SymIROpCode::Add_SC, op->lhs(), op->rhs());
      } else if (lhsc) {
        ops.emplace_back(denox::SymIROpCode::Add_SC, op->rhs(), op->lhs());
      } else {
        diag::invalid_state();
      }
      break;
    case denox::dnx::SymIROpCode_SUB:
      if (!rhsc && !lhsc) {
        ops.emplace_back(denox::SymIROpCode::Sub_SS, op->lhs(), op->rhs());
      } else if (rhsc) {
        ops.emplace_back(denox::SymIROpCode::Sub_SC, op->lhs(), op->rhs());
      } else if (lhsc) {
        ops.emplace_back(denox::SymIROpCode::Sub_CS, op->lhs(), op->rhs());
      } else {
        diag::invalid_state();
      }
      break;
    case denox::dnx::SymIROpCode_MUL:
      if (!rhsc && !lhsc) {
        ops.emplace_back(denox::SymIROpCode::Mul_SS, op->lhs(), op->rhs());
      } else if (rhsc) {
        ops.emplace_back(denox::SymIROpCode::Mul_SC, op->lhs(), op->rhs());
      } else if (lhsc) {
        ops.emplace_back(denox::SymIROpCode::Mul_SC, op->rhs(), op->lhs());
      } else {
        diag::invalid_state();
      }
      break;
    case denox::dnx::SymIROpCode_DIV:
      if (!rhsc && !lhsc) {
        ops.emplace_back(denox::SymIROpCode::Div_SS, op->lhs(), op->rhs());
      } else if (rhsc) {
        ops.emplace_back(denox::SymIROpCode::Div_SC, op->lhs(), op->rhs());
      } else if (lhsc) {
        ops.emplace_back(denox::SymIROpCode::Div_CS, op->lhs(), op->rhs());
      } else {
        diag::invalid_state();
      }
      break;
    case denox::dnx::SymIROpCode_MOD:
      if (!rhsc && !lhsc) {
        ops.emplace_back(denox::SymIROpCode::Mod_SS, op->lhs(), op->rhs());
      } else if (rhsc) {
        ops.emplace_back(denox::SymIROpCode::Mod_SC, op->lhs(), op->rhs());
      } else if (lhsc) {
        ops.emplace_back(denox::SymIROpCode::Mod_CS, op->lhs(), op->rhs());
      } else {
        diag::invalid_state();
      }
      break;
    case denox::dnx::SymIROpCode_MIN:
      if (!rhsc && !lhsc) {
        ops.emplace_back(denox::SymIROpCode::Min_SS, op->lhs(), op->rhs());
      } else if (rhsc) {
        ops.emplace_back(denox::SymIROpCode::Min_SC, op->lhs(), op->rhs());
      } else if (lhsc) {
        ops.emplace_back(denox::SymIROpCode::Min_SC, op->rhs(), op->lhs());
      } else {
        diag::invalid_state();
      }
      break;
    case denox::dnx::SymIROpCode_MAX:
      if (!rhsc && !lhsc) {
        ops.emplace_back(denox::SymIROpCode::Max_SS, op->lhs(), op->rhs());
      } else if (rhsc) {
        ops.emplace_back(denox::SymIROpCode::Max_SC, op->lhs(), op->rhs());
      } else if (lhsc) {
        ops.emplace_back(denox::SymIROpCode::Max_SC, op->rhs(), op->lhs());
      } else {
        diag::invalid_state();
      }
      break;
    default:
      diag::unreachable();
    }
  }
  return SymIR{
      .varCount = varCount,
      .ops = std::move(ops),
  };
}

static memory::vector<ModelTensor> parse_tensors(const dnx::Model *dnx) {
  memory::vector<ModelTensor> tensors;
  uint32_t tensorCount = dnx->tensors()->size();
  tensors.resize(tensorCount);
  for (uint32_t t = 0; t < tensorCount; ++t) {

    const auto *tensor = dnx->tensors()->Get(t);
    tensors[t].size = parse_sym(tensor->size_type(), tensor->size());
    tensors[t].offset = parse_sym(tensor->offset_type(), tensor->offset());
    tensors[t].buffer = tensor->buffer();

    if (tensor->info() != nullptr) {
      const auto *info = tensor->info();
      if (info->name()) {
        tensors[t].name = info->name()->str();
      }
      if (info->width()) {
        tensors[t].width = parse_sym(info->width_type(), info->width());
      }
      if (info->height()) {
        tensors[t].height = parse_sym(info->height_type(), info->height());
      }
      if (info->channels()) {
        tensors[t].channels =
            parse_sym(info->channels_type(), info->channels());
      }
      switch (info->storage()) {
      case dnx::TensorStorage_StorageBuffer:
        tensors[t].storage = TensorStorage::StorageBuffer;
        break;
      case dnx::TensorStorage_StorageImage:
        tensors[t].storage = TensorStorage::StorageImage;
        break;
      case dnx::TensorStorage_SampledStorageImage:
        tensors[t].storage = TensorStorage::SampledStorageImage;
        break;
      }
      switch (info->format()) {
      case dnx::TensorFormat_UNKNOWN:
        tensors[t].format = TensorFormat::Optimal;
        break;
      case dnx::TensorFormat_SSBO_HWC:
        tensors[t].format = TensorFormat::SSBO_HWC;
        break;
      case dnx::TensorFormat_SSBO_CHW:
        tensors[t].format = TensorFormat::SSBO_CHW;
        break;
      case dnx::TensorFormat_SSBO_CHWC8:
        tensors[t].format = TensorFormat::SSBO_CHWC8;
        break;
      case dnx::TensorFormat_TEX_RGBA:
        tensors[t].format = TensorFormat::TEX_RGBA;
        break;
      case dnx::TensorFormat_TEX_RGB:
        tensors[t].format = TensorFormat::TEX_RGB;
        break;
      case dnx::TensorFormat_TEX_RG:
        tensors[t].format = TensorFormat::TEX_RG;
        break;
      case dnx::TensorFormat_TEX_R:
        tensors[t].format = TensorFormat::TEX_R;
        break;
      }
      switch (info->type()) {
      case dnx::ScalarType_I16:
      case dnx::ScalarType_U16:
      case dnx::ScalarType_I32:
      case dnx::ScalarType_U32:
      case dnx::ScalarType_I64:
      case dnx::ScalarType_U64:
        diag::invalid_state();
      case dnx::ScalarType_F16:
        tensors[t].dtype = TensorDataType::Float16;
        break;
      case dnx::ScalarType_F32:
        tensors[t].dtype = TensorDataType::Float32;
        break;
      case dnx::ScalarType_F64:
        tensors[t].dtype = TensorDataType::Float64;
        break;
      }
    }

    tensors[t].isInput = std::find(dnx->inputs()->begin(), dnx->inputs()->end(),
                                   t) != dnx->inputs()->end();
    tensors[t].isOutput =
        std::find(dnx->outputs()->begin(), dnx->outputs()->end(), t) !=
        dnx->outputs()->end();
    tensors[t].isParam =
        std::find_if(dnx->initializers()->begin(), dnx->initializers()->end(),
                     [&](const dnx::TensorInitializer *init) {
                       return init->tensor() == t;
                     }) != dnx->initializers()->end();
  }
  return tensors;
}

static memory::vector<ModelBuffer> parse_buffers(const dnx::Model *dnx) {
  uint32_t bufferCount = dnx->buffers()->size();
  memory::vector<ModelBuffer> buffers(bufferCount);
  for (uint32_t b = 0; b < bufferCount; ++b) {
    const auto *buffer = dnx->buffers()->Get(b);
    buffers[b].size = parse_sym(buffer->size_type(), buffer->size());
    buffers[b].alignment = buffer->alignment();
    for (uint32_t t = 0; t < dnx->tensors()->size(); ++t) {
      if (dnx->tensors()->Get(t)->buffer() == b) {
        buffers[b].tensors.push_back(t);
      }
    }
  }
  return buffers;
}

static memory::vector<ModelInitializer>
parse_initializers(const dnx::Model *dnx) {
  uint32_t initCount = dnx->initializers()->size();
  memory::vector<ModelInitializer> initializers(initCount);
  for (uint32_t i = 0; i < initCount; ++i) {
    const auto &init = dnx->initializers()->Get(i);
    initializers[i].tensor = init->tensor();
    initializers[i].data.resize(init->data()->size());
    std::memcpy(initializers[i].data.data(),
                reinterpret_cast<const std::byte *>(init->data()->data()),
                init->data()->size());
  }
  return initializers;
}

static memory::vector<ValueName> parse_value_names(const dnx::Model *dnx) {
  uint32_t count = dnx->value_names()->size();
  memory::vector<ValueName> valueNames(count);
  for (uint32_t i = 0; i < count; ++i) {
    const auto &vn = dnx->value_names()->Get(i);
    valueNames[i].name = vn->name()->str();
    valueNames[i].value = parse_sym(vn->value_type(), vn->value());
  }
  return valueNames;
}

Model::Model(const ContextHandle &context, memory::span<const std::byte> dnxbuf)
    : m_context(context) {
  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t *>(dnxbuf.data()), dnxbuf.size());
  if (!denox::dnx::VerifyModelBuffer(verifier)) {
    DENOX_ERROR("Failed to verify dnx file format!");
    diag::invalid_state();
  }
  const dnx::Model *dnx = denox::dnx::GetModel(dnxbuf.data());

  m_symir = parse_symir(dnx);
  m_tensors = parse_tensors(dnx);
  m_buffers = parse_buffers(dnx);
  m_inputs.assign(dnx->inputs()->begin(), dnx->inputs()->end());
  m_outputs.assign(dnx->outputs()->begin(), dnx->outputs()->end());
  m_initializers = parse_initializers(dnx);

  auto [dpr, cmds] = parse_cmds(context, dnx, m_tensors);
  m_descriptorPoolRequirements = std::move(dpr);
  m_cmds = std::move(cmds);

  m_valueNames = parse_value_names(dnx);
}

Model::~Model() { release(); }

void Model::release() {
  for (const auto &cmd : m_cmds) {
    if (std::holds_alternative<ModelDispatch>(cmd)) {
      const auto &dispatch = std::get<ModelDispatch>(cmd);
      for (const auto &set : dispatch.descriptorSets) {
        m_context->destroyDescriptorSetLayout(set.descriptorSetLayout);
      }
      m_context->destroyPipelineLayout(dispatch.pipelineLayout);
      m_context->destroyPipeline(dispatch.pipeline);
    }
  }
  m_valueNames.clear();
  m_initializers.clear();
  m_tensors.clear();
  m_buffers.clear();
  m_inputs.clear();
  m_outputs.clear();
  m_descriptorPoolRequirements.poolSizes.clear();
  m_descriptorPoolRequirements.maxSets = 0;
  m_context = nullptr;
  m_cmds.clear();
  m_symir.varCount = 0;
  m_symir.ops.clear();
}

} // namespace denox::runtime

#include "context.hpp"
#include "denox/runtime.hpp"
#include "dnx.h"
#include "model.hpp"
#include "vma.hpp"
#include <algorithm>
#include <fmt/printf.h>
#include <stdexcept>
#include <type_traits>
#include <vulkan/vulkan_core.h>

namespace denox {

std::uint64_t parseUnsignedScalarLiteral(const dnx::ScalarLiteral *literal) {
  std::uint64_t value;
  switch (literal->dtype()) {
  case dnx::ScalarType_I16: {
    std::int16_t x;
    std::memcpy(&x, literal->bytes(), sizeof(std::int16_t));
    value = static_cast<std::size_t>(x);
    break;
  }
  case dnx::ScalarType_U16: {
    std::uint16_t x;
    std::memcpy(&x, literal->bytes(), sizeof(std::uint16_t));
    value = static_cast<std::size_t>(x);
    break;
  }
  case dnx::ScalarType_I32: {
    std::int32_t x;
    std::memcpy(&x, literal->bytes(), sizeof(std::int32_t));
    value = static_cast<std::size_t>(x);
    break;
  }
  case dnx::ScalarType_U32: {
    std::uint32_t x;
    std::memcpy(&x, literal->bytes(), sizeof(std::uint32_t));
    value = static_cast<std::size_t>(x);
    break;
  }
  case dnx::ScalarType_I64: {
    std::int64_t x;
    std::memcpy(&x, literal->bytes(), sizeof(std::int64_t));
    value = static_cast<std::size_t>(x);
    break;
  }
  case dnx::ScalarType_U64: {
    std::uint64_t x;
    std::memcpy(&x, literal->bytes()->data(), sizeof(std::uint64_t));
    value = static_cast<std::size_t>(x);
    break;
  }
  case dnx::ScalarType_F16:
  case dnx::ScalarType_F32:
  case dnx::ScalarType_F64:
    throw std::runtime_error("Invalid size dtype.");
  default:
    throw std::runtime_error("Unexpected size dtype.");
  }
  return value;
}

static uint64_t evalUnsignedScalarSource(runtime::ModelInstance *mi,
                                         dnx::ScalarSource type,
                                         const void *scalarSource) {
  switch (type) {
  case dnx::ScalarSource_NONE:
    throw std::runtime_error("invalid state.");
  case dnx::ScalarSource_literal:
    return parseUnsignedScalarLiteral(
        static_cast<const dnx::ScalarLiteral *>(scalarSource));
  case dnx::ScalarSource_symbolic:
    return mi->vars[static_cast<const dnx::SymRef *>(scalarSource)->sid()];
  }
}

int create_runtime_context(const char *deviceName, RuntimeContext *context) {
  auto *ctx = new runtime::Context(deviceName);
  *context = static_cast<void *>(ctx);
  return 0;
}

void destroy_runtime_context(RuntimeContext context) {
  auto *ctx = reinterpret_cast<runtime::Context *>(context);
  delete ctx;
}

int create_runtime_model(RuntimeContext context, const void *dnx,
                         size_t dnxSize, RuntimeModel *out_model) {
  flatbuffers::Verifier verifier(static_cast<const std::uint8_t *>(dnx),
                                 dnxSize);
  if (!denox::dnx::VerifyModelBuffer(verifier)) {
    throw std::runtime_error("Failed to verify dnx file format.");
  }
  void *dnxBuffer = malloc(dnxSize);
  std::memcpy(dnxBuffer, dnx, dnxSize);
  const dnx::Model *dnxModel = dnx::GetModel(dnxBuffer);
  auto *model = new runtime::Model(dnxBuffer, dnxModel);

  auto ctx = reinterpret_cast<runtime::Context *>(context);

  model->pipelines.resize(model->dnx->dispatches()->size());
  for (std::size_t d = 0; d < model->dnx->dispatches()->size(); ++d) {
    std::uint8_t dispatchType = model->dnx->dispatches_type()->Get(d);
    switch (dispatchType) {
    case dnx::Dispatch_ComputeDispatch: {
      const dnx::ComputeDispatch *dispatch =
          model->dnx->dispatches()->GetAs<dnx::ComputeDispatch>(d);
      std::vector<VkDescriptorSetLayout> descriptorSetLayouts(
          dispatch->bindings()->size());
      for (std::size_t b = 0; b < dispatch->bindings()->size(); ++b) {
        const dnx::DescriptorSetBinding *setBindings =
            dispatch->bindings()->Get(b);
        std::vector<VkDescriptorSetLayoutBinding> bindings(
            setBindings->bindings()->size());
        for (std::size_t i = 0; i < setBindings->bindings()->size(); ++i) {
          const dnx::DescriptorBinding *binding =
              setBindings->bindings()->Get(i);
          bindings[i].binding = static_cast<std::uint32_t>(binding->binding());
          bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
          bindings[i].descriptorCount = 1;
        }
        descriptorSetLayouts[b] = ctx->createDescriptorSetLayout(bindings);
        model->descriptorSetLayouts.push_back(descriptorSetLayouts[b]);
      }
      std::uint32_t pushConstantSize =
          static_cast<std::uint32_t>(dispatch->push_constant()->size());
      VkPipelineLayout layout =
          ctx->createPipelineLayout(descriptorSetLayouts, pushConstantSize);
      model->pipelineLayouts.push_back(layout);

      std::span<const std::uint32_t> binary{dispatch->spirv_src()->data(),
                                            dispatch->spirv_src()->size()};
      const char *entry_point = dispatch->entry_point()->c_str();
      VkPipeline pipeline =
          ctx->createComputePipeline(layout, binary, entry_point);
      model->pipelines[d] = pipeline;
      break;
    }
    case dnx::Dispatch_NONE:
      break;
    default:
      throw std::runtime_error("Unreachable switch case statment.");
    }
  }

  VkCommandPool cmdPool = ctx->createCommandPool();
  std::vector<runtime::Buffer> stagingBuffers;
  model->initalizedBuffers.reserve(model->dnx->initializers()->size());
  {
    VkCommandBuffer cmd = ctx->allocBeginCommandBuffer(cmdPool);

    for (std::size_t b = 0; b < model->dnx->initializers()->size(); ++b) {
      const dnx::BufferInitializer &bufferInitializers =
          *model->dnx->initializers()->Get(b);
      const dnx::Buffer &bufferInfo =
          *model->dnx->buffers()->Get(bufferInitializers.buffer());
      unsigned int alignment = bufferInfo.alignment();
      const dnx::ScalarLiteral *literal = bufferInfo.size_as_literal();
      std::uint64_t size = parseUnsignedScalarLiteral(literal);
      runtime::Buffer stage = ctx->createBuffer(
          size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

      ctx->copy(stage.allocation, bufferInitializers.data()->data(), size);

      runtime::Buffer local =
          ctx->createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

      ctx->cmdMemoryBarrier(
          cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

      ctx->cmdCopy(cmd, local, stage, size);

      model->initalizedBuffers.push_back(local);
      stagingBuffers.push_back(stage);
    }
    ctx->endSubmitWaitCommandBuffer(cmdPool, cmd);
  }
  for (const auto &buffer : stagingBuffers) {
    ctx->destroyBuffer(buffer);
  }
  stagingBuffers.clear();
  ctx->destroyCommandPool(cmdPool);
  *out_model = model;
  return 0;
}

void destroy_runtime_model(RuntimeContext context, RuntimeModel model) {
  assert(context != nullptr);
  assert(model != nullptr);
  auto ctx = reinterpret_cast<runtime::Context *>(context);
  auto m = reinterpret_cast<runtime::Model *>(model);
  for (VkDescriptorSetLayout layout : m->descriptorSetLayouts) {
    ctx->destroyDescriptorSetLayout(layout);
  }
  for (VkPipelineLayout layout : m->pipelineLayouts) {
    ctx->destroyPipelineLayout(layout);
  }
  for (VkPipeline pipeline : m->pipelines) {
    ctx->destroyPipeline(pipeline);
  }
  for (const auto &buffer : m->initalizedBuffers) {
    ctx->destroyBuffer(buffer);
  }
}

int create_runtime_model_instance(RuntimeContext context, RuntimeModel model,
                                  int dynamicExtentCount,
                                  DynamicExtent *dynamicExtents,
                                  RuntimeModelInstance *instance) {
  assert(context != nullptr);
  assert(model != nullptr);
  const auto m = reinterpret_cast<runtime::Model *>(model);
  auto ctx = reinterpret_cast<runtime::Context *>(context);

  auto mi = new runtime::ModelInstance();
  mi->model = m;
  { // Evaluate all symbolic expressions.
    const dnx::SymIR *ir = m->dnx->sym_ir();
    std::size_t dpsize = static_cast<std::size_t>(ir->var_count()) +
                         static_cast<std::size_t>(ir->ops()->size());
    mi->vars.resize(dpsize);
    for (std::size_t i = 0; i < dynamicExtentCount; ++i) {
      const DynamicExtent &extent = dynamicExtents[i];
      auto it = std::find_if(
          m->dnx->value_names()->begin(), m->dnx->value_names()->end(),
          [&extent](const dnx::ValueName *valueName) {
            return std::strcmp(valueName->name()->c_str(), extent.name) == 0;
          });
      if (it == m->dnx->value_names()->end()) {
        throw std::runtime_error(
            fmt::format("Dynamic extent {} does not exist.", extent.name));
      }
      const dnx::ValueName *valueName = *it;
      if (valueName->value_type() == dnx::ScalarSource_literal) {
        throw std::runtime_error(
            fmt::format("Dynamic extent {} is not dynamic.", extent.name));
      }
      const dnx::SymRef *symRef = valueName->value_as_symbolic();
      if (symRef->sid() >= ir->var_count()) {
        throw std::runtime_error(fmt::format(
            "Dynamic extent {} is not a dynamic variable.", extent.name));
      }
      std::uint32_t sid = symRef->sid();
      mi->vars[sid] = static_cast<std::int64_t>(extent.value);
    }
    for (std::uint64_t pc = 0; pc < ir->ops()->size(); ++pc) {
      std::uint64_t rx = pc + ir->var_count();
      const dnx::SymIROp *op = ir->ops()->Get(pc);
      std::int64_t lhs;
      if (op->opcode() & dnx::SymIROpCode_LHSC) {
        lhs = op->lhs();
      } else {
        lhs = mi->vars[static_cast<std::uint64_t>(op->lhs())];
      }
      std::int64_t rhs;
      if (op->opcode() & dnx::SymIROpCode_RHSC) {
        rhs = op->rhs();
      } else {
        rhs = mi->vars[static_cast<std::uint64_t>(op->rhs())];
      }
      std::underlying_type_t<dnx::SymIROpCode> opcode =
          op->opcode() & ~(dnx::SymIROpCode_LHSC | dnx::SymIROpCode_RHSC);

      switch (opcode) {
      case dnx::SymIROpCode_NOP:
        break;
      case dnx::SymIROpCode_ADD:
        mi->vars[rx] = lhs + rhs;
        break;
      case dnx::SymIROpCode_SUB:
        mi->vars[rx] = lhs - rhs;
        break;
      case dnx::SymIROpCode_MUL:
        mi->vars[rx] = lhs * rhs;
        break;
      case dnx::SymIROpCode_DIV:
        mi->vars[rx] = lhs / rhs;
        break;
      case dnx::SymIROpCode_MOD:
        mi->vars[rx] = (lhs % rhs);
        if (mi->vars[rx] < 0) {
          mi->vars[rx] += rhs;
        }
        break;
      case dnx::SymIROpCode_MIN:
        mi->vars[rx] = std::min(lhs, rhs);
        break;
      case dnx::SymIROpCode_MAX:
        mi->vars[rx] = std::max(lhs, rhs);
        break;
      default:
        throw std::runtime_error("Invalid sym instruction.");
      }
    }
  }

  { // Create (get) buffers.
    mi->buffers.resize(m->dnx->buffers()->size(),
                       runtime::Buffer{
                           .buffer = VK_NULL_HANDLE,
                           .allocation = VK_NULL_HANDLE,
                       });
    mi->ownedBuffers.resize(m->dnx->buffers()->size(), false);
    // Setup initalizers.
    for (std::size_t i = 0; i < m->dnx->initializers()->size(); ++i) {
      const dnx::BufferInitializer *init = m->dnx->initializers()->Get(i);
      mi->buffers[init->buffer()] = m->initalizedBuffers[i];
    }

    for (std::size_t b = 0; b < m->dnx->buffers()->size(); ++b) {
      if (mi->buffers[b].buffer != VK_NULL_HANDLE) {
        // skip initalized buffers.
        continue;
      }
      const dnx::Buffer *buffer = m->dnx->buffers()->Get(b);
      std::size_t size =
          evalUnsignedScalarSource(mi, buffer->size_type(), buffer->size());
      // See if the buffer is a input / output buffer.
      VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      for (std::size_t i = 0; i < m->dnx->inputs()->size(); ++i) {
        const dnx::Input *input = m->dnx->inputs()->Get(i);
        const dnx::Tensor *tensor = m->dnx->tensors()->Get(input->tensor());
        if (tensor->buffer() == b) {
          usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
          break;
        }
      }
      for (std::size_t o = 0; o < m->dnx->outputs()->size(); ++o) {
        const dnx::Output *output = m->dnx->outputs()->Get(o);
        const dnx::Tensor *tensor = m->dnx->tensors()->Get(output->tensor());
        if (tensor->buffer() == b) {
          usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
          break;
        }
      }
      mi->buffers[b] = ctx->createBuffer(size, usage);
      mi->ownedBuffers[b] = true;
    }
  }

  *instance = mi;
  return 0;
}

void destroy_runtime_model_instance(RuntimeContext context,
                                    RuntimeModelInstance instance) {
  auto ctx = reinterpret_cast<runtime::Context *>(context);
  auto mi = reinterpret_cast<runtime::ModelInstance *>(instance);
  for (std::size_t b = 0; b < mi->buffers.size(); ++b) {
    if (mi->ownedBuffers[b]) {
      ctx->destroyBuffer(mi->buffers[b]);
    }
  }
  delete mi;
}

int eval_runtime_model_instance(RuntimeContext context,
                                RuntimeModelInstance instance, void **inputs,
                                void **outputs) {
  auto ctx = reinterpret_cast<runtime::Context *>(context);
  auto mi = reinterpret_cast<runtime::ModelInstance *>(instance);
  auto m = mi->model;

  // 1. Allocate output.
  std::size_t outputCount = m->dnx->outputs()->size();
  std::size_t inputCount = m->dnx->inputs()->size();

  runtime::Buffer *outputStages = new runtime::Buffer[outputCount];
  std::size_t *outputSizes = new std::size_t[outputCount];
  for (std::size_t o = 0; o < m->dnx->outputs()->size(); ++o) {
    const dnx::Output *output = m->dnx->outputs()->Get(o);
    const dnx::Tensor *tensor = m->dnx->tensors()->Get(output->tensor());
    std::uint64_t size =
        evalUnsignedScalarSource(mi, tensor->size_type(), tensor->size());
    outputSizes[o] = size;
    outputStages[o] =
        ctx->createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
  }

  runtime::Buffer *inputStages = new runtime::Buffer[inputCount];
  for (std::size_t i = 0; i < static_cast<std::size_t>(inputCount); ++i) {
    const dnx::Input *input = m->dnx->inputs()->Get(i);
    const dnx::Tensor *tensor = m->dnx->tensors()->Get(input->tensor());
    std::uint64_t size =
        evalUnsignedScalarSource(mi, tensor->size_type(), tensor->size());
    inputStages[i] = ctx->createBuffer(
        size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    ctx->copy(inputStages[i].allocation, inputs[i], size);
  }

  VkCommandPool cmdPool = ctx->createCommandPool();
  VkCommandBuffer cmd = ctx->allocBeginCommandBuffer(cmdPool);

  // upload input.
  for (std::size_t i = 0; i < m->dnx->inputs()->size(); ++i) {
    const dnx::Input *input = m->dnx->inputs()->Get(i);
    const dnx::Tensor *tensor = m->dnx->tensors()->Get(input->tensor());
    std::size_t size =
        evalUnsignedScalarSource(mi, tensor->size_type(), tensor->size());
    std::size_t offset =
        evalUnsignedScalarSource(mi, tensor->offset_type(), tensor->offset());
    ctx->cmdCopy(cmd, mi->buffers[tensor->buffer()], inputStages[i], size,
                 offset, 0);
  }
  // Wait for transfer to be done.
  ctx->cmdMemoryBarrier(
      cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

  // Wait for compute to be done.
  ctx->cmdMemoryBarrier(
      cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
  // download output.
  for (std::size_t o = 0; o < m->dnx->outputs()->size(); ++o) {
    const dnx::Output *output = m->dnx->outputs()->Get(o);
    const dnx::Tensor *tensor = m->dnx->tensors()->Get(output->tensor());
    std::size_t size =
        evalUnsignedScalarSource(mi, tensor->size_type(), tensor->size());
    std::size_t offset =
        evalUnsignedScalarSource(mi, tensor->offset_type(), tensor->offset());
    // Wait for transfer to be done.
    ctx->cmdCopy(cmd, outputStages[o], mi->buffers[tensor->buffer()], size, 0,
                 offset);
  }
  ctx->cmdMemoryBarrier(
      cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);

  ctx->endSubmitWaitCommandBuffer(cmdPool, cmd);
  ctx->destroyCommandPool(cmdPool);

  for (std::size_t i = 0; i < inputCount; ++i) {
    ctx->destroyBuffer(inputStages[i]);
  }
  delete[] inputStages;

  for (std::size_t o = 0; o < outputCount; ++o) {
    ctx->copy(outputs[o], outputStages[o].allocation, outputSizes[o]);
    ctx->destroyBuffer(outputStages[o]);
  }
  delete[] outputStages;
  delete[] outputSizes;

  return 0;
}

int get_runtime_model_input_count(RuntimeModel model) {
  auto m = reinterpret_cast<runtime::Model *>(model);
  return m->dnx->inputs()->size();
}

int get_runtime_model_output_count(RuntimeModel model) {
  auto m = reinterpret_cast<runtime::Model *>(model);
  return m->dnx->outputs()->size();
}

void get_runtime_model_instance_input_shape(RuntimeModelInstance instance,
                                            int input_index, size_t *width,
                                            size_t *height, size_t *channels) {
  auto mi = reinterpret_cast<runtime::ModelInstance *>(instance);
  auto m = mi->model;
  const dnx::Input *input = m->dnx->inputs()->Get(input_index);
  if (width != nullptr) {
    *width = evalUnsignedScalarSource(mi, input->width_type(), input->width());
  }

  if (height != nullptr) {
    *height =
        evalUnsignedScalarSource(mi, input->height_type(), input->height());
  }

  if (channels != nullptr) {
    *channels =
        evalUnsignedScalarSource(mi, input->channels_type(), input->channels());
  }
}

void get_runtime_model_instance_output_shape(RuntimeModelInstance instance,
                                             int output_index, size_t *width,
                                             size_t *height, size_t *channels) {
  auto mi = reinterpret_cast<runtime::ModelInstance *>(instance);
  auto m = mi->model;

  const dnx::Output *output = m->dnx->outputs()->Get(output_index);
  if (width != nullptr) {
    *width =
        evalUnsignedScalarSource(mi, output->width_type(), output->width());
  }

  if (height != nullptr) {
    *height =
        evalUnsignedScalarSource(mi, output->height_type(), output->height());
  }

  if (channels != nullptr) {
    *channels = evalUnsignedScalarSource(mi, output->channels_type(),
                                         output->channels());
  }
}

void get_runtime_model_instance_input_byte_size(RuntimeModelInstance instance,
                                                int input_index,
                                                size_t *byteSize) {
  auto mi = reinterpret_cast<runtime::ModelInstance *>(instance);
  auto m = mi->model;

  const dnx::Input *input = m->dnx->inputs()->Get(input_index);
  const dnx::Tensor *tensor = m->dnx->tensors()->Get(input->tensor());
  *byteSize = evalUnsignedScalarSource(mi, tensor->size_type(), tensor->size());
}

void get_runtime_model_instance_output_byte_size(RuntimeModelInstance instance,
                                                 int output_index,
                                                 size_t *byteSize) {
  auto mi = reinterpret_cast<runtime::ModelInstance *>(instance);
  auto m = mi->model;

  const dnx::Output *output = m->dnx->outputs()->Get(output_index);
  const dnx::Tensor *tensor = m->dnx->tensors()->Get(output->tensor());
  *byteSize = evalUnsignedScalarSource(mi, tensor->size_type(), tensor->size());
}

} // namespace denox

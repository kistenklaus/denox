#include "context.hpp"
#include "denox/runtime.hpp"
#include "dnx.h"
#include "dnx_parse_helpers.hpp"
#include "instance.hpp"
#include "model.hpp"
#include "vma.hpp"
#include <algorithm>
#include <cstring>
#include <forward_list>
#include <stdexcept>
#include <variant>
#include <vulkan/vulkan_core.h>

namespace denox {

static std::vector<std::int64_t>
interpret_symir(const dnx::Model *dnx, std::span<const Extent> dynamicExtents) {
  std::size_t varCount = dnx->sym_ir()->var_count();
  std::size_t opCount = dnx->sym_ir()->ops()->size();
  std::vector<std::int64_t> dp(varCount + opCount);

  // Variable initialization:
  for (std::size_t i = 0; i < dynamicExtents.size(); ++i) {

    const auto [scalar_type, scalar_source] =
        dnx::getScalarSourceOfValueName(dnx, dynamicExtents[i].name);
    if (scalar_type == dnx::ScalarSource_NONE) {
      throw std::runtime_error(fmt::format(
          "dynamic extent \"{}\" does not exist.", dynamicExtents[i].name));
    }
    if (scalar_type != dnx::ScalarSource_symbolic) {
      continue;
    }
    std::uint32_t sid = static_cast<const dnx::SymRef *>(scalar_source)->sid();
    if (sid >= varCount) {
      continue;
    }
    dp[sid] = dynamicExtents[i].value;
  }

  for (std::uint64_t pc = 0; pc < opCount; ++pc) {
    const dnx::SymIROp *op = dnx->sym_ir()->ops()->Get(pc);
    std::uint64_t rx = pc + varCount;
    std::int64_t lhs;
    if (op->opcode() & dnx::SymIROpCode_LHSC) {
      lhs = op->lhs();
    } else {
      lhs = dp[static_cast<std::uint64_t>(op->lhs())];
    }
    std::int64_t rhs;
    if (op->opcode() & dnx::SymIROpCode_RHSC) {
      rhs = op->rhs();
    } else {
      rhs = dp[static_cast<std::uint64_t>(op->rhs())];
    }

    std::underlying_type_t<dnx::SymIROpCode> opcode =
        op->opcode() & ~(dnx::SymIROpCode_LHSC | dnx::SymIROpCode_RHSC);
    switch (opcode) {
    case dnx::SymIROpCode_NOP:
      break;
    case dnx::SymIROpCode_ADD:
      dp[rx] = lhs + rhs;
      break;
    case dnx::SymIROpCode_SUB:
      dp[rx] = lhs - rhs;
      break;
    case dnx::SymIROpCode_MUL:
      dp[rx] = lhs * rhs;
      break;
    case dnx::SymIROpCode_DIV:
      dp[rx] = lhs / rhs;
      break;
    case dnx::SymIROpCode_MOD: {
      dp[rx] = lhs % rhs;
      if (dp[rx] < 0) {
        dp[rx] += rhs;
      }
      break;
    }
    case dnx::SymIROpCode_MIN:
      dp[rx] = std::min(lhs, rhs);
      break;
    case dnx::SymIROpCode_MAX:
      dp[rx] = std::max(lhs, rhs);
      break;
    default:
      throw std::runtime_error("Invalid sym instruction.");
    }
  }
  return dp;
}

template <typename InOutput>
static runtime::InstanceTensorInfo
parse_tensor_info(const dnx::Model *dnx, const InOutput *tensorInfo,
                  std::span<const std::int64_t> symbolValues) {
  runtime::InstanceTensorInfo info;
  info.name = tensorInfo->name()->c_str();
  info.tensor = tensorInfo->tensor();
  info.channels.value = dnx::parseUnsignedScalarSource(
      tensorInfo->channels_type(), tensorInfo->channels(), symbolValues);
  info.channels.name = dnx::reverse_value_name_search(dnx, tensorInfo->channels_type(), tensorInfo->channels());

  info.width.value = dnx::parseUnsignedScalarSource(
      tensorInfo->width_type(), tensorInfo->width(), symbolValues);
  info.width.name = dnx::reverse_value_name_search(dnx, tensorInfo->width_type(), tensorInfo->width());
  info.height.value = dnx::parseUnsignedScalarSource(
      tensorInfo->height_type(), tensorInfo->height(), symbolValues);
  info.height.name = dnx::reverse_value_name_search(dnx, tensorInfo->height_type(), tensorInfo->height());
  return info;
}

static std::vector<runtime::InstanceBuffer>
create_buffers(runtime::Context *ctx, const runtime::Model *m,
               const runtime::Instance *mi) {
  const std::span<const std::int64_t> symbolValues = mi->symbolValues;
  const dnx::Model *dnx = m->dnx;
  std::size_t bufferCount = dnx->buffers()->size();
  std::vector<runtime::InstanceBuffer> buffers(bufferCount);

  for (std::size_t b = 0; b < bufferCount; ++b) {
    const dnx::Buffer *buffer = dnx->buffers()->Get(b);

    std::uint64_t size = dnx::parseUnsignedScalarSource(
        buffer->size_type(), buffer->size(), symbolValues);

    // Check if input:
    bool isInput = std::find_if(mi->inputs.begin(), mi->inputs.end(),
                                [&](const runtime::InstanceTensorInfo &info) {
                                  const dnx::Tensor *tensor =
                                      m->dnx->tensors()->Get(info.tensor);
                                  return tensor->buffer() == b;
                                }) != mi->inputs.end();
    bool isOutput = std::find_if(mi->outputs.rbegin(), mi->outputs.rend(),
                                 [&](const runtime::InstanceTensorInfo &info) {
                                   const dnx::Tensor *tensor =
                                       m->dnx->tensors()->Get(info.tensor);
                                   return tensor->buffer() == b;
                                 }) != mi->outputs.rend();
    bool isInitialized =
        std::find_if(dnx->initializers()->begin(), dnx->initializers()->end(),
                     [b](const dnx::BufferInitializer *initalizer) {
                       return initalizer->buffer() == b;
                     }) != dnx->initializers()->end();

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (isInput || isInitialized) {
      usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }
    if (isOutput) {
      usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    }

    runtime::Buffer runtimeBuffer = ctx->createBuffer(size, usage);
    runtime::InstanceBuffer instanceBuffer;
    instanceBuffer.buffer = runtimeBuffer;
    instanceBuffer.size = size;
    buffers[b] = instanceBuffer;
  }

  return buffers;
}

static void upload_initalizers(runtime::Context *ctx,
                               const runtime::Instance *mi) {
  const auto *dnx = mi->model->dnx;
  std::size_t initalizerCount = dnx->initializers()->size();

  std::vector<runtime::Buffer> stagingBuffers(initalizerCount);

  for (std::size_t i = 0; i < initalizerCount; ++i) {
    const dnx::BufferInitializer *initalizer = dnx->initializers()->Get(i);
    const void *data = initalizer->data()->data();
    const runtime::InstanceBuffer &buffer = mi->buffers[initalizer->buffer()];
    // See issue #48, currently buffer initalizers only allow initalization of
    // full buffers.
    assert(initalizer->data()->size() == buffer.size);
    runtime::Buffer stage = ctx->createBuffer(
        buffer.size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    ctx->copy(stage.allocation, data, buffer.size);
    stagingBuffers[i] = stage;
  }

  VkCommandPool cmdPool = ctx->createCommandPool();
  VkCommandBuffer cmd = ctx->allocBeginCommandBuffer(cmdPool);
  ctx->cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

  for (std::size_t i = 0; i < initalizerCount; ++i) {
    const dnx::BufferInitializer *initalizer = dnx->initializers()->Get(i);
    const auto &stage = stagingBuffers[i];
    const runtime::InstanceBuffer &buffer = mi->buffers[initalizer->buffer()];
    ctx->cmdCopy(cmd, buffer.buffer, stage, buffer.size);
  }

  ctx->endSubmitWaitCommandBuffer(cmdPool, cmd);
  ctx->destroyCommandPool(cmdPool);

  for (const runtime::Buffer &buffer : stagingBuffers) {
    ctx->destroyBuffer(buffer);
  }
}

static std::vector<runtime::InstanceTensor>
create_tensors(const runtime::Model *m, const runtime::Instance *mi) {
  std::span<const std::int64_t> symbolValues = mi->symbolValues;
  const dnx::Model *dnx = m->dnx;
  std::size_t tensorCount = dnx->tensors()->size();
  std::vector<runtime::InstanceTensor> tensors(tensorCount);

  for (std::size_t t = 0; t < tensorCount; ++t) {
    const dnx::Tensor *tensor = dnx->tensors()->Get(t);
    std::uint64_t size = dnx::parseUnsignedScalarSource(
        tensor->size_type(), tensor->size(), symbolValues);
    std::uint64_t offset = dnx::parseUnsignedScalarSource(
        tensor->offset_type(), tensor->offset(), symbolValues);

    runtime::InstanceTensor instanceTensor;
    instanceTensor.size = size;
    instanceTensor.offset = offset;
    instanceTensor.buffer = tensor->buffer();
    tensors[t] = instanceTensor;
  }
  return tensors;
}

static VkDescriptorPool create_descriptor_pool(runtime::Context *ctx,
                                               const runtime::Model *m) {
  return ctx->createDescriptorPool(m->descriptorPoolRequirements.maxSets,
                                   m->descriptorPoolRequirements.poolSizes);
}

static std::vector<runtime::InstanceCmd>
create_cmds(runtime::Context *ctx, const runtime::Model *m,
            const runtime::Instance *mi) {
  std::span<const std::int64_t> symbolValues = mi->symbolValues;
  const dnx::Model *dnx = m->dnx;
  std::uint64_t cmdCount = m->cmds.size();
  std::vector<runtime::InstanceCmd> cmds;
  cmds.reserve(cmdCount);

  for (std::size_t pc = 0; pc < cmdCount; ++pc) {
    const auto &cmd = m->cmds[pc];
    if (std::holds_alternative<runtime::ModelBarrier>(cmd)) {
      const runtime::ModelBarrier barrier =
          std::get<runtime::ModelBarrier>(cmd);
      assert(barrier.imageMemoryBarriers.empty()); // not yet implemented.
      runtime::InstanceBarrier instanceBarrier;
      std::vector<VkBufferMemoryBarrier> bufferMemoryBarriers(
          barrier.bufferBarriers.size());
      for (std::size_t b = 0; b < barrier.bufferBarriers.size(); ++b) {
        const runtime::ModelBufferBarrier &bufferBarrier =
            barrier.bufferBarriers[b];
        const runtime::InstanceTensor &tensor =
            mi->tensors[bufferBarrier.tensorId];
        const runtime::InstanceBuffer &buffer = mi->buffers[tensor.buffer];

        VkBufferMemoryBarrier &out = bufferMemoryBarriers[b];
        out.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        out.pNext = nullptr;
        out.srcAccessMask = bufferBarrier.srcAccess;
        out.dstAccessMask = bufferBarrier.dstAccess;
        out.srcQueueFamilyIndex = ctx->getQueueFamily();
        out.dstQueueFamilyIndex = ctx->getQueueFamily();
        out.buffer = buffer.buffer.vkbuffer;
        out.offset = tensor.offset;
        out.size = tensor.size;
      }
      cmds.emplace_back(instanceBarrier);
    } else if (std::holds_alternative<runtime::ModelDispatch>(cmd)) {
      const runtime::ModelDispatch &dispatch =
          std::get<runtime::ModelDispatch>(cmd);
      runtime::InstanceDispatch instanceDispatch;
      instanceDispatch.dispatch = &dispatch;
      std::vector<VkDescriptorSetLayout> descriptorSetLayouts(
          dispatch.descriptorSets.size());
      for (std::size_t s = 0; s < descriptorSetLayouts.size(); ++s) {
        descriptorSetLayouts[s] =
            dispatch.descriptorSets[s].descriptorSetLayout;
      }
      instanceDispatch.descriptorSets =
          new VkDescriptorSet[dispatch.descriptorSets.size()];
      ctx->allocDescriptorSets(mi->descriptorPool, descriptorSetLayouts,
                               instanceDispatch.descriptorSets);

      // Parse workgroup count (i.e. dispatch size).
      instanceDispatch.workgroupCounts = new std::uint32_t[3];
      instanceDispatch.workgroupCounts[0] = dnx::parseUnsignedScalarSource(
          dispatch.dispatch->workgroup_count_x_type(),
          dispatch.dispatch->workgroup_count_x(), symbolValues);
      instanceDispatch.workgroupCounts[1] = dnx::parseUnsignedScalarSource(
          dispatch.dispatch->workgroup_count_y_type(),
          dispatch.dispatch->workgroup_count_y(), symbolValues);
      instanceDispatch.workgroupCounts[2] = dnx::parseUnsignedScalarSource(
          dispatch.dispatch->workgroup_count_z_type(),
          dispatch.dispatch->workgroup_count_z(), symbolValues);

      // Build push constant buffer.
      std::size_t pushConstantRange =
          dispatch.dispatch->push_constant()->size();
      void *pushConstantBuffer = malloc(pushConstantRange);
      for (std::size_t f = 0;
           f < dispatch.dispatch->push_constant()->fields()->size(); ++f) {
        const dnx::PushConstantField *field =
            dispatch.dispatch->push_constant()->fields()->Get(f);
        dnx::memcpyPushConstantField(pushConstantBuffer, field, symbolValues);
      }
      instanceDispatch.pushConstantValues = pushConstantBuffer;

      cmds.emplace_back(instanceDispatch);
    } else {
      throw std::runtime_error("invalid ModelCmd.");
    }
  }

  return cmds;
}

void update_descriptor_sets(runtime::Context *ctx,
                            const runtime::Instance *mi) {
  std::forward_list<VkDescriptorBufferInfo>
      bufferInfos; // simply holds storage.
  std::vector<VkWriteDescriptorSet> writeInfos;
  for (std::size_t pc = 0; pc < mi->cmds.size(); ++pc) {
    const auto &cmd = mi->cmds[pc];
    if (std::holds_alternative<runtime::InstanceDispatch>(cmd)) {
      const runtime::InstanceDispatch &dispatch =
          std::get<runtime::InstanceDispatch>(cmd);
      const runtime::ModelDispatch &modelDispatch = *dispatch.dispatch;
      const dnx::ComputeDispatch *dnxDispatch = modelDispatch.dispatch;
      for (std::size_t s = 0; s < dnxDispatch->bindings()->size(); ++s) {
        const dnx::DescriptorSetBinding *setBinding =
            dnxDispatch->bindings()->Get(s);
        for (std::size_t b = 0; b < setBinding->bindings()->size(); ++b) {
          const dnx::DescriptorBinding *binding =
              setBinding->bindings()->Get(b);
          const runtime::InstanceTensor &tensor =
              mi->tensors[binding->tensor()];
          const runtime::InstanceBuffer &buffer = mi->buffers[tensor.buffer];

          VkWriteDescriptorSet writeInfo;
          writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
          writeInfo.pNext = nullptr;
          writeInfo.dstSet = dispatch.descriptorSets[s];
          writeInfo.dstBinding = binding->binding();
          writeInfo.dstArrayElement = 0;
          writeInfo.descriptorCount = 1; // array elements.
          writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          VkDescriptorBufferInfo bufferInfo;
          bufferInfo.buffer = buffer.buffer.vkbuffer;
          bufferInfo.offset = tensor.offset;
          bufferInfo.range = tensor.size;
          bufferInfos.push_front(bufferInfo);
          writeInfo.pBufferInfo = &bufferInfos.front();
          writeInfos.push_back(writeInfo);
        }
      }
    }
  }
  ctx->updateDescriptorSets(writeInfos);
}

int create_runtime_instance(RuntimeContext context, RuntimeModel model,
                            int dynamicExtentCount, Extent *dynamicExtents,
                            RuntimeInstance *instance) {
  assert(context);
  assert(model);
  runtime::Context *ctx = reinterpret_cast<runtime::Context *>(context);
  const runtime::Model *m = reinterpret_cast<runtime::Model *>(model);
  const dnx::Model *dnx = m->dnx;

  runtime::Instance *mi = new runtime::Instance(m);

  // 1. Interpret symbolic ir, based on the dynamicExtent specializations.
  mi->symbolValues = interpret_symir(
      dnx,
      std::span{dynamicExtents, static_cast<std::size_t>(dynamicExtentCount)});

  // 2. Collect input & outputs.
  for (std::size_t i = 0; i < dnx->inputs()->size(); ++i) {
    mi->inputs.push_back(
        parse_tensor_info(dnx, dnx->inputs()->Get(i), mi->symbolValues));
  }
  for (std::size_t o = 0; o < dnx->outputs()->size(); ++o) {
    mi->outputs.push_back(
        parse_tensor_info(dnx, dnx->outputs()->Get(o), mi->symbolValues));
  }

  // 3. Create buffers and parse tensor views.
  mi->buffers = create_buffers(ctx, m, mi);
  mi->tensors = create_tensors(m, mi);
  upload_initalizers(ctx, mi);

  mi->descriptorPool = create_descriptor_pool(ctx, m);

  mi->cmds = create_cmds(ctx, m, mi);

  update_descriptor_sets(ctx, mi);

  *instance = mi;
  return 0;
}

void destroy_runtime_instance(RuntimeContext context,
                              RuntimeInstance instance) {
  assert(context);
  assert(instance);
  runtime::Context *ctx = static_cast<runtime::Context *>(context);
  runtime::Instance *mi = static_cast<runtime::Instance *>(instance);

  for (std::size_t d = 0; d < mi->cmds.size(); ++d) {
    const auto &cmd = mi->cmds[d];
    if (std::holds_alternative<runtime::InstanceDispatch>(cmd)) {
      const auto &dispatch = std::get<runtime::InstanceDispatch>(cmd);
      std::size_t setCount = dispatch.dispatch->descriptorSets.size();
      delete[] dispatch.descriptorSets;
      delete[] dispatch.workgroupCounts;
      free(dispatch.pushConstantValues);
    }
  }

  ctx->destroyDescriptorPool(mi->descriptorPool);

  for (std::size_t b = 0; b < mi->buffers.size(); ++b) {
    ctx->destroyBuffer(mi->buffers[b].buffer);
  }

  delete mi;
}

} // namespace denox

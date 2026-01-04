#include "instance.hpp"
#include "context.hpp"
#include "denox/diag/not_implemented.hpp"
#include "denox/symbolic/SymIR.hpp"
#include "model.hpp"
#include "vma.hpp"
#include <algorithm>
#include <cstring>
#include <fmt/format.h>
#include <forward_list>
#include <iterator>
#include <stdexcept>
#include <variant>
#include <vulkan/vulkan_core.h>

namespace denox {

static std::vector<runtime::Buffer>
create_buffers(const runtime::ModelHandle &model, const SymIREval &symeval) {
  const auto modelBuffers = model->buffers();
  std::size_t bufferCount = modelBuffers.size();

  std::vector<runtime::Buffer> buffers(bufferCount);

  for (std::size_t b = 0; b < bufferCount; ++b) {
    const runtime::ModelBuffer &buffer = modelBuffers[b];

    uint64_t size = static_cast<uint64_t>(symeval[buffer.size]);

    bool isInput =
        std::ranges::find_if(buffer.tensors, [&](const uint32_t &tensor) {
          return model->tensors()[tensor].isInput;
        }) != buffer.tensors.end();

    bool isOutput =
        std::ranges::find_if(buffer.tensors, [&](const uint32_t &tensor) {
          return model->tensors()[tensor].isOutput;
        }) != buffer.tensors.end();

    bool isParam =
        std::ranges::find_if(buffer.tensors, [&](const uint32_t &tensor) {
          return model->tensors()[tensor].isParam;
        }) != buffer.tensors.end();

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (isInput || isParam) {
      usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }
    if (isOutput) {
      usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    }

    buffers[b] = model->context()->createBuffer(size, usage);
  }
  return buffers;
}

static void upload_initalizers(const runtime::ModelHandle &model,
                               const SymIREval &symeval,
                               memory::span<const runtime::Buffer> buffers) {
  const auto initializers = model->initializers();
  std::size_t initalizerCount = initializers.size();

  const runtime::ContextHandle &ctx = model->context();

  std::vector<runtime::Buffer> stagingBuffers(initalizerCount);

  for (std::size_t i = 0; i < initalizerCount; ++i) {
    const runtime::ModelInitializer &init = initializers[i];
    stagingBuffers[i] = ctx->createBuffer(
        init.data.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    ctx->copy(stagingBuffers[i].allocation, init.data.data(), init.data.size());
  }

  // tmp command pools.
  VkCommandPool cmdPool = ctx->createCommandPool();
  VkCommandBuffer cmd = ctx->allocBeginCommandBuffer(cmdPool);
  ctx->cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

  for (std::size_t i = 0; i < initalizerCount; ++i) {
    const auto &init = initializers[i];
    const runtime::ModelTensor tensor = model->tensors()[init.tensor];
    const uint64_t offset = static_cast<uint64_t>(symeval[tensor.offset]);
    // sort of assumed here, but might also work without it, but preferable
    // the initializers should have the same size as the tensor beeing
    // initialized.
    assert(static_cast<size_t>(symeval[tensor.size]) == init.data.size());
    ctx->cmdCopy(cmd, buffers[tensor.buffer], stagingBuffers[i],
                 init.data.size(), offset, 0);
  }

  ctx->endSubmitWaitCommandBuffer(cmdPool, cmd);
  ctx->destroyCommandPool(cmdPool);

  for (const runtime::Buffer &buffer : stagingBuffers) {
    ctx->destroyBuffer(buffer);
  }
}

static VkDescriptorPool create_descriptor_pool(runtime::ModelHandle &model) {
  return model->context()->createDescriptorPool(
      model->descriptorPoolRequirements().maxSets,
      model->descriptorPoolRequirements().poolSizes);
}

static std::vector<runtime::InstanceCmd>
create_cmds(const runtime::ModelHandle &model, const SymIREval &symeval,
            memory::span<const runtime::Buffer> buffers,
            VkDescriptorPool descriptorPool) {
  const runtime::ContextHandle &ctx = model->context();

  const auto modelCmds = model->cmds();
  uint32_t cmdCount = static_cast<uint32_t>(modelCmds.size());

  memory::vector<runtime::InstanceCmd> cmds;
  cmds.reserve(cmdCount * 2);

  for (std::size_t pc = 0; pc < cmdCount; ++pc) {
    const auto &cmd = modelCmds[pc];
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

        const runtime::ModelTensor &tensor =
            model->tensors()[bufferBarrier.tensorId];
        uint64_t offset = static_cast<uint64_t>(symeval[tensor.offset]);
        uint64_t size = static_cast<uint64_t>(symeval[tensor.size]);

        const runtime::Buffer &buffer = buffers[tensor.buffer];

        VkBufferMemoryBarrier &out = bufferMemoryBarriers[b];
        out.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        out.pNext = nullptr;
        out.srcAccessMask = bufferBarrier.srcAccess;
        out.dstAccessMask = bufferBarrier.dstAccess;
        out.srcQueueFamilyIndex = ctx->getQueueFamily();
        out.dstQueueFamilyIndex = ctx->getQueueFamily();
        out.buffer = buffer.vkbuffer;
        out.offset = offset;
        out.size = size;
      }
      cmds.emplace_back(instanceBarrier);
    } else if (std::holds_alternative<runtime::ModelDispatch>(cmd)) {
      const runtime::ModelDispatch &dispatch =
          std::get<runtime::ModelDispatch>(cmd);
      runtime::InstanceDispatch instanceDispatch;
      instanceDispatch.modelDispatch = &dispatch;

      std::vector<VkDescriptorSetLayout> descriptorSetLayouts(
          dispatch.descriptorSets.size());
      for (std::size_t s = 0; s < descriptorSetLayouts.size(); ++s) {
        descriptorSetLayouts[s] =
            dispatch.descriptorSets[s].descriptorSetLayout;
      }

      instanceDispatch.descriptorSets.resize(dispatch.descriptorSets.size());

      ctx->allocDescriptorSets(descriptorPool, descriptorSetLayouts,
                               instanceDispatch.descriptorSets.data());

      instanceDispatch.workgroupCountX =
          static_cast<uint32_t>(symeval[dispatch.workgroupCountX]);
      instanceDispatch.workgroupCountY =
          static_cast<uint32_t>(symeval[dispatch.workgroupCountY]);
      instanceDispatch.workgroupCountZ =
          static_cast<uint32_t>(symeval[dispatch.workgroupCountZ]);

      // Build push constant buffer.
      uint16_t pcRange = dispatch.pushConstantRange;
      instanceDispatch.pc.resize(pcRange);
      for (const auto &[offset, pc] : dispatch.pushConstants) {
        int64_t value = symeval[pc.sym()];
        switch (pc.type().kind()) {
        case memory::DtypeKind::F16:
        case memory::DtypeKind::F32:
        case memory::DtypeKind::F64:
          diag::not_implemented();
        case memory::DtypeKind::U32: {
          uint32_t x = static_cast<uint32_t>(value);
          std::memcpy(instanceDispatch.pc.data() + offset, &x, 4);
          break;
        }
        case memory::DtypeKind::I32: {
          int32_t x = static_cast<int32_t>(value);
          std::memcpy(instanceDispatch.pc.data() + offset, &x, 4);
          break;
        }
        case memory::DtypeKind::U64: {
          uint64_t x = static_cast<uint64_t>(value);
          std::memcpy(instanceDispatch.pc.data() + offset, &x, 8);
          break;
        }
        case memory::DtypeKind::I64: {
          std::memcpy(instanceDispatch.pc.data() + offset, &value, 8);
          break;
        }
        }
      }

      cmds.emplace_back(instanceDispatch);
    } else {
      throw std::runtime_error("invalid ModelCmd.");
    }
  }

  return cmds;
}

void update_descriptor_sets(const runtime::ModelHandle &model,
                            const SymIREval &symeval,
                            memory::span<const runtime::Buffer> buffers,
                            memory::span<const runtime::InstanceCmd> cmds) {
  std::forward_list<VkDescriptorBufferInfo> monotone_allocator;

  std::vector<VkWriteDescriptorSet> writeInfos;
  for (std::size_t pc = 0; pc < cmds.size(); ++pc) {
    const auto &cmd = cmds[pc];
    if (std::holds_alternative<runtime::InstanceDispatch>(cmd)) {
      const runtime::InstanceDispatch &dispatch =
          std::get<runtime::InstanceDispatch>(cmd);
      const runtime::ModelDispatch &modelDispatch = *dispatch.modelDispatch;

      for (std::size_t s = 0; s < modelDispatch.descriptorSets.size(); ++s) {
        for (const auto &binding : modelDispatch.descriptorSets[s].bindings) {
          const runtime::ModelTensor &tensor = model->tensors()[binding.tensor];
          const uint64_t offset = static_cast<uint64_t>(symeval[tensor.offset]);
          const uint64_t size = static_cast<uint64_t>(symeval[tensor.size]);

          VkWriteDescriptorSet writeInfo;
          writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
          writeInfo.pNext = nullptr;
          writeInfo.dstSet = dispatch.descriptorSets[s];
          writeInfo.dstBinding = binding.binding;
          writeInfo.dstArrayElement = 0;
          writeInfo.descriptorCount = 1; // array elements.
          writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          VkDescriptorBufferInfo bufferInfo;
          bufferInfo.buffer = buffers[tensor.buffer].vkbuffer,
          bufferInfo.offset = offset;
          bufferInfo.range = size;
          monotone_allocator.push_front(bufferInfo);
          writeInfo.pBufferInfo = &monotone_allocator.front();
          writeInfo.pImageInfo = nullptr;
          writeInfo.pTexelBufferView = nullptr;
          writeInfos.push_back(writeInfo);
        }
      }
    }
  }
  model->context()->updateDescriptorSets(writeInfos);
}

runtime::Instance::Instance(const ModelHandle &model,
                            memory::span<const SymSpec> specs)
    : m_model(model), m_symeval(model->symir().eval(specs)) {
  m_buffers = create_buffers(m_model, m_symeval);
  upload_initalizers(m_model, m_symeval, m_buffers);
  m_descriptorPool = create_descriptor_pool(m_model);
  m_cmds = create_cmds(m_model, m_symeval, m_buffers, m_descriptorPool);
  update_descriptor_sets(m_model, m_symeval, m_buffers, m_cmds);
}

runtime::Instance::~Instance() { release(); }

void runtime::Instance::release() {
  const auto &ctx = m_model->context();
  assert(ctx);
  ctx->destroyDescriptorPool(m_descriptorPool);
  for (const auto &buffer : m_buffers) {
    ctx->destroyBuffer(buffer);
  }
  m_model = nullptr;
  m_descriptorPool = VK_NULL_HANDLE;
  m_cmds.clear();
  m_buffers.clear();
}

std::shared_ptr<runtime::Instance>
runtime::Instance::make(const ModelHandle &model,
                        memory::span<const ValueSpec> specs) {
  memory::vector<SymSpec> symSpecs;
  for (const auto &spec : specs) {
    auto it =
        std::ranges::find_if(model->valueNames(), [&](const ValueName &vn) {
          return vn.name == spec.name;
        });
    if (it == model->valueNames().end()) {
      continue;
    }
    if (it->value.isSymbolic()) {
      symSpecs.emplace_back(it->value.sym(), spec.value);
    }
  }
  return make(model, symSpecs);
}

void runtime::Instance::infer(void *const *pInputs, void **pOutputs) const {
  const auto &ctx = m_model->context();
  const auto inputs = m_model->inputs();
  const auto outputs = m_model->outputs();

  memory::vector<runtime::Buffer> inputStages(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    const runtime::ModelTensor &tensor = m_model->tensors()[inputs[i]];
    const uint64_t size = static_cast<uint64_t>(m_symeval[tensor.size]);
    inputStages[i] = ctx->createBuffer(
        size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    ctx->copy(inputStages[i].allocation, pInputs[i], size);
  }

  memory::vector<runtime::Buffer> outputStages(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    const runtime::ModelTensor &tensor = m_model->tensors()[outputs[i]];
    const uint64_t size = static_cast<uint64_t>(m_symeval[tensor.size]);

    outputStages[i] =
        ctx->createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
    ctx->copy(outputStages[i].allocation, pOutputs[i], size);
  }

  VkCommandPool cmdPool = ctx->createCommandPool();
  VkCommandBuffer cmd = ctx->allocBeginCommandBuffer(cmdPool);

  ctx->cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

  for (size_t i = 0; i < inputs.size(); ++i) {
    const runtime::ModelTensor &tensor = m_model->tensors()[inputs[i]];
    const uint64_t size = static_cast<uint64_t>(m_symeval[tensor.size]);
    const uint64_t offset = static_cast<uint64_t>(m_symeval[tensor.offset]);
    ctx->cmdCopy(cmd, m_buffers[tensor.buffer], inputStages[i], size, offset,
                 0);
  }

  ctx->cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

  for (const auto &op : m_cmds) {
    if (std::holds_alternative<runtime::InstanceDispatch>(op)) {
      const auto &dispatch = std::get<runtime::InstanceDispatch>(op);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                        dispatch.modelDispatch->pipeline);
      vkCmdPushConstants(cmd, dispatch.modelDispatch->pipelineLayout,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         static_cast<uint32_t>(dispatch.pc.size()),
                         dispatch.pc.data());
      vkCmdBindDescriptorSets(
          cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          dispatch.modelDispatch->pipelineLayout, 0,
          static_cast<uint32_t>(dispatch.descriptorSets.size()),
          dispatch.descriptorSets.data(), 0, nullptr);
      vkCmdDispatch(cmd, dispatch.workgroupCountX, dispatch.workgroupCountY,
                    dispatch.workgroupCountZ);
    } else if (std::holds_alternative<runtime::InstanceBarrier>(op)) {
      const auto &barrier = std::get<runtime::InstanceBarrier>(op);
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                           static_cast<uint32_t>(barrier.bufferBarrier.size()),
                           barrier.bufferBarrier.data(), 0, nullptr);
    }
  }

  ctx->cmdMemoryBarrier(
      cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

  for (size_t i = 0; i < outputs.size(); ++i) {
    const runtime::ModelTensor &tensor = m_model->tensors()[outputs[i]];
    const uint64_t size = static_cast<uint64_t>(m_symeval[tensor.size]);
    const uint64_t offset = static_cast<uint64_t>(m_symeval[tensor.offset]);
    ctx->cmdCopy(cmd, outputStages[i], m_buffers[tensor.buffer], size, 0,
                 offset);
  }

  ctx->cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);

  ctx->endSubmitWaitCommandBuffer(cmdPool, cmd);
  ctx->destroyCommandPool(cmdPool);

  for (size_t i = 0; i < outputs.size(); ++i) {
    const runtime::ModelTensor &tensor = m_model->tensors()[outputs[i]];
    const uint64_t size = static_cast<uint64_t>(m_symeval[tensor.size]);
    const uint64_t offset = static_cast<uint64_t>(m_symeval[tensor.offset]);
    ctx->copy(pOutputs[i], outputStages[i].allocation, size, offset);
  }

  for (const auto &b : inputStages) {
    ctx->destroyBuffer(b);
  }
  for (const auto &b : outputStages) {
    ctx->destroyBuffer(b);
  }
}

runtime::InstanceBenchmarkResult runtime::Instance::bench() const {
  const auto &ctx = m_model->context();

  const uint32_t queryCount = static_cast<uint32_t>(m_cmds.size()) * 2 + 1;
  VkQueryPool queryPool = ctx->createTimestampQueryPool(queryCount);

  VkCommandPool cmdPool = ctx->createCommandPool();
  VkCommandBuffer cmd = ctx->allocBeginCommandBuffer(cmdPool);

  ctx->cmdResetQueryPool(cmd, queryPool, 0, queryCount);

  struct Timing {
    memory::string input_desc;
    memory::string output_desc;
    memory::string name;
    uint32_t start;
    uint32_t end;
    uint64_t memoryReads;
    uint64_t memoryWrites;
  };

  memory::vector<Timing> timings;
  uint32_t previousTimestamp = 0;
  ctx->cmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queryPool,
                         previousTimestamp);

  ctx->cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

  for (const auto &op : m_cmds) {
    if (std::holds_alternative<runtime::InstanceDispatch>(op)) {
      const auto &dispatch = std::get<runtime::InstanceDispatch>(op);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                        dispatch.modelDispatch->pipeline);
      vkCmdPushConstants(cmd, dispatch.modelDispatch->pipelineLayout,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         static_cast<uint32_t>(dispatch.pc.size()),
                         dispatch.pc.data());
      vkCmdBindDescriptorSets(
          cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          dispatch.modelDispatch->pipelineLayout, 0,
          static_cast<uint32_t>(dispatch.descriptorSets.size()),
          dispatch.descriptorSets.data(), 0, nullptr);
      vkCmdDispatch(cmd, dispatch.workgroupCountX, dispatch.workgroupCountY,
                    dispatch.workgroupCountZ);
      uint32_t start = previousTimestamp;
      uint32_t end = ++previousTimestamp;
      ctx->cmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queryPool,
                             end);

      memory::vector<uint32_t> inputs;
      memory::vector<uint32_t> outputs;
      for (const auto &set : dispatch.modelDispatch->descriptorSets) {
        for (const auto &binding : set.bindings) {
          const auto &tensor = m_model->tensors()[binding.tensor];
          if (tensor.isParam) {
            continue;
          }
          if (binding.access == Access::ReadOnly ||
              binding.access == Access::ReadWrite) {
            inputs.push_back(binding.tensor);
          }
          if (binding.access == Access::WriteOnly ||
              binding.access == Access::ReadWrite) {
            outputs.push_back(binding.tensor);
          }
        }
      }

      const auto format_tensor = [](const runtime::ModelTensor &tensor,
                                    const SymIREval &symeval) {
        memory::string chan;
        if (tensor.channels.has_value()) {
          chan = fmt::format("{}", symeval[*tensor.channels]);
        } else {
          chan = "?";
        }

        if (tensor.format.has_value()) {
          return fmt::format("{}[{}]", *tensor.format, chan);
        } else {
          return fmt::format("UNI[{}]", chan);
        }
      };

      memory::string input_desc;
      if (inputs.size() == 1) {
        input_desc =
            format_tensor(m_model->tensors()[inputs.front()], m_symeval);
      } else {
        input_desc = "{";
        for (uint32_t i = 0; i < inputs.size(); ++i) {
          if (i != 0) {
            input_desc.push_back(',');
          }
          input_desc += format_tensor(m_model->tensors()[inputs[i]], m_symeval);
        }
        input_desc.push_back('}');
      }

      memory::string output_desc;

      if (outputs.size() == 1) {
        output_desc =
            format_tensor(m_model->tensors()[outputs.front()], m_symeval);
      } else {
        output_desc = "{";
        for (uint32_t i = 0; i < outputs.size(); ++i) {
          if (i != 0) {
            output_desc.push_back(',');
          }
          output_desc +=
              format_tensor(m_model->tensors()[outputs[i]], m_symeval);
        }
        output_desc.push_back('}');
      }

      size_t memoryReads = 0;
      size_t memoryWrites = 0;
      if (dispatch.modelDispatch->memoryReads) {
        memoryReads = static_cast<uint64_t>(
            m_symeval[*dispatch.modelDispatch->memoryReads]);
      }
      if (dispatch.modelDispatch->memoryWrites) {
        memoryWrites = static_cast<uint64_t>(
            m_symeval[*dispatch.modelDispatch->memoryWrites]);
      }

      timings.push_back(Timing{
          .input_desc = input_desc,
          .output_desc = output_desc,
          .name = dispatch.modelDispatch->name.value_or("unnamed"),
          .start = start,
          .end = end,
          .memoryReads = memoryReads,
          .memoryWrites = memoryWrites,
      });
    } else if (std::holds_alternative<runtime::InstanceBarrier>(op)) {
      const auto &barrier = std::get<runtime::InstanceBarrier>(op);
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                           static_cast<uint32_t>(barrier.bufferBarrier.size()),
                           barrier.bufferBarrier.data(), 0, nullptr);
    }
  }

  ctx->cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

  ++previousTimestamp;
  ctx->cmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queryPool,
                         previousTimestamp);

  ctx->cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
  ++previousTimestamp;

  ctx->endSubmitWaitCommandBuffer(cmdPool, cmd);
  ctx->destroyCommandPool(cmdPool);
  memory::vector<uint64_t> timestamps =
      ctx->getQueryResults(queryPool, previousTimestamp);

  ctx->destroyQueryPool(queryPool);

  memory::vector<InstanceBenchmarkResult::Timing> out(timings.size());
  for (size_t i = 0; i < timings.size(); ++i) {
    const auto &t = timings[i];

    out[i].name = t.name;
    out[i].input = t.input_desc;
    out[i].output = t.output_desc;
    out[i].memoryWrites = t.memoryWrites;
    out[i].memoryReads = t.memoryReads;
    float duration = ctx->timestampDifference(timestamps, t.start, t.end);
    out[i].latency = std::chrono::duration<float, std::milli>{duration};
  }
  return InstanceBenchmarkResult{.timings = std::move(out)};
}

memory::string runtime::InstanceBenchmarkResult::report() const {
  memory::string out;
  uint64_t totalAccesses = 0;
  float totalDuration = 0;
  for (const auto &t : timings) {
    auto s = fmt::format("{:.3f}", t.latency.count());
    auto pos = s.find('.');
    auto int_part = s.substr(0, pos);
    auto frac_part = s.substr(pos + 1);
    totalDuration += t.latency.count();
    uint64_t accesses = t.memoryReads + t.memoryWrites;
    totalAccesses += accesses;
    float throughput =
        (static_cast<float>(accesses) / (t.latency.count() * 1e-3f)) * 1e-9f;
    fmt::println("{:>22} \x1B[34m{:-^40}>\x1B[0m {:<22} :{:>3}.{:<3}ms "
                 "\x1B[90m({:>4} GB/s)\x1B[0m    {:.3f}MB",
                 t.input, t.name, t.output, int_part, frac_part,
                 static_cast<uint32_t>(std::round(throughput)),
                 static_cast<float>(accesses) * 1e-6f);
  }

  fmt::println("\x1B[31m{:>89} {:.3f}ms\x1B[0m \x1B[90m({:>4} GB/s)\x1B[0m",
               "Total time :", totalDuration,
               std::round((static_cast<float>(totalAccesses) / (totalDuration * 1e-3f)) *
                   1e-9f));
  return out;
}

} // namespace denox

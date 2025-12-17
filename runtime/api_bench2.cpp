
#include "Db.hpp"
#include "context.hpp"
#include "denox/runtime.hpp"
#include "shader/GlslCompiler.hpp"
#include "shader/ShaderBinary.hpp"
#include "vma.hpp"
#include <absl/strings/str_format.h>
#include <algorithm>
#include <alloca.h>
#include <chrono>
#include <cmath>
#include <fmt/base.h>
#include <fmt/format.h>
#include <forward_list>
#include <glslang/SPIRV/GlslangToSpv.h>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <unordered_set>
#include <vulkan/vulkan_core.h>
using namespace std::chrono_literals;

namespace denox {

static constexpr std::chrono::duration MIN_GLOBAL_WARMUP = 10s;
static constexpr size_t BATCH_SIZE = 10;
static constexpr size_t JIT_WARMUP_ITERATIONS = 3;
static constexpr size_t L2_WARMUP_ITERATIONS = 50;

static constexpr size_t PIPELINE_STAGES = 2;

inline std::size_t align_up(std::size_t offset,
                            std::size_t alignment) noexcept {
  assert(alignment && (alignment & (alignment - 1)) == 0 &&
         "alignment must be power of two");
  return (offset + alignment - 1) & ~(alignment - 1);
}

std::vector<runtime::ComputeDispatch *> select_targets(runtime::Db &db,
                                                       size_t N) {
  constexpr uint64_t INITIAL_SAMPLES = 5;

  std::random_device rng;
  std::mt19937 prng(rng());
  std::vector<runtime::ComputeDispatch *> cold;

  for (auto &dispatch : db.dispatches) {
    if (dispatch.time.samples < INITIAL_SAMPLES) {
      cold.push_back(&dispatch);
      if (cold.size() == N) {
        std::ranges::shuffle(cold.begin(), cold.end(), prng);
        return cold;
      }
    }
  }
  if (!cold.empty()) {
    return cold;
  }

  struct Item {
    runtime::ComputeDispatch *dispatch;
    double priority;
  };

  std::vector<Item> items;
  items.reserve(db.dispatches.size());

  for (auto &dispatch : db.dispatches) {
    double n = double(dispatch.time.samples);
    double sigma = double(dispatch.time.std_derivation_ns);

    double sem = (n > 0.0) ? (sigma / std::sqrt(n))
                           : std::numeric_limits<double>::infinity();

    items.push_back({&dispatch, sem});
  }

  if (N > items.size()) {
    N = items.size();
  }
  std::partial_sort(items.begin(), items.begin() + N, items.end(),
                    [](const Item &a, const Item &b) {
                      return a.priority > b.priority; // descending
                    });

  std::vector<runtime::ComputeDispatch *> result;
  result.reserve(N);

  for (auto *d : cold) {
    result.push_back(d);
  }

  for (size_t i = 0; i < N - cold.size(); ++i) {
    result.push_back(items[i].dispatch);
  }

  std::ranges::shuffle(result.begin(), result.end(), prng);

  return result;
}

static std::pair<VkPipelineLayout, std::map<uint32_t, VkDescriptorSetLayout>>
create_compute_dispatch_layouts(runtime::Context *ctx,
                                const runtime::ComputeDispatch &dispatch) {

  std::map<uint32_t, std::vector<VkDescriptorSetLayoutBinding>> bindings;
  for (const runtime::TensorBinding &binding : dispatch.bindings) {
    VkDescriptorSetLayoutBinding layoutBinding{};
    layoutBinding.binding = binding.binding;
    layoutBinding.descriptorCount = 1;
    layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[binding.set].push_back(layoutBinding);
  }

  std::map<uint32_t, VkDescriptorSetLayout> descriptorSetLayouts;
  std::vector<VkDescriptorSetLayout> layouts;
  for (const auto &[set, binding] : bindings) {
    VkDescriptorSetLayout layout = ctx->createDescriptorSetLayout(binding);
    layouts.push_back(layout);
    descriptorSetLayouts.emplace(set, layout);
  }

  VkPipelineLayout pipelineLayout =
      ctx->createPipelineLayout(layouts, dispatch.pushConstant.size());
  return std::make_pair(pipelineLayout, descriptorSetLayouts);
}
static VkPipeline create_compute_dispatch_pipeline(
    runtime::Context *ctx, const runtime::ComputeDispatch &dispatch,
    const runtime::Db &db, VkPipelineLayout pipelineLayout) {
  VkPipeline pipeline = ctx->createComputePipeline(
      pipelineLayout, db.binaries[dispatch.binaryId].source, "main");
  return pipeline;
}

static VkDescriptorPool
create_descriptor_pool(runtime::Context *ctx,
                       const runtime::ComputeDispatch &dispatch) {
  std::set<uint32_t> sets;
  for (const auto &binding : dispatch.bindings) {
    sets.insert(binding.set);
  }
  std::vector<VkDescriptorPoolSize> sizes;
  VkDescriptorPoolSize storageBufferDescriptors;
  storageBufferDescriptors.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  storageBufferDescriptors.descriptorCount = dispatch.bindings.size();
  sizes.push_back(storageBufferDescriptors);
  return ctx->createDescriptorPool(sets.size(), sizes);
}

static std::vector<runtime::Buffer>
allocate_buffers(runtime::Context *ctx,
                 const runtime::ComputeDispatch &dispatch) {
  std::vector<runtime::Buffer> buffers;
  buffers.reserve(dispatch.bindings.size());
  for (const runtime::TensorBinding &tensorBinding : dispatch.bindings) {
    if (tensorBinding.access == runtime::Access::ReadOnly ||
        tensorBinding.access == runtime::Access::WriteOnly) {
      runtime::Buffer buffer = ctx->createBuffer(
          tensorBinding.byteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT);
      buffers.push_back(buffer);
    } else {
      runtime::Buffer buffer = ctx->createBuffer(
          tensorBinding.byteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT);
      buffers.push_back(buffer);
    }
  }
  return buffers;
}

static std::vector<VkDescriptorSet> create_and_update_descriptor_sets(
    runtime::Context *ctx, VkDescriptorPool pool,
    const std::map<uint32_t, VkDescriptorSetLayout> &layouts,
    const runtime::ComputeDispatch &dispatch, const runtime::Buffer &buffer) {

  std::vector<uint32_t> setOrder;
  std::vector<VkDescriptorSetLayout> layoutsVec;
  for (const auto &[set, layout] : layouts) {
    layoutsVec.push_back(layout);
    setOrder.push_back(set);
  }
  std::vector<VkDescriptorSet> setVec(layouts.size());
  ctx->allocDescriptorSets(pool, layoutsVec, setVec.data());
  std::map<uint32_t, VkDescriptorSet> sets;
  for (size_t i = 0; i < setOrder.size(); ++i) {
    sets.emplace(setOrder[i], setVec[i]);
  }

  std::forward_list<VkDescriptorBufferInfo> bufferInfosArena;

  std::vector<VkWriteDescriptorSet> writeInfos;
  size_t offset = 0;
  for (size_t i = 0; i < dispatch.bindings.size(); ++i) {
    const auto &tensorBinding = dispatch.bindings[i];
    VkWriteDescriptorSet writeInfo{};
    writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeInfo.pNext = nullptr;
    writeInfo.dstSet = sets[tensorBinding.set];
    writeInfo.dstBinding = tensorBinding.binding;
    writeInfo.dstArrayElement = 0;
    writeInfo.descriptorCount = 1;
    writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

    auto &bufferInfo = bufferInfosArena.emplace_front();
    offset = align_up(offset, tensorBinding.minAlignment);
    bufferInfo.buffer = buffer.vkbuffer;
    bufferInfo.range = tensorBinding.byteSize;
    bufferInfo.offset = offset;
    offset += tensorBinding.byteSize;
    writeInfo.pBufferInfo = &bufferInfo;

    writeInfos.push_back(writeInfo);
  }

  ctx->updateDescriptorSets(writeInfos);
  return setVec;
}

struct WarmupKernel {
  VkPipeline pipeline;
  VkPipelineLayout pipelineLayout;
  VkDescriptorPool descriptorPool;
  VkDescriptorSetLayout descriptorSetLayout;
  VkDescriptorSet descriptorSet;
  runtime::Buffer buffer;
};

static WarmupKernel create_warmup_kernel(runtime::Context *ctx) {
  WarmupKernel kernel;

  glsl::GlslCompiler glslCompiler;
  glsl::ShaderBinary shaderBinary =
      *glslCompiler.read("./runtime/shader/warmup.comp").compile();

  VkDescriptorPoolSize size;
  size.descriptorCount = 1;
  size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  kernel.descriptorPool = ctx->createDescriptorPool(1, {&size, 1});

  VkDescriptorSetLayoutBinding binding{};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  kernel.descriptorSetLayout = ctx->createDescriptorSetLayout({&binding, 1});

  kernel.pipelineLayout =
      ctx->createPipelineLayout({&kernel.descriptorSetLayout, 1}, 0);
  kernel.pipeline = ctx->createComputePipeline(kernel.pipelineLayout,
                                               shaderBinary.spv, "main");

  kernel.descriptorSet = ctx->allocDescriptorSet(kernel.descriptorPool,
                                                 kernel.descriptorSetLayout);

  constexpr size_t WARMUP_BUFFER_SIZE = 48000000;
  kernel.buffer =
      ctx->createBuffer(WARMUP_BUFFER_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

  VkWriteDescriptorSet writeInfo{};
  writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writeInfo.pNext = nullptr;
  writeInfo.dstSet = kernel.descriptorSet;
  writeInfo.dstBinding = 0;
  writeInfo.dstArrayElement = 0;
  writeInfo.descriptorCount = 1;
  writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  VkDescriptorBufferInfo bufferInfo{};
  bufferInfo.buffer = kernel.buffer.vkbuffer;
  bufferInfo.offset = 0;
  bufferInfo.range = WARMUP_BUFFER_SIZE;
  writeInfo.pBufferInfo = &bufferInfo;
  ctx->updateDescriptorSets({&writeInfo, 1});

  return kernel;
}

static void cmdWarmup(runtime::Context *ctx, VkCommandBuffer cmd,
                      const WarmupKernel &kernel) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          kernel.pipelineLayout, 0, 1, &kernel.descriptorSet, 0,
                          nullptr);
  vkCmdDispatch(cmd, 1000, 100, 10);
}

static void warmup(runtime::Context *ctx, const WarmupKernel &kernel,
                   VkCommandPool cmdPool,
                   std::chrono::duration<long> duration) {

  auto start = std::chrono::high_resolution_clock::now();
  while (std::chrono::high_resolution_clock::now() - start < duration) {
    VkCommandBuffer cmd = ctx->allocBeginCommandBuffer(cmdPool);
    cmdWarmup(ctx, cmd, kernel);
    ctx->endSubmitWaitCommandBuffer(cmdPool, cmd);
  }
}

static void destroy_warmup_kernel(runtime::Context *ctx, WarmupKernel kernel) {
  ctx->destroyBuffer(kernel.buffer);
  ctx->destroyPipeline(kernel.pipeline);
  ctx->destroyPipelineLayout(kernel.pipelineLayout);
  ctx->destroyDescriptorSetLayout(kernel.descriptorSetLayout);
  ctx->destroyDescriptorPool(kernel.descriptorPool);
}

struct BenchDispatch {
  runtime::ComputeDispatch *dispatch;
  VkPipeline pipeline;
  VkPipelineLayout pipelineLayout;
  std::map<uint32_t, VkDescriptorSetLayout> descriptorSetLayouts;
  std::vector<VkDescriptorSet> descriptorSets;
};

struct Batch {
  std::vector<BenchDispatch> targets;
  runtime::Buffer buffer;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkQueryPool queryPool = VK_NULL_HANDLE;
};

static Batch create_batch(runtime::Context *ctx, runtime::Db &db) {
  std::vector<runtime::ComputeDispatch *> targets =
      select_targets(db, BATCH_SIZE);

  Batch batch;
  size_t peakBufferSize = 0;
  size_t setCount = 0;
  size_t ssboBindingCount = 0;
  for (size_t i = 0; i < targets.size(); ++i) {
    auto *target = targets[i];
    size_t memoryRequirements = 0;
    std::unordered_set<uint32_t> usedSets;
    for (size_t b = 0; b < target->bindings.size(); ++b) {
      const auto &binding = target->bindings[b];
      memoryRequirements = align_up(memoryRequirements, binding.minAlignment);
      memoryRequirements += binding.byteSize;
      usedSets.insert(binding.set);
    }
    ssboBindingCount += target->bindings.size();
    setCount += usedSets.size();
    peakBufferSize = std::max(peakBufferSize, memoryRequirements);
  }

  batch.buffer =
      ctx->createBuffer(peakBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  VkDescriptorPoolSize poolSize;
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = ssboBindingCount;

  batch.descriptorPool = ctx->createDescriptorPool(setCount, {&poolSize, 1});

  batch.targets.reserve(targets.size());
  for (size_t i = 0; i < targets.size(); ++i) {
    auto *target = targets[i];
    BenchDispatch bench;

    auto [pipelineLayout, _descriptorSetLayouts] =
        create_compute_dispatch_layouts(ctx, *target);
    bench.dispatch = target;
    bench.pipelineLayout = pipelineLayout;
    bench.descriptorSetLayouts = std::move(_descriptorSetLayouts);

    bench.pipeline =
        create_compute_dispatch_pipeline(ctx, *target, db, pipelineLayout);

    bench.descriptorSets = create_and_update_descriptor_sets(
        ctx, batch.descriptorPool, bench.descriptorSetLayouts, *target,
        batch.buffer);

    batch.targets.push_back(bench);
  }
  batch.queryPool = ctx->createTimestampQueryPool(targets.size() * 2);

  return batch;
}

static void destroy_batch(runtime::Context *ctx, Batch &batch) {
  ctx->destroyQueryPool(batch.queryPool);
  for (const auto &target : batch.targets) {
    ctx->destroyPipeline(target.pipeline);
    ctx->destroyPipelineLayout(target.pipelineLayout);
    for (const auto &[_, layout] : target.descriptorSetLayouts) {
      ctx->destroyDescriptorSetLayout(layout);
    }
  }
  batch.targets.clear();
  ctx->destroyBuffer(batch.buffer);
  ctx->destroyDescriptorPool(batch.descriptorPool);
}

static void read_batch(runtime::Context *ctx, Batch &batch) {
  const size_t N = batch.targets.size();
  const size_t QUERY_COUNT = N * 2;

  std::vector<uint64_t> timestamps =
      ctx->getQueryResults(batch.queryPool, QUERY_COUNT);

  for (size_t i = 0; i < N; ++i) {
    auto &bench = batch.targets[i];
    auto *dispatch = bench.dispatch;

    const uint32_t t0 = uint32_t(i * 2 + 0);
    const uint32_t t1 = uint32_t(i * 2 + 1);

    if (timestamps[t1] <= timestamps[t0]) {
      fmt::println(
          "\x1B[33m[WARNING\x1B[33m] Timestamps wrapped, skipping dispatch {}",
          i);
      continue;
    }

    uint64_t totalLatency = ctx->timestampNanoDifference(timestamps, t0, t1);

    if (totalLatency > 1e11) { // ~100ms wrap protection
      fmt::println("\x1B[33m[WARNING\x1B[33m] Timestamp wrap detected, "
                   "skipping dispatch {}",
                   i);
      continue;
    }

    const double mean_new = double(totalLatency);
    constexpr double var_new = 0.0;
    constexpr double m = 1.0;

    uint64_t old_samples = dispatch->time.samples;

    if (old_samples > 0) {
      double n = double(old_samples);

      double mean_old = double(dispatch->time.latency_ns);
      double std_old = double(dispatch->time.std_derivation_ns);
      double var_old = std_old * std_old;

      double mean_total = (n * mean_old + m * mean_new) / (n + m);

      double var_total =
          (n * (var_old + (mean_old - mean_total) * (mean_old - mean_total)) +
           m * (var_new + (mean_new - mean_total) * (mean_new - mean_total))) /
          (n + m);

      dispatch->time.samples = uint64_t(n + m);

      dispatch->time.latency_ns = uint64_t(mean_total + 0.5);
      dispatch->time.std_derivation_ns = uint64_t(std::sqrt(var_total) + 0.5);

    } else {
      dispatch->time.samples = 1;
      dispatch->time.latency_ns = uint64_t(mean_new + 0.5);
      dispatch->time.std_derivation_ns = 0;
    }
  }
}

static void record_dispatch(runtime::Context *ctx, VkCommandBuffer cmd,
                            BenchDispatch &target) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, target.pipeline);
  vkCmdBindDescriptorSets(
      cmd, VK_PIPELINE_BIND_POINT_COMPUTE, target.pipelineLayout, 0,
      target.descriptorSets.size(), target.descriptorSets.data(), 0, nullptr);
  vkCmdPushConstants(cmd, target.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     target.dispatch->pushConstant.size(),
                     target.dispatch->pushConstant.data());

  vkCmdDispatch(cmd, target.dispatch->workgroupCountX,
                target.dispatch->workgroupCountY,
                target.dispatch->workgroupCountZ);
}

static void record_batch(runtime::Context *ctx, VkCommandBuffer cmd,
                         Batch &batch, WarmupKernel &warmupKernel) {
  cmdWarmup(ctx, cmd, warmupKernel); // <- rewarm a bit, takes around 100ms
  // let jit kick in.
  for (size_t t = 0; t < batch.targets.size(); ++t) {
    auto &target = batch.targets[t];
    for (size_t i = 0; i < JIT_WARMUP_ITERATIONS; ++i) {
      record_dispatch(ctx, cmd, target);
    }
  }
  ctx->cmdResetQueryPool(cmd, batch.queryPool, 0, batch.targets.size() * 2);

  for (size_t t = 0; t < batch.targets.size(); ++t) {
    auto &target = batch.targets[t];
    for (size_t i = 0; i < L2_WARMUP_ITERATIONS; ++i) {
      record_dispatch(ctx, cmd, target);
    }
    ctx->cmdMemoryBarrierComputeShader(cmd);

    ctx->cmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           batch.queryPool, t * 2);
    record_dispatch(ctx, cmd, target);
    ctx->cmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           batch.queryPool, t * 2 + 1);

    ctx->cmdMemoryBarrierComputeShader(cmd);
  }
}

int bench_runtime_instance(RuntimeContext context, const char *dbfile,
                           size_t minSamples) {
  auto *ctx = static_cast<runtime::Context *>(context);
  WarmupKernel warmupKernel = create_warmup_kernel(ctx);

  auto db = runtime::Db::open(dbfile);
  if (db.dispatches.empty()) {
    return 0;
  }

  fmt::println("\x1B[34mWarming up\x1B[0m");

  VkCommandPool cmdPool = ctx->createCommandPool();
  warmup(ctx, warmupKernel, cmdPool, MIN_GLOBAL_WARMUP);

  std::array<VkCommandBuffer, PIPELINE_STAGES> cmds;
  for (size_t i = 0; i < PIPELINE_STAGES; ++i) {
    cmds[i] = ctx->allocCommandBuffer(cmdPool);
  }

  std::array<VkFence, PIPELINE_STAGES> fences;
  for (size_t i = 0; i < PIPELINE_STAGES; ++i) {
    fences[i] = ctx->createFence(true);
  }

  std::array<Batch, PIPELINE_STAGES> batches;

  int stage = PIPELINE_STAGES - 1;

  while (true) {
    int next = (stage + 1) % PIPELINE_STAGES;
    ctx->waitFence(fences[next]);
    ctx->resetFence(fences[next]);

    if (!batches[next].targets.empty()) {
      read_batch(ctx, batches[next]);
      uint64_t total_samples = 0;
      for (const auto& dispatch : db.dispatches) {
        total_samples += dispatch.time.samples;
      }

      db.write_back();

      uint64_t maxVar = 0;
      double maxSEM = 0;
      uint64_t minSamples = std::numeric_limits<uint64_t>::max();
      uint64_t maxSamples = 0;
      for (size_t i = 0; i < db.dispatches.size(); ++i) {
        maxVar = std::max(db.dispatches[i].time.std_derivation_ns, maxVar);
        minSamples = std::min(db.dispatches[i].time.samples, minSamples);
        maxSamples = std::max(db.dispatches[i].time.samples, maxSamples);
        double SEM = db.dispatches[i].time.std_derivation_ns /
            std::sqrt(db.dispatches[i].time.samples);
        maxSEM = std::max(maxSEM, SEM);
      }
      fmt::println("max-std: {}ms", maxVar * 1e-6);
      fmt::println("samples: {} / {}", total_samples, db.dispatches.size() * 6);
      fmt::println("SEM    : {}", static_cast<uint64_t>(maxSEM));

      destroy_batch(ctx, batches[next]);
    }
    batches[next] = create_batch(ctx, db);

    ctx->beginCommandBuffer(cmds[next]);
    record_batch(ctx, cmds[next], batches[next], warmupKernel);
    ctx->endCommandBuffer(cmds[next]);

    ctx->submit(cmds[next], fences[next]);
    stage = next;
  }

  ctx->destroyCommandPool(cmdPool);

  destroy_warmup_kernel(ctx, warmupKernel);
  return 0;
}

} // namespace denox

#include "denox/runtime/db.hpp"
#include "denox/algorithm/align_up.hpp"
#include "denox/db/DbComputeDispatch.hpp"
#include "denox/device_info/query/query_driver_device_info.hpp"
#include "denox/glsl/GlslCompiler.hpp"
#include "denox/memory/container/hashmap.hpp"
#include "denox/memory/container/small_vector.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/runtime/context.hpp"
#include "denox/spirv/SpirvBinary.hpp"
#include "denox/spirv/SpirvTools.hpp"
#include <algorithm>
#include <chrono>
#include <fmt/ostream.h>
#include <forward_list>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vulkan/vulkan_core.h>

// 100, 1000, 100 with jit=2 worked well, but didn't converge completely.


static constexpr size_t EPOCH_SIZE = 100;
static constexpr size_t EPOCH_SAMPLES = EPOCH_SIZE * 10;
static constexpr size_t BATCH_SIZE = EPOCH_SIZE; 

static constexpr size_t JIT_WARMUP_ITERATIONS = 0;
static constexpr size_t L2_WARMUP_ITERATONS = 10;
static constexpr size_t PIPELINE_STAGES = 2;

using namespace denox;

struct BenchmarkState {
  std::mt19937 prng;
  std::unique_ptr<spirv::SpirvTools> tools;
  std::unique_ptr<spirv::GlslCompiler> glslCompiler;
};

static BenchmarkState
create_benchmark_state(const runtime::ContextHandle &ctx) {
  std::random_device rng;
  std::mt19937 prng(rng());

  auto device_info = denox::query_driver_device_info(
      ctx->vkInstance(), ctx->vkPhysicalDevice(), ApiVersion::VULKAN_1_4);
  auto tools = std::make_unique<spirv::SpirvTools>(device_info);
  auto glslCompiler =
      std::make_unique<spirv::GlslCompiler>(tools.get(), device_info);

  return BenchmarkState{
      .prng = std::move(prng),
      .tools = std::move(tools),
      .glslCompiler = std::move(glslCompiler),
  };
}

static memory::vector<uint32_t> select_targets_from_candidates(
    BenchmarkState &state, const denox::Db &db,
    memory::span<const uint32_t> candidates, size_t N, uint32_t minSamples,
    bool selectUnique = true,
    memory::optional<memory::span<const uint64_t>> samplesInFlight =
        memory::nullopt) {
  memory::vector<uint32_t> result;
  result.reserve(N);

  const auto dispatches = db.dispatches();

  auto inflight = [&](uint32_t candidateIndex) -> uint64_t {
    if (!samplesInFlight.has_value())
      return 0;
    return (*samplesInFlight)[candidateIndex];
  };

  // ------------------------------------------------------------
  // Phase 1: cold-first (minSamples), accounting for in-flight
  // ------------------------------------------------------------
  for (uint32_t i = 0; i < candidates.size() && result.size() < N; ++i) {
    uint32_t d = candidates[i]; // DB dispatch index

    uint64_t dbSamples = 0;
    if (dispatches[d].time.has_value())
      dbSamples = dispatches[d].time->samples;

    uint64_t effectiveSamples = dbSamples + inflight(i);

    if (effectiveSamples >= minSamples) {
      continue;
    }
    if (selectUnique) {
      result.push_back(i);
      continue;
    }

    uint64_t required = minSamples - effectiveSamples;
    uint64_t remaining = N - result.size();
    uint64_t count = std::min(required, remaining);

    for (uint32_t k = 0; k < count; ++k)
      result.push_back(i); // index into candidates
  }

  if (result.size() == N) {
    std::ranges::shuffle(result, state.prng);
    return result;
  }

  // ------------------------------------------------------------
  // Phase 2: SEM-based selection (no replacement, no in-flight,
  // exclude already-converged dispatches)
  // ------------------------------------------------------------
  struct Item {
    uint32_t candidateIndex; // index into candidates
    double priority;
  };

  const uint64_t remaining = N - result.size();

  memory::vector<Item> items;
  items.reserve(candidates.size());

  for (uint32_t i = 0; i < candidates.size(); ++i) {
    uint32_t d = candidates[i];
    const auto &dispatch = dispatches[d];

    // Exclude dispatches with no data yet from SEM phase
    if (!dispatch.time.has_value()) {
      items.emplace_back(i, std::numeric_limits<double>::infinity());
      continue;
    }

    const auto &t = *dispatch.time;
    const uint64_t n = t.samples;

    // SEM / mean priority (larger = worse)
    double priority;

    if (n > 1 && t.std_derivation_ns > 0 && t.latency_ns > 0) {
      double sem = static_cast<double>(t.std_derivation_ns) /
                   std::sqrt(static_cast<double>(n));

      priority = sem / static_cast<double>(t.latency_ns);
    } else {
      priority = std::numeric_limits<double>::infinity();
    }

    items.push_back({i, priority});
  }

  const uint64_t take = std::min<uint64_t>(items.size(), remaining);

  std::partial_sort(
      items.begin(), items.begin() + static_cast<std::ptrdiff_t>(take),
      items.end(),
      [](const Item &a, const Item &b) { return a.priority > b.priority; });

  for (uint32_t k = 0; k < take; ++k)
    result.push_back(items[k].candidateIndex);

  std::ranges::shuffle(result, state.prng);
  return result;
}

struct EpochDispatch {
  VkPipeline pipeline;
  VkPipelineLayout layout;
  memory::vector<VkDescriptorSetLayout> descriptorLayouts;
  memory::vector<VkDescriptorSet> descriptorSets; // ordered by set index.
  memory::span<const uint8_t> pc;
  uint32_t workgroupCountX;
  uint32_t workgroupCountY;
  uint32_t workgroupCountZ;
};

struct EpochStage {
  VkCommandBuffer cmd;
  VkQueryPool queryPool;
  VkFence fence;
};

struct Epoch {
  memory::vector<uint32_t> targets;
  VkCommandPool cmdPool;
  runtime::Buffer buffer;
  memory::vector<VkPipeline> pipelines;
  memory::vector<VkPipelineLayout> pipelineLayouts;
  memory::vector<VkDescriptorSetLayout> descriptorLayouts;
  VkDescriptorPool descriptorPool;
  memory::vector<EpochDispatch> dispatches;
  std::array<EpochStage, PIPELINE_STAGES> stages;
};

struct Timing {
  uint32_t samples;
  std::chrono::duration<float, std::milli> mean_latency;
  std::chrono::duration<float, std::milli> std_derivation;
};

struct EpochBenchResults {
  memory::vector<Timing> timings;
};

static Epoch create_epoch(const runtime::ContextHandle &ctx,
                          const denox::Db &db, memory::span<uint32_t> targets) {

  const auto dbdispatches = db.dispatches();
  const auto dbbinaries = db.binaries();

  memory::vector<VkPipeline> pipelines;
  memory::vector<VkPipelineLayout> pipelineLayouts;
  memory::vector<VkDescriptorSetLayout> descriptorSetLayouts;
  memory::vector<EpochDispatch> dispatches;

  uint32_t maxSets = 0;
  uint32_t storageBufferDescriptorCount = 0;
  size_t peakBufferSize = 0;

  memory::hash_map<uint32_t, uint32_t> binaryCache;

  for (uint32_t target : targets) {
    const auto &dbdispatch = dbdispatches[target];

    size_t totalBufferSize = 0;
    uint32_t maxSet = 0;
    for (const auto &binding : dbdispatch.bindings) {
      totalBufferSize = algorithm::align_up(totalBufferSize, binding.alignment);
      totalBufferSize += binding.byteSize;
      maxSet = std::max(maxSet, binding.set);
      storageBufferDescriptorCount += 1;
    }
    const uint32_t setCount = maxSet + 1;
    maxSets += setCount;
    peakBufferSize = std::max(peakBufferSize, totalBufferSize);

    const uint32_t binaryId = dbdispatch.binaryId;
    if (binaryCache.contains(binaryId)) {
      const auto &ref = dispatches[binaryCache[binaryId]];
      dispatches.push_back(EpochDispatch{
          .pipeline = ref.pipeline,
          .layout = ref.layout,
          .descriptorLayouts = ref.descriptorLayouts,
          .descriptorSets = {}, // <- allocated later.
          .pc = dbdispatch.pushConstant,
          .workgroupCountX = dbdispatch.workgroupCountX,
          .workgroupCountY = dbdispatch.workgroupCountY,
          .workgroupCountZ = dbdispatch.workgroupCountZ,
      });
    } else {
      memory::small_vector<
          memory::small_vector<VkDescriptorSetLayoutBinding, 4>, 4>
          bindings(setCount);
      for (const auto &binding : dbdispatch.bindings) {
        bindings[binding.set].push_back(VkDescriptorSetLayoutBinding{
            .binding = binding.binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        });
      }

      memory::vector<VkDescriptorSetLayout> setLayouts(setCount);
      for (uint32_t set = 0; set < setCount; ++set) {
        setLayouts[set] = ctx->createDescriptorSetLayout(bindings[set]);
        descriptorSetLayouts.push_back(setLayouts[set]);
      }

      VkPipelineLayout layout = ctx->createPipelineLayout(
          setLayouts, static_cast<uint32_t>(dbdispatch.pushConstant.size()));

      VkPipeline pipeline = ctx->createComputePipeline(
          layout, dbbinaries[binaryId].spvBinary.spv, "main");

      pipelines.push_back(pipeline);
      pipelineLayouts.push_back(layout);

      binaryCache[binaryId] = static_cast<uint32_t>(dispatches.size());

      dispatches.push_back(EpochDispatch{
          .pipeline = pipeline,
          .layout = layout,
          .descriptorLayouts = setLayouts,
          .descriptorSets = {}, // <- allocated later
          .pc = dbdispatch.pushConstant,
          .workgroupCountX = dbdispatch.workgroupCountX,
          .workgroupCountY = dbdispatch.workgroupCountY,
          .workgroupCountZ = dbdispatch.workgroupCountZ,
      });
    }
  }

  runtime::Buffer buffer =
      ctx->createBuffer(peakBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  memory::vector<VkDescriptorPoolSize> descriptorPoolSizes{
      VkDescriptorPoolSize{
          .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = storageBufferDescriptorCount,
      },
  };

  VkDescriptorPool descriptorPool =
      ctx->createDescriptorPool(maxSets, descriptorPoolSizes);

  for (uint32_t x = 0; x < targets.size(); ++x) {
    const uint32_t target = targets[x];
    const auto &dbdispatch = dbdispatches[target];
    auto &dispatch = dispatches[x];
    uint32_t setCount =
        static_cast<uint32_t>(dispatch.descriptorLayouts.size());
    dispatch.descriptorSets.resize(setCount);
    ctx->allocDescriptorSets(descriptorPool, dispatch.descriptorLayouts,
                             dispatch.descriptorSets.data());

    std::forward_list<VkDescriptorBufferInfo> monotone_allocator;
    memory::vector<VkWriteDescriptorSet> writeInfos;
    writeInfos.reserve(dbdispatch.bindings.size());
    uint64_t offset = 0;
    for (const auto &binding : dbdispatch.bindings) {
      auto &writeInfo = writeInfos.emplace_back();
      writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writeInfo.pNext = nullptr;
      writeInfo.dstSet = dispatch.descriptorSets[binding.set];
      writeInfo.dstBinding = binding.binding;
      writeInfo.dstArrayElement = 0;
      writeInfo.descriptorCount = 1;
      writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writeInfo.pImageInfo = nullptr;
      offset = algorithm::align_up(offset, binding.alignment);
      auto &bufferInfo = monotone_allocator.emplace_front();
      bufferInfo.buffer = buffer.vkbuffer;
      bufferInfo.range = binding.byteSize;
      bufferInfo.offset = offset;
      offset += binding.byteSize;
      writeInfo.pBufferInfo = &bufferInfo;
      ;
      writeInfo.pTexelBufferView = nullptr;
    }
    ctx->updateDescriptorSets(writeInfos);
  }

  VkCommandPool cmdPool = ctx->createCommandPool();

  std::array<EpochStage, PIPELINE_STAGES> stages;
  for (size_t i = 0; i < stages.size(); ++i) {
    stages[i].cmd = ctx->allocCommandBuffer(cmdPool);
    stages[i].queryPool = ctx->createTimestampQueryPool(BATCH_SIZE * 2);
    stages[i].fence = ctx->createFence(true);
  }

  return Epoch{
      .targets = {targets.begin(), targets.end()},
      .cmdPool = cmdPool,
      .buffer = buffer,
      .pipelines = std::move(pipelines),
      .pipelineLayouts = std::move(pipelineLayouts),
      .descriptorLayouts = std::move(descriptorSetLayouts),
      .descriptorPool = descriptorPool,
      .dispatches = std::move(dispatches),
      .stages = std::move(stages),
  };
}

static void destroy_epoch(const runtime::ContextHandle &ctx, Epoch epoch) {
  for (const auto pipeline : epoch.pipelines) {
    ctx->destroyPipeline(pipeline);
  }
  for (const auto pipelineLayout : epoch.pipelineLayouts) {
    ctx->destroyPipelineLayout(pipelineLayout);
  }
  for (const auto setLayout : epoch.descriptorLayouts) {
    ctx->destroyDescriptorSetLayout(setLayout);
  }
  ctx->destroyCommandPool(epoch.cmdPool);
  ctx->destroyBuffer(epoch.buffer);

  ctx->destroyDescriptorPool(epoch.descriptorPool);
  for (const auto &stage : epoch.stages) {
    ctx->destroyQueryPool(stage.queryPool);
    ctx->destroyFence(stage.fence);
  }
}

struct Batch {
  memory::vector<uint32_t> dispatches; // <- indexes into EpochDispatch
};

static Batch create_batch(BenchmarkState &state, const denox::Db &db,
                          const Epoch &epoch, uint32_t minSamples,
                          memory::span<uint64_t> samplesInFlight) {
  memory::vector<uint32_t> dispatches = select_targets_from_candidates(
      state, db, epoch.targets, BATCH_SIZE, minSamples, false, samplesInFlight);
  assert(!dispatches.empty());
  for (uint32_t x : dispatches) {
    samplesInFlight[x] += 1;
  }

  return Batch{
      .dispatches = std::move(dispatches),
  };
}

static void record_dispatch(VkCommandBuffer cmd,
                            const EpochDispatch &dispatch) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dispatch.pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dispatch.layout,
                          0,
                          static_cast<uint32_t>(dispatch.descriptorSets.size()),
                          dispatch.descriptorSets.data(), 0, nullptr);
  vkCmdPushConstants(cmd, dispatch.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     static_cast<uint32_t>(dispatch.pc.size()),
                     dispatch.pc.data());
  vkCmdDispatch(cmd, dispatch.workgroupCountX, dispatch.workgroupCountY,
                dispatch.workgroupCountZ);
}

static void record_batch(VkCommandBuffer cmd, const runtime::ContextHandle &ctx,
                         const Epoch &epoch, const EpochStage &stage,
                         const Batch &batch) {

  // jit warmup
  for (uint32_t d : batch.dispatches) {
    const auto &dispatch = epoch.dispatches[d];
    for (size_t i = 0; i < JIT_WARMUP_ITERATIONS; ++i) {
      record_dispatch(cmd, dispatch);
    }
  }

  for (uint32_t t = 0; t < batch.dispatches.size(); ++t) {
    uint32_t d = batch.dispatches[t];
    const auto &dispatch = epoch.dispatches[d];
    // l2 warmup
    for (size_t i = 0; i < L2_WARMUP_ITERATONS; ++i) {
      record_dispatch(cmd, dispatch);
    }
    // ctx->cmdMemoryBarrierComputeShader(cmd);
    ctx->cmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           stage.queryPool, t * 2);
    record_dispatch(cmd, dispatch);
    ctx->cmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           stage.queryPool, t * 2 + 1);
    // ctx->cmdMemoryBarrierComputeShader(cmd);
  }
}

static void read_batch(const runtime::ContextHandle &ctx,
                       const EpochStage &stage, const Batch &batch,
                       memory::span<Timing> timings) {
  const size_t N = batch.dispatches.size();
  const uint32_t QUERY_COUNT = static_cast<uint32_t>(N * 2);

  memory::vector<uint64_t> timestamps =
      ctx->getQueryResults(stage.queryPool, QUERY_COUNT);

  for (size_t i = 0; i < N; ++i) {
    const uint32_t epochDispatchIndex = batch.dispatches[i];
    Timing &timing = timings[epochDispatchIndex];

    const uint32_t t0 = static_cast<uint32_t>(i * 2 + 0);
    const uint32_t t1 = static_cast<uint32_t>(i * 2 + 1);

    uint64_t ts0 = timestamps[t0];
    uint64_t ts1 = timestamps[t1];

    // Defensive checks (same semantics as old code)
    if (ts1 <= ts0) {
      // Timestamp wrap or invalid ordering
      continue;
    }

    uint64_t latency_ns = ctx->timestampNanoDifference(timestamps, t0, t1);

    // ~100 ms sanity guard (same as before)
    if (latency_ns > 100'000'000'000ull) {
      continue;
    }

    // Incremental mean / variance update (Welford-style, batch weight = 1)
    const double x = static_cast<double>(latency_ns);

    if (timing.samples == 0) {
      timing.samples = 1;
      timing.mean_latency = std::chrono::duration<float, std::milli>(x * 1e-6);
      timing.std_derivation = std::chrono::duration<float, std::milli>(0.0f);
    } else {
      const double n = static_cast<double>(timing.samples);
      const double mean_old = static_cast<double>(timing.mean_latency.count());
      const double std_old = static_cast<double>(timing.std_derivation.count());
      const double var_old = std_old * std_old;

      const double mean_new = (n * mean_old + x * 1e-6) / (n + 1.0);

      const double var_new =
          (n * (var_old + (mean_old - mean_new) * (mean_old - mean_new)) +
           (x * 1e-6 - mean_new) * (x * 1e-6 - mean_new)) /
          (n + 1.0);

      timing.samples += 1;
      timing.mean_latency = std::chrono::duration<float, std::milli>(mean_new);
      timing.std_derivation =
          std::chrono::duration<float, std::milli>(std::sqrt(var_new));
    }
  }
}

static EpochBenchResults bench_epoch(BenchmarkState &state, const denox::Db &db,
                                     const runtime::ContextHandle &ctx,
                                     const Epoch &epoch,
                                     const runtime::DbBenchOptions &options,
                                     uint32_t samples) {
  memory::vector<Timing> timings(epoch.targets.size());
  std::array<Batch, PIPELINE_STAGES> batches;
  size_t stage = PIPELINE_STAGES - 1;

  uint64_t sampleCount = 0;
  memory::vector<uint64_t> samplesInFlight(epoch.targets.size());

  while (sampleCount < samples) {
    size_t next = (stage + 1) % PIPELINE_STAGES;
    ctx->waitFence(epoch.stages[next].fence);
    ctx->resetFence(epoch.stages[next].fence);
    if (!batches[next].dispatches.empty()) {
      read_batch(ctx, epoch.stages[next], batches[next], timings);
      sampleCount += batches[next].dispatches.size();
    }
    batches[next] =
        create_batch(state, db, epoch, options.minSamples, samplesInFlight);

    VkCommandBuffer cmd = epoch.stages[next].cmd;
    ctx->resetCommandBuffer(cmd);
    ctx->beginCommandBuffer(cmd);
    ctx->cmdResetQueryPool(cmd, epoch.stages[next].queryPool, 0,
                           BATCH_SIZE * 2);
    record_batch(epoch.stages[next].cmd, ctx, epoch, epoch.stages[next],
                 batches[next]);
    ctx->endCommandBuffer(epoch.stages[next].cmd);

    ctx->submit(epoch.stages[next].cmd, epoch.stages[next].fence);
    stage = next;
  }

  for (size_t i = 0; i < PIPELINE_STAGES; ++i) {
    ctx->waitFence(epoch.stages[i].fence);
    if (!batches[i].dispatches.empty()) {
      read_batch(ctx, epoch.stages[i], batches[i], timings);
    }
  }

  return EpochBenchResults{
      .timings = std::move(timings),
  };
}

static bool print_progress_report(const denox::Db &db,
                                  const runtime::DbBenchOptions &options,
                                  diag::Logger &logger) {
  const auto dispatches = db.dispatches();
  const uint64_t total = dispatches.size();

  uint64_t noData = 0;
  uint64_t insufficientSamples = 0;
  uint64_t insufficientPrecision = 0;
  uint64_t converged = 0;

  for (size_t i = 0; i < total; ++i) {
    const auto &d = dispatches[i];

    if (!d.time.has_value()) {
      noData += 1;
      insufficientSamples += 1;
      insufficientPrecision += 1;
      continue;
    }

    const auto &t = *d.time;
    const uint64_t n = t.samples;

    // --- minSamples check ---
    bool samplesOk = (n >= options.minSamples);
    if (!samplesOk)
      insufficientSamples += 1;

    // --- relative SEM check ---
    bool precisionOk = false;
    if (n > 1 && t.latency_ns > 0 && t.std_derivation_ns > 0) {
      double sem_ns = static_cast<double>(t.std_derivation_ns) /
                      std::sqrt(static_cast<double>(n));

      double relError = sem_ns / static_cast<double>(t.latency_ns);

      precisionOk = (relError <= static_cast<double>(options.maxRelativeError));
    }

    if (!precisionOk)
      insufficientPrecision += 1;

    if (samplesOk && precisionOk)
      converged += 1;
  }

  float progress =
      static_cast<float>(converged) / static_cast<float>(total) * 100.0f;

  logger.info("[{:>3}%] converged: {} / {} | "
              "minSamples pending: {} | "
              "precision pending: {} | "
              "no data: {}",
              static_cast<uint64_t>(std::floor(progress)), converged, total,
              insufficientSamples, insufficientPrecision, noData);

  return converged == total;
}

void denox::runtime::Db::bench(const DbBenchOptions &options) {
  assert(options.minSamples >= 1);

  BenchmarkState state = create_benchmark_state(m_context);

  diag::Logger logger("runtime.bench.db", true);

  // global warmup (10s or something, to send gpu out of boost)

  memory::vector<uint32_t> iota(m_db.dispatches().size());
  std::iota(iota.begin(), iota.end(), 0);

  bool running = !print_progress_report(m_db, options, logger);
  while (running) {
    memory::vector<uint32_t> targets = select_targets_from_candidates(
        state, m_db, iota, EPOCH_SIZE, options.minSamples);
    Epoch epoch = create_epoch(m_context, m_db, targets);
    auto result =
        bench_epoch(state, m_db, m_context, epoch, options, EPOCH_SAMPLES);
    uint32_t count =
        static_cast<uint32_t>(std::min(result.timings.size(), targets.size()));
    for (uint32_t i = 0; i < count; ++i) {
      const auto timing = result.timings[i];
      m_db.add_dispatch_benchmark_result(targets[i], timing.samples,
                                         timing.mean_latency,
                                         timing.std_derivation);
    }

    running = !print_progress_report(m_db, options, logger);

    if (options.saveProgress) {
      m_db.atomic_writeback();
    }
    destroy_epoch(m_context, std::move(epoch));
  }
}

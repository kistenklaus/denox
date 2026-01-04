#pragma once

#include "context.hpp"
#include "denox/common/ValueSpec.hpp"
#include "denox/memory/tensor/ActivationTensor.hpp"
#include "denox/symbolic/SymGraphEval.hpp"
#include "model.hpp"
#include <algorithm>
#include <vector>

namespace denox::runtime {

struct InstanceDispatch {
  const ModelDispatch *modelDispatch;
  memory::vector<VkDescriptorSet> descriptorSets;
  memory::vector<std::byte> pc;

  uint32_t workgroupCountX;
  uint32_t workgroupCountY;
  uint32_t workgroupCountZ;
};

struct InstanceBarrier {
  std::vector<VkBufferMemoryBarrier> bufferBarrier;
};

using InstanceCmd = std::variant<InstanceBarrier, InstanceDispatch>;

struct InstanceBenchmarkResult {
  struct Timing {
    memory::string name;
    memory::string input;
    memory::string output;
    uint64_t memoryReads;
    uint64_t memoryWrites;
    std::chrono::duration<float, std::milli> latency;
  };
  memory::vector<Timing> timings;

  memory::string report() const;
};

class Instance {
public:
  static std::shared_ptr<Instance>
  make(const ModelHandle &model, std::initializer_list<ValueSpec> specs) {
    return make(model, std::span{specs.begin(), specs.end()});
  }

  static std::shared_ptr<Instance> make(const ModelHandle &model,
                                        memory::span<const ValueSpec> specs);

  static std::shared_ptr<Instance> make(const ModelHandle &model,
                                        memory::span<const SymSpec> specs) {
    return std::shared_ptr<Instance>(new Instance(model, specs));
  }

  void infer(void *const *inputs, void **outputs) const;

  InstanceBenchmarkResult bench() const;

  ~Instance();
  Instance(const Instance &) = delete;
  Instance &operator=(const Instance &) = delete;
  Instance(Instance &&) = delete;
  Instance &operator=(Instance &&) = delete;

  void release();

private:
  Instance(const ModelHandle &model, memory::span<const SymSpec> specs);

  ModelHandle m_model;
  SymIREval m_symeval;
  VkDescriptorPool m_descriptorPool;
  memory::vector<InstanceCmd> m_cmds;
  memory::vector<runtime::Buffer> m_buffers;
};

using InstanceHandle = std::shared_ptr<Instance>;

} // namespace denox::runtime

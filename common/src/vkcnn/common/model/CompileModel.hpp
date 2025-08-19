#pragma once

#include "vkcnn/common/cmd/CmdOpSchedule.hpp"
#include "vkcnn/common/mr/MemoryRequirements.hpp"
namespace vkcnn {

class CompiledModel {
public:
  const MemoryRequirements &memoryRequirements() const {
    return m_memoryRequirements;
  }

  const CmdOpSchedule &schedule() const { return m_schedule; }

  CompiledModel(CmdOpSchedule schedule, MemoryRequirements memoryRequirements)
      : m_schedule(std::move(schedule)),
        m_memoryRequirements(std::move(memoryRequirements)) {}

private:
  CmdOpSchedule m_schedule;
  MemoryRequirements m_memoryRequirements;
};

} // namespace vkcnn

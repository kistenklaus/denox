#pragma once

#include "vkcnn/common/cmd/record/CmdRecordBarrier.hpp"

namespace vkcnn {

class CmdOpSchedule;

class CmdOpBarrier {
public:
  friend CmdOpSchedule;
  unsigned int inputTensor() const { return m_record->inputTensor; }
  unsigned int outputTensor() const { return m_record->outputTensor; }

private:
  CmdOpBarrier(const CmdRecordBarrier *record) : m_record(record) {}
  const CmdRecordBarrier *m_record;
};

} // namespace vkcnn

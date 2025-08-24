#pragma once

#include "vkcnn/common/cmd/CmdOpScheduleHeader.hpp"
#include "vkcnn/common/cmd/ParameterList.hpp"
#include "vkcnn/common/cmd/record/CmdRecordDispatch.hpp"
#include "vkcnn/common/cmd/record/SizedRecordHeader.hpp"
#include "vkcnn/common/memory/align.hpp"
#include <span>

namespace vkcnn {

class CmdOpSchedule;

class CmdOpDispatch {
public:
  friend CmdOpSchedule;
  unsigned int inputTensor() const { return m_record->inputTensor; }
  unsigned int outputTensor() const { return m_record->outputTensor; }

  std::span<const std::uint32_t> spriv() const {
    auto header = static_cast<const CmdOpScheduleHeader *>(m_buffer);
    auto spirvHeaderPtr =
        static_cast<const std::byte *>(m_buffer) + header->spirv_offset;
    auto spirvHeader =
        reinterpret_cast<const SizedRecordHeader *>(spirvHeaderPtr);
    std::size_t spirvSize = spirvHeader->size;
    auto spirv = reinterpret_cast<const std::uint32_t *>(
        align_up(spirvHeaderPtr, alignof(std::uint32_t)));
    return std::span{spirv, spirvSize};
  }

  ParameterList parameters() const {
    auto raw = reinterpret_cast<const std::byte *>(m_record) +
               sizeof(CmdRecordDispatch);
    raw = align_up(raw, alignof(std::uint64_t));
    auto parameterPtr = reinterpret_cast<const std::uint64_t *>(raw);
    return ParameterList{m_record->parameterCount, parameterPtr, m_buffer};
  }

private:
  CmdOpDispatch(const void *buffer, const CmdRecordDispatch *record)
      : m_buffer(buffer), m_record(record) {}

  const void *m_buffer;
  const CmdRecordDispatch *m_record;
};

} // namespace vkcnn

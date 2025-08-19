#pragma once

#include "vkcnn/common/cmd/CmdOpBuffer.hpp"
#include <memory>

namespace vkcnn {

class CmdOpSchedule {
public:
  CmdOpSchedule(const CmdOpBuffer& buffer)
      : m_buffer(std::make_shared<CmdOpBuffer>(buffer)) {
    m_buffer->shrink_to_fit();
  }

  CmdOpSchedule(CmdOpBuffer &&buffer)
      : m_buffer(std::make_shared<CmdOpBuffer>(std::move(buffer))) {
    m_buffer->shrink_to_fit();
  }

  auto begin() const { return m_buffer->begin(); }
  auto end() const { return m_buffer->end(); }
  auto byteSize() const { return m_buffer->byteSize(); }

private:
  std::shared_ptr<CmdOpBuffer> m_buffer;
};

} // namespace vkcnn

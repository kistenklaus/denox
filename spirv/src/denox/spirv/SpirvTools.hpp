#pragma once

#include "denox/device_info/DeviceInfo.hpp"
#include "denox/spirv/SpirvBinary.hpp"
#include "denox/memory/container/span.hpp"
#include <fmt/format.h>

namespace denox::spirv {

struct SpirvDescriptorRebind {
  uint8_t src_set;
  uint8_t src_binding;
  uint8_t dst_set;
  uint8_t dst_binding;
};

class SpirvTools {
public:
  SpirvTools(const DeviceInfo &deviceInfo);
  SpirvTools(const SpirvTools &) = delete;
  SpirvTools &operator=(const SpirvTools &) = delete;
  SpirvTools(SpirvTools &o);
  SpirvTools &operator=(SpirvTools &);
  ~SpirvTools();

  bool validate(const SpirvBinary &binary);
  bool optimize(SpirvBinary &binary);

  bool rebind(SpirvBinary &binary,
              memory::span<const SpirvDescriptorRebind> rebinds);

  std::string_view get_error_msg() const { return m_log; }

private:
private:
  std::string m_log;
  const char *m_current_stage = nullptr;

  void * /*spvtools::SpirvTools*/ m_tools;
  void * /*spvtools::Optimizer*/ m_optimizer;
};

} // namespace denox::spirv

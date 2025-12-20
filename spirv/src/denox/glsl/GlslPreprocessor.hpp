#pragma once
#include "denox/memory/container/string_view.hpp"
#include "denox/memory/container/string.hpp"

namespace denox::spirv {

class GlslPreprocessor {
public:
  GlslPreprocessor() = default;

  memory::string preprocess(memory::string_view src);

  void set_enable_unroll_translation(bool v) { m_enableUnroll = v; }

private:
  bool m_enableUnroll = true;
};

} // namespace denox::spirv

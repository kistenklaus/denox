#pragma once
#include "denox/memory/container/string.hpp"
#include "denox/memory/container/vector.hpp"
#include <cstdint>
#include <regex>

namespace denox::compiler {

class ShaderPreprocessor {
public:
  struct Block {
    memory::string id;
    std::uint32_t begin_idx;  // '#pragma ... begin_block(...)' line (0-based)
    std::uint32_t end_idx;    // '#pragma ... end_block(...)'   line (0-based)
    std::uint32_t body_begin; // begin_idx + 1
    std::uint32_t body_end;   // end_idx (one past last body line)
    std::uint32_t start_ln;   // 1-based original line number of body_begin
  };

  explicit ShaderPreprocessor(memory::string filename)
      : m_filename(std::move(filename)) {}

  // Main entry: returns transformed source with #line resyncs.
  memory::string preprocess(const memory::string &src);

  // Optional: expose captured blocks (for tools / debugging)
  const memory::vector<Block> &blocks() const { return m_blocks; }

  // Toggle: translate '#pragma unroll' → [[unroll]] (guarded)
  void set_enable_unroll_translation(bool v) { m_enableUnroll = v; }

private:
private:
  memory::string m_filename;
  memory::vector<Block> m_blocks;
  bool m_enableUnroll = true;

  const std::regex m_rxBegin{
      R"(^\s*#\s*pragma\s+(?:denox\s+)?begin_block\s*\(\s*([A-Za-z_]\w*)\s*\)\s*;?\s*(?://.*)?$)"};

  const std::regex m_rxEnd{
      R"(^\s*#\s*pragma\s+(?:denox\s+)?end_block(?:\s*\(\s*([A-Za-z_]\w*)\s*\))?\s*;?\s*(?://.*)?$)"};

  const std::regex m_rxInline{
      R"(^\s*#\s*pragma\s+(?:denox\s+)?inline_block\s*\(\s*([A-Za-z_]\w*)\s*\)\s*;?\s*(?://.*)?$)"};

  const std::regex m_rxUnroll{
      R"(^\s*#\s*pragma\s+(?:denox\s+)?unroll\b\s*;?\s*(?://.*)?$)"};
};

} // namespace denox::compiler

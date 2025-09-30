#pragma once
#include "memory/container/string.hpp"
#include "memory/container/vector.hpp"
#include <regex>
#include <string>
#include <string_view>

namespace denox::compiler {

class ShaderPreprocessor {
public:
  struct Block {
    memory::string id;
    memory::string src;   // original lines of the block (joined with '\n')
    std::size_t start_ln; // 1-based line of first line inside the block body
    std::size_t end_ln;   // 1-based line of the closing #pragma end_block
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
  // Helpers (impl in .cpp)
  static void split_lines(const memory::string &s,
                          memory::vector<memory::string> &out);
  void append_resync(memory::string &out, std::size_t next_original_line) const;
  void capture_block(const memory::vector<memory::string> &lines,
                     std::size_t begin_idx, std::size_t end_idx,
                     const memory::string &id);

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

#include "shaders/compiler/ShaderPreprocessor.hpp"
#include <algorithm>
#include <fmt/format.h>
#include <stdexcept>
#include <string_view>

namespace denox::compiler {

void ShaderPreprocessor::split_lines(const memory::string& s,
                                     memory::vector<memory::string>& out) {
  out.clear();
  std::string_view v{s.c_str(), s.size()};
  std::size_t start = 0;
  while (start <= v.size()) {
    std::size_t nl = v.find('\n', start);
    std::size_t len = (nl == std::string_view::npos) ? v.size() - start : nl - start;
    memory::string line(v.substr(start, len));
    if (!line.empty() && line.back() == '\r') line.pop_back();
    out.push_back(std::move(line));
    if (nl == std::string_view::npos) break;
    start = nl + 1;
  }
}

void ShaderPreprocessor::append_resync(memory::string& out,
                                       std::size_t next_original_line) const {
  // #line uses 1-based line numbers; next_original_line is already 1-based
  out += fmt::format("#line {} \"{}\"\n", next_original_line, m_filename);
}

void ShaderPreprocessor::capture_block(const memory::vector<memory::string>& lines,
                                       std::size_t begin_idx, std::size_t end_idx,
                                       const memory::string& id) {
  // begin_idx points at '#pragma begin_block', end_idx at '#pragma end_block'
  // Block body is (begin_idx+1) .. (end_idx-1), both 0-based
  Block b;
  b.id = id;
  b.start_ln = begin_idx + 2; // first line AFTER begin_... (1-based)
  b.end_ln   = end_idx + 1;   // line of end_block itself (1-based)
  for (std::size_t i = begin_idx + 1; i < end_idx; ++i) {
    b.src += lines[i];
    b.src.push_back('\n');
  }
  m_blocks.push_back(std::move(b));
}

memory::string ShaderPreprocessor::preprocess(const memory::string& src) {
  m_blocks.clear();

  memory::vector<memory::string> lines;
  split_lines(src, lines);

  memory::string out;
  out.reserve(src.size() + 256);

  const std::size_t n = lines.size();
  for (std::size_t i = 0; i < n; ++i) {
    const memory::string& L = lines[i];

    // 1) Handle '#pragma begin_block(...)' → capture until matching end, erase
    std::smatch m;
    if (std::regex_match(L, m, m_rxBegin)) {
      const memory::string blockId = m[1].str();
      // Find matching end (no nesting allowed)
      std::size_t j = i + 1;
      for (; j < n; ++j) {
        if (std::regex_match(lines[j], m, m_rxBegin)) {
          throw std::runtime_error("Recursive #pragma blocks are not supported");
        }
        if (std::regex_match(lines[j], m, m_rxEnd)) break;
      }
      if (j >= n) {
        throw std::runtime_error(fmt::format(
            "Unterminated block \"{}\": missing '#pragma end_block'", blockId));
      }

      // Capture the block (preserve its original line numbers)
      capture_block(lines, i, j, blockId);

      // We drop the entire region [i..j] from output.
      // Resync the next output line to the original source line (j+1).
      append_resync(out, /*next_original_line*/ j + 2);
      i = j; // skip to end_block (loop will ++i to the next line)
      continue;
    }

    // 2) Handle '#pragma inline_block(ID)' → insert block content
    if (std::regex_match(L, m, m_rxInline)) {
      const memory::string targetId = m[1].str();
      auto it = std::find_if(m_blocks.begin(), m_blocks.end(),
                             [&](const Block& b){ return b.id == targetId; });
      if (it == m_blocks.end()) {
        throw std::runtime_error(fmt::format(
            "#pragma inline_block references unknown block \"{}\"", targetId));
      }

      // Map diagnostics to the block's ORIGINAL position,
      // then restore mapping back to the caller line (i+1).
      append_resync(out, it->start_ln);
      out += it->src;
      append_resync(out, /*resume at next source line after inline*/ i + 2);
      continue; // we consumed the inline line
    }

    // 3) Translate '#pragma unroll' → [[unroll]] with extension guard
    if (m_enableUnroll && std::regex_match(L, m, m_rxUnroll)) {
      out += "#ifdef GL_EXT_control_flow_attributes\n[[unroll]]\n#endif\n";
      // The original line is replaced by three physical lines; resync to the
      // next original line for correct diagnostics.
      append_resync(out, i + 2);
      continue;
    }

    // 4) All other lines: forward as-is
    out += L;
    out.push_back('\n');
  }

  return out;
}

} // namespace denox::compiler::shaders

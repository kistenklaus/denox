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
  out += fmt::format("#line {}\n", next_original_line);
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

  // -------- Pass 0: split to lines --------
  memory::vector<memory::string> lines;
  split_lines(src, lines);
  const std::size_t n = lines.size();

  // -------- Pass 1: capture all blocks (supports nesting) --------
  struct Beg {
    memory::string id;
    std::size_t begin_idx; // line index of '#pragma begin_block'
  };
  memory::vector<Beg> stack;
  // map begin -> end (for skipping regions later)
  std::vector<long long> end_for_begin(n, -1);

  std::smatch m;
  for (std::size_t i = 0; i < n; ++i) {
    const memory::string& L = lines[i];
    if (std::regex_match(L, m, m_rxBegin)) {
      stack.push_back(Beg{memory::string(m[1].str()), i});
    } else if (std::regex_match(L, m, m_rxEnd)) {
      if (stack.empty()) {
        throw std::runtime_error(
          fmt::format("Stray '#pragma end_block' at line {}", i + 1));
      }
      Beg beg = std::move(stack.back());
      stack.pop_back();

      // Capture this block [beg.begin_idx .. i]
      capture_block(lines, beg.begin_idx, i, beg.id);
      end_for_begin[beg.begin_idx] = static_cast<long long>(i);
    }
  }
  if (!stack.empty()) {
    // Report the first unterminated block for clarity
    const Beg& b = stack.back();
    throw std::runtime_error(
      fmt::format("Unterminated block \"{}\" starting at line {}", b.id, b.begin_idx + 1));
  }

  // Helper: find a captured block by id
  auto find_block = [&](const memory::string& id) -> const Block& {
    auto it = std::find_if(m_blocks.begin(), m_blocks.end(),
                           [&](const Block& b){ return b.id == id; });
    if (it == m_blocks.end()) {
      throw std::runtime_error(
        fmt::format("#pragma inline_block references unknown block \"{}\"", id));
    }
    return *it;
  };

  // Inline recursion guard (per-expansion path)
  struct InlineGuard {
    std::vector<memory::string>& stack;
    memory::string name;
    InlineGuard(std::vector<memory::string>& s, memory::string n)
      : stack(s), name(std::move(n)) { stack.push_back(name); }
    ~InlineGuard() {
      if (!stack.empty() && stack.back() == name) stack.pop_back();
    }
  };
  std::vector<memory::string> inline_stack;

  // Expands a block, skipping nested definitions inside it,
  // expanding nested inline_block pragmas, and maintaining #line.
  std::function<void(const Block&)> expand_block = [&](const Block& blk) {
    // Cycle check
    auto it = std::find(inline_stack.begin(), inline_stack.end(), blk.id);
    if (it != inline_stack.end()) {
      // Build a nice chain for diagnostics
      memory::string chain;
      for (const auto& n : inline_stack) {
        if (!chain.empty()) chain += " -> ";
        chain += n;
      }
      if (!chain.empty()) chain += " -> ";
      chain += blk.id;
      throw std::runtime_error("Recursive #pragma inline_block expansion detected: " + std::string(chain));
    }
    InlineGuard guard{inline_stack, blk.id};

    // Resync to the first line of the block body in the ORIGINAL source
    append_resync(/*out*/ *(new memory::string()), 0); // placeholder to satisfy compiler (will be optimized out)
  };

  // We can't write into a placeholder; re-create expand_block correctly with 'out' captured by reference.
  memory::string out;
  out.reserve(src.size() + 256);

  expand_block = [&](const Block& blk) {
    InlineGuard guard{inline_stack, blk.id};
    // Resync logical line numbers to the block's original first body line
    append_resync(out, blk.start_ln);

    // Walk the block source line-by-line
    memory::vector<memory::string> blines;
    split_lines(blk.src, blines);

    // We'll need local scanning to skip nested begin/end pairs that were captured separately.
    // Track depth to find matching end inside this block text.
    for (std::size_t k = 0; k < blines.size(); ++k) {
      const memory::string& L2 = blines[k];

      // BEGIN inside block body → skip until matching END (nested definition)
      if (std::regex_match(L2, m, m_rxBegin)) {
        int depth = 1;
        std::size_t t = k;
        for (++t; t < blines.size(); ++t) {
          if (std::regex_match(blines[t], m, m_rxBegin)) ++depth;
          else if (std::regex_match(blines[t], m, m_rxEnd)) {
            if (--depth == 0) break;
          }
        }
        if (depth != 0) {
          throw std::runtime_error(
            fmt::format("Unterminated nested block inside \"{}\" starting at original line {}",
                        blk.id, blk.start_ln + k));
        }
        // Resync after the nested end: next original line = (blk.start_ln + t)
        append_resync(out, blk.start_ln + static_cast<int>(t) + 1);
        k = t; // skip the whole nested region
        continue;
      }

      // INLINE inside block body → expand recursively
      if (std::regex_match(L2, m, m_rxInline)) {
        const memory::string nestedId = m[1].str();
        const Block& nb = find_block(nestedId);
        expand_block(nb);
        // Resume mapping at the next original line after the inline directive
        append_resync(out, blk.start_ln + static_cast<int>(k) + 1);
        continue;
      }

      // Unroll pragma translation (optional)
      if (m_enableUnroll && std::regex_match(L2, m, m_rxUnroll)) {
        out += "#ifdef GL_EXT_control_flow_attributes\n[[unroll]]\n#endif\n";
        append_resync(out, blk.start_ln + static_cast<int>(k) + 1);
        continue;
      }

      // Normal line
      out += L2;
      out.push_back('\n');
    }
  };

  // -------- Pass 2: emit file (skip definitions; expand inlines) --------
  for (std::size_t i = 0; i < n; ++i) {
    const memory::string& L = lines[i];

    // If this line is a begin of a captured block, skip the whole region and resync
    if (end_for_begin[i] != -1) {
      std::size_t j = static_cast<std::size_t>(end_for_begin[i]);
      append_resync(out, /*next original line*/ j + 2);
      i = j; // jump to end; loop ++ will move to the next line
      continue;
    }

    // Inline directive at top level
    if (std::regex_match(L, m, m_rxInline)) {
      const memory::string targetId = m[1].str();
      const Block& b = find_block(targetId);
      expand_block(b);
      // Resume mapping at the next original line after this inline
      append_resync(out, i + 2);
      continue;
    }

    // Unroll pragma translation at top level
    if (m_enableUnroll && std::regex_match(L, m, m_rxUnroll)) {
      out += "#ifdef GL_EXT_control_flow_attributes\n[[unroll]]\n#endif\n";
      append_resync(out, i + 2);
      continue;
    }

    // All other lines pass through
    out += L;
    out.push_back('\n');
  }

  return out;
}

} // namespace denox::compiler::shaders

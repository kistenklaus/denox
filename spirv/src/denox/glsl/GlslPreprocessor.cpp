#include "denox/glsl/GlslPreprocessor.hpp"
#include "denox/diag/invalid_argument.hpp"
#include "denox/diag/logging.hpp"
#include "denox/memory/container/hashmap.hpp"
#include "denox/memory/container/string_view.hpp"
#include "denox/memory/container/vector.hpp"
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <fmt/format.h>

namespace denox::spirv {

static inline bool is_space(char c) noexcept { return c == ' ' || c == '\t'; }

static inline memory::string_view ltrim(memory::string_view s) noexcept {
  while (!s.empty() && is_space(s.front())) {
    s.remove_prefix(1);
  }
  return s;
}

static inline bool is_ident_char(char c) noexcept {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || (c == '_');
}

static inline bool is_ident_first(char c) noexcept {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c == '_');
}

static inline memory::string_view parse_ident(memory::string_view &s) noexcept {
  memory::string_view t = s;
  if (t.empty() || !is_ident_first(t.front())) {
    return {};
  }
  size_t n = 1;
  while (n < t.size() && is_ident_char(t[n])) {
    ++n;
  }
  memory::string_view out = t.substr(0, n);
  s.remove_prefix(n);
  return out;
}

static inline bool consume_char(memory::string_view &s, char ch) noexcept {
  if (!s.empty() && s.front() == ch) {
    s.remove_prefix(1);
    return true;
  }
  return false;
}

static inline bool tail_ok(memory::string_view s) noexcept {
  s = ltrim(s);
  if (s.empty()) {
    return true;
  }
  return (s.size() >= 2 && s[0] == '/' && s[1] == '/');
}

enum class PragmaKind : uint8_t { None, Begin, End, Inline, Unroll };

struct ParsedPragma {
  PragmaKind kind = PragmaKind::None;
  memory::string_view id{};
};

static inline ParsedPragma parse_pragma(memory::string_view line) noexcept {
  line = ltrim(line);
  if (line.empty() || line.front() != '#') {
    return {};
  }
  line.remove_prefix(1);
  line = ltrim(line);
  {
    memory::string_view tmp = line;
    memory::string_view tok = parse_ident(tmp);
    if (tok != "pragma")
      return {};
    if (tmp.empty() || !is_space(tmp.front()))
      return {};
    line = tmp;
  }

  line = ltrim(line);

  {
    memory::string_view tmp = line;
    memory::string_view tok = parse_ident(tmp);
    if (tok == "denox") {
      if (tmp.empty() || !is_space(tmp.front())) {
        return {};
      }
      line = ltrim(tmp);
    }
  }

  memory::string_view kw = parse_ident(line);
  if (kw.empty()) {
    return {};
  }
  line = ltrim(line);

  auto parse_paren_id = [&](memory::string_view &s,
                            memory::string_view &out_id) noexcept -> bool {
    memory::string_view t = ltrim(s);
    if (!consume_char(t, '(')) {
      return false;
    }
    t = ltrim(t);
    memory::string_view id = parse_ident(t);
    if (id.empty()) {
      return false;
    }
    t = ltrim(t);
    if (!consume_char(t, ')')) {
      return false;
    }
    s = t;
    out_id = id;
    return true;
  };

  // optional ";"
  auto consume_optional_semicolon = [&](memory::string_view &s) noexcept {
    s = ltrim(s);
    if (!s.empty() && s.front() == ';') {
      s.remove_prefix(1);
    }
  };

  if (kw == "begin_block") {
    memory::string_view id{};
    if (!parse_paren_id(line, id)) {
      return {};
    }
    consume_optional_semicolon(line);
    if (!tail_ok(line)) {
      return {};
    }
    return {PragmaKind::Begin, id};
  }

  if (kw == "inline_block") {
    memory::string_view id{};
    if (!parse_paren_id(line, id)) {
      return {};
    }
    consume_optional_semicolon(line);
    if (!tail_ok(line)) {
      return {};
    }
    return {PragmaKind::Inline, id};
  }

  if (kw == "end_block") {
    memory::string_view maybe_id{};
    {
      memory::string_view tmp = line;
      if (parse_paren_id(tmp, maybe_id)) {
        line = tmp;
      }
    }
    consume_optional_semicolon(line);
    if (!tail_ok(line))
      return {};
    return {PragmaKind::End, {}};
  }

  if (kw == "unroll") {
    consume_optional_semicolon(line);
    if (!tail_ok(line))
      return {};
    return {PragmaKind::Unroll, {}};
  }

  return {};
}

memory::string GlslPreprocessor::preprocess(memory::string_view src) {

  struct Block {
    memory::string id;
    uint32_t begin_idx;
    uint32_t end_idx;
    uint32_t body_begin;
    uint32_t body_end;
    uint32_t start_ln;
  };

  memory::vector<Block> blocks;

  blocks.clear();

  const char *const data = src.data();
  const size_t sz = src.size();

  memory::vector<uint32_t> line_start;
  memory::vector<uint32_t> line_len;
  line_start.reserve(512);
  line_len.reserve(512);

  const char *p = data;
  const char *const end = data + sz;

  while (true) {
    const char *nl = static_cast<const char *>(
        memchr(p, '\n', static_cast<size_t>(end - p)));
    const char *line_end = nl ? nl : end;

    if (line_end > p && line_end[-1] == '\r') {
      --line_end;
    }

    line_start.push_back(static_cast<uint32_t>(p - data));
    line_len.push_back(static_cast<uint32_t>(line_end - p));

    if (!nl) {
      break;
    }
    p = nl + 1;

    if (p == end) {
      line_start.push_back(static_cast<uint32_t>(end - data));
      line_len.push_back(0);
      break;
    }
  }

  const uint32_t n_lines = static_cast<uint32_t>(line_start.size());

  auto line_view = [&](uint32_t i) noexcept -> memory::string_view {
    return memory::string_view{data + line_start[i], line_len[i]};
  };

  struct Beg {
    memory::string id;
    uint32_t begin_idx;
  };

  memory::vector<Beg> stack;
  stack.reserve(16);

  memory::vector<int32_t> end_for_begin(n_lines, -1);

  for (uint32_t i = 0; i < n_lines; ++i) {
    const ParsedPragma pp = parse_pragma(line_view(i));

    if (pp.kind == PragmaKind::Begin) {
      stack.push_back(Beg{memory::string(pp.id), i});
      continue;
    }

    if (pp.kind == PragmaKind::End) {
      if (stack.empty()) {
        DENOX_ERROR("Stray '#pragma end_block' at line {}", i + 1);
        diag::invalid_argument();
      }

      Beg beg = std::move(stack.back());
      stack.pop_back();

      Block b;
      b.id = std::move(beg.id);
      b.begin_idx = beg.begin_idx;
      b.end_idx = i;
      b.body_begin = b.begin_idx + 1;
      b.body_end = b.end_idx;
      b.start_ln = b.body_begin + 1;

      blocks.push_back(std::move(b));
      end_for_begin[static_cast<size_t>(beg.begin_idx)] =
          static_cast<int32_t>(i);
      continue;
    }
  }

  if (!stack.empty()) {
    const Beg &b = stack.back();

    DENOX_ERROR("Unterminated block \"{}\" starting at line {}", b.id,
                b.begin_idx + 1);
    diag::invalid_argument();
  }

  memory::hash_map<memory::string_view, uint32_t> block_index;
  block_index.reserve(blocks.size() * 2 + 1);

  for (uint32_t bi = 0; bi < static_cast<uint32_t>(blocks.size()); ++bi) {
    const auto &b = blocks[bi];
    memory::string_view key{b.id.data(), b.id.size()};
    if (block_index.find(key) == block_index.end()) {
      block_index.emplace(key, bi);
    }
  }

  auto get_block_index = [&](memory::string_view id) -> uint32_t {
    auto it = block_index.find(id);
    if (it == block_index.end()) {
      DENOX_ERROR("#pragma inline_block references unknown block \"{}\"", id);
      diag::invalid_argument();
    }
    return it->second;
  };

  memory::string out;
  out.reserve(src.size() + 256);

  auto append_resync = [&](uint32_t next_original_line_1based) {
    out.append("#line ");
    char buf[32];
    auto [ptr, ec] =
        std::to_chars(buf, buf + sizeof(buf), next_original_line_1based);
    (void)ec;
    out.append(buf, static_cast<size_t>(ptr - buf));
    out.push_back('\n');
  };

  static constexpr memory::string_view kUnrollSnippet =
      "#ifdef GL_EXT_control_flow_attributes\n[[unroll]]\n#endif\n";

  memory::vector<uint32_t> inline_stack;
  inline_stack.reserve(16);

  auto throw_cycle = [&](uint32_t next_idx) {
    memory::string chain;
    for (uint32_t idx : inline_stack) {
      if (!chain.empty()) {
        chain += " -> ";
      }
      chain += blocks[idx].id;
    }
    if (!chain.empty()) {
      chain += " -> ";
    }
    chain += blocks[next_idx].id;

    DENOX_ERROR("Recursive #pragma inline_block expansion detected: {}", chain);
    diag::invalid_argument();
  };

  auto expand_block = [&](auto &&self, uint32_t blk_idx) -> void {
    if (std::find(inline_stack.begin(), inline_stack.end(), blk_idx) !=
        inline_stack.end()) {
      throw_cycle(blk_idx);
    }
    inline_stack.push_back(blk_idx);

    const Block &blk = blocks[blk_idx];

    append_resync(blk.start_ln);

    for (uint32_t i = blk.body_begin; i < blk.body_end; ++i) {
      const int32_t end_idx = end_for_begin[static_cast<size_t>(i)];
      if (end_idx != -1) {
        const uint32_t j = static_cast<uint32_t>(end_idx);
        append_resync(j + 2);
        i = j;
        continue;
      }

      const memory::string_view L = line_view(i);
      const ParsedPragma pp = parse_pragma(L);

      if (pp.kind == PragmaKind::Inline) {
        const uint32_t nb_idx = get_block_index(pp.id);
        self(self, nb_idx);
        append_resync(i + 2);
        continue;
      }

      if (m_enableUnroll && pp.kind == PragmaKind::Unroll) {
        out.append(kUnrollSnippet.data(), kUnrollSnippet.size());
        append_resync(i + 2);
        continue;
      }

      out.append(L.data(), L.size());
      out.push_back('\n');
    }

    inline_stack.pop_back();
  };

  for (uint32_t i = 0; i < n_lines; ++i) {
    const int32_t end_idx = end_for_begin[static_cast<size_t>(i)];
    if (end_idx != -1) {
      const uint32_t j = static_cast<uint32_t>(end_idx);
      append_resync(j + 2);
      i = j;
      continue;
    }

    const memory::string_view L = line_view(i);
    const ParsedPragma pp = parse_pragma(L);

    if (pp.kind == PragmaKind::Inline) {
      const uint32_t b_idx = get_block_index(pp.id);
      expand_block(expand_block, b_idx);
      append_resync(i + 2);
      continue;
    }

    if (m_enableUnroll && pp.kind == PragmaKind::Unroll) {
      out.append(kUnrollSnippet.data(), kUnrollSnippet.size());
      append_resync(i + 2);
      continue;
    }

    out.append(L.data(), L.size());
    out.push_back('\n');
  }

  return out;
}

} // namespace denox::spirv

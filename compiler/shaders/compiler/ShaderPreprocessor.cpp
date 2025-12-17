#include "shaders/compiler/ShaderPreprocessor.hpp"
#include <charconv>
#include <cstdint>
#include <cstring>       // memchr
#include <fmt/format.h>
#include <unordered_map>
#include <vector>

namespace denox::compiler {

namespace {

// --- small ASCII helpers ---
static inline bool is_space(char c) noexcept { return c == ' ' || c == '\t'; }

static inline std::string_view ltrim(std::string_view s) noexcept {
  while (!s.empty() && is_space(s.front())) s.remove_prefix(1);
  return s;
}

static inline bool is_ident_char(char c) noexcept {
  return (c >= 'a' && c <= 'z') ||
         (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') ||
         (c == '_');
}

static inline bool is_ident_first(char c) noexcept {
  return (c >= 'a' && c <= 'z') ||
         (c >= 'A' && c <= 'Z') ||
         (c == '_');
}

static inline std::string_view parse_ident(std::string_view& s) noexcept {
  std::string_view t = s;
  if (t.empty() || !is_ident_first(t.front())) return {};
  std::size_t n = 1;
  while (n < t.size() && is_ident_char(t[n])) ++n;
  std::string_view out = t.substr(0, n);
  s.remove_prefix(n);
  return out;
}

static inline bool consume_char(std::string_view& s, char ch) noexcept {
  if (!s.empty() && s.front() == ch) {
    s.remove_prefix(1);
    return true;
  }
  return false;
}

static inline bool tail_ok(std::string_view s) noexcept {
  s = ltrim(s);
  if (s.empty()) return true;
  return (s.size() >= 2 && s[0] == '/' && s[1] == '/');
}

enum class PragmaKind : std::uint8_t { None, Begin, End, Inline, Unroll };

struct ParsedPragma {
  PragmaKind kind = PragmaKind::None;
  std::string_view id{}; // begin/inline: required; end: optional/ignored
};

// Parser that mirrors your regexes:
//
//  ^\s*#\s*pragma\s+(?:denox\s+)?KW ... ;?\s*(?://.*)?$
//
// begin_block / inline_block require: \(\s*(ID)\s*\)
// end_block allows optional: ( \(\s*(ID)\s*\) )?
// unroll allows: \bunroll\b with optional ; and comment
//
static inline ParsedPragma parse_pragma(std::string_view line) noexcept {
  line = ltrim(line);
  if (line.empty() || line.front() != '#') return {};
  line.remove_prefix(1);
  line = ltrim(line);

  // parse "pragma" as an identifier token (must be exactly "pragma")
  {
    std::string_view tmp = line;
    std::string_view tok = parse_ident(tmp);
    if (tok != "pragma") return {};
    // regex requires at least one whitespace after 'pragma': \s+
    if (tmp.empty() || !is_space(tmp.front())) return {};
    line = tmp;
  }

  line = ltrim(line);

  // optional "denox" token, requires at least one whitespace after it if present: (?:denox\s+)?
  {
    std::string_view tmp = line;
    std::string_view tok = parse_ident(tmp);
    if (tok == "denox") {
      if (tmp.empty() || !is_space(tmp.front())) return {}; // must be \s+ after denox
      line = ltrim(tmp);
    }
  }

  // directive keyword
  std::string_view kw = parse_ident(line);
  if (kw.empty()) return {};
  line = ltrim(line);

  // Helpers for parsing "( ID )"
  auto parse_paren_id = [&](std::string_view& s, std::string_view& out_id) noexcept -> bool {
    std::string_view t = ltrim(s);
    if (!consume_char(t, '(')) return false;
    t = ltrim(t);
    std::string_view id = parse_ident(t);
    if (id.empty()) return false;
    t = ltrim(t);
    if (!consume_char(t, ')')) return false;
    s = t;
    out_id = id;
    return true;
  };

  // optional ";"
  auto consume_optional_semicolon = [&](std::string_view& s) noexcept {
    s = ltrim(s);
    if (!s.empty() && s.front() == ';') {
      s.remove_prefix(1);
    }
  };

  if (kw == "begin_block") {
    std::string_view id{};
    if (!parse_paren_id(line, id)) return {};
    consume_optional_semicolon(line);
    if (!tail_ok(line)) return {};
    return {PragmaKind::Begin, id};
  }

  if (kw == "inline_block") {
    std::string_view id{};
    if (!parse_paren_id(line, id)) return {};
    consume_optional_semicolon(line);
    if (!tail_ok(line)) return {};
    return {PragmaKind::Inline, id};
  }

  if (kw == "end_block") {
    // optional "(ID)" — captured by your regex but your logic ignores it
    std::string_view maybe_id{};
    {
      std::string_view tmp = line;
      if (parse_paren_id(tmp, maybe_id)) {
        line = tmp; // accept it, but ignore id
      }
    }
    consume_optional_semicolon(line);
    if (!tail_ok(line)) return {};
    return {PragmaKind::End, {}};
  }

  if (kw == "unroll") {
    consume_optional_semicolon(line);
    if (!tail_ok(line)) return {};
    return {PragmaKind::Unroll, {}};
  }

  return {};
}

} // namespace

memory::string ShaderPreprocessor::preprocess(const memory::string &src) {
  m_blocks.clear();

  const char* const data = src.c_str();
  const std::size_t sz = src.size();

  // -------- Build line table as views into src (no per-line allocations) --------
  memory::vector<std::uint32_t> line_start;
  memory::vector<std::uint32_t> line_len;
  line_start.reserve(512);
  line_len.reserve(512);

  const char* p = data;
  const char* const end = data + sz;

  while (true) {
    const char* nl = static_cast<const char*>(
        memchr(p, '\n', static_cast<std::size_t>(end - p)));
    const char* line_end = nl ? nl : end;

    if (line_end > p && line_end[-1] == '\r') --line_end;

    line_start.push_back(static_cast<std::uint32_t>(p - data));
    line_len.push_back(static_cast<std::uint32_t>(line_end - p));

    if (!nl) break;
    p = nl + 1;

    // match your old split_lines: if ends with '\n', append final empty line
    if (p == end) {
      line_start.push_back(static_cast<std::uint32_t>(end - data));
      line_len.push_back(0);
      break;
    }
  }

  const std::uint32_t n_lines = static_cast<std::uint32_t>(line_start.size());

  auto line_view = [&](std::uint32_t i) noexcept -> std::string_view {
    return std::string_view{data + line_start[i], line_len[i]};
  };

  // -------- Pass 1: capture blocks + build begin->end jump table --------
  struct Beg { memory::string id; std::uint32_t begin_idx; };

  memory::vector<Beg> stack;
  stack.reserve(16);

  std::vector<std::int32_t> end_for_begin(n_lines, -1);

  for (std::uint32_t i = 0; i < n_lines; ++i) {
    const ParsedPragma pp = parse_pragma(line_view(i));

    if (pp.kind == PragmaKind::Begin) {
      stack.push_back(Beg{memory::string(pp.id), i});
      continue;
    }

    if (pp.kind == PragmaKind::End) {
      if (stack.empty()) {
        throw std::runtime_error(
            fmt::format("Stray '#pragma end_block' at line {}", i + 1));
      }

      Beg beg = std::move(stack.back());
      stack.pop_back();

      Block b;
      b.id         = std::move(beg.id);
      b.begin_idx  = beg.begin_idx;
      b.end_idx    = i;
      b.body_begin = b.begin_idx + 1;
      b.body_end   = b.end_idx;
      b.start_ln   = b.body_begin + 1; // 1-based

      m_blocks.push_back(std::move(b));
      end_for_begin[static_cast<std::size_t>(beg.begin_idx)] = static_cast<std::int32_t>(i);
      continue;
    }
  }

  if (!stack.empty()) {
    const Beg &b = stack.back();
    throw std::runtime_error(
        fmt::format("Unterminated block \"{}\" starting at line {}", b.id, b.begin_idx + 1));
  }

  // id -> block index (first captured wins, matching your find_if-on-vector order)
  std::unordered_map<std::string_view, std::uint32_t> block_index;
  block_index.reserve(m_blocks.size() * 2 + 1);

  for (std::uint32_t bi = 0; bi < static_cast<std::uint32_t>(m_blocks.size()); ++bi) {
    const auto& b = m_blocks[bi];
    std::string_view key{b.id.data(), b.id.size()};
    if (block_index.find(key) == block_index.end()) {
      block_index.emplace(key, bi);
    }
  }

  auto get_block_index = [&](std::string_view id) -> std::uint32_t {
    auto it = block_index.find(id);
    if (it == block_index.end()) {
      throw std::runtime_error(fmt::format(
          "#pragma inline_block references unknown block \"{}\"", id));
    }
    return it->second;
  };

  // -------- Output helpers --------
  memory::string out;
  out.reserve(src.size() + 256);

  auto append_resync = [&](std::uint32_t next_original_line_1based) {
    out.append("#line ");
    char buf[32];
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), next_original_line_1based);
    (void)ec;
    out.append(buf, static_cast<std::size_t>(ptr - buf));
    out.push_back('\n');
  };

  static constexpr std::string_view kUnrollSnippet =
      "#ifdef GL_EXT_control_flow_attributes\n[[unroll]]\n#endif\n";

  // -------- Recursive expansion by block index (cycle guard) --------
  std::vector<std::uint32_t> inline_stack;
  inline_stack.reserve(16);

  auto throw_cycle = [&](std::uint32_t next_idx) {
    memory::string chain;
    for (std::uint32_t idx : inline_stack) {
      if (!chain.empty()) chain += " -> ";
      chain += m_blocks[idx].id;
    }
    if (!chain.empty()) chain += " -> ";
    chain += m_blocks[next_idx].id;

    throw std::runtime_error(
        "Recursive #pragma inline_block expansion detected: " + std::string(chain));
  };

  auto expand_block = [&](auto&& self, std::uint32_t blk_idx) -> void {
    if (std::find(inline_stack.begin(), inline_stack.end(), blk_idx) != inline_stack.end()) {
      throw_cycle(blk_idx);
    }
    inline_stack.push_back(blk_idx);

    const Block& blk = m_blocks[blk_idx];

    append_resync(blk.start_ln);

    for (std::uint32_t i = blk.body_begin; i < blk.body_end; ++i) {
      // skip nested definitions using global begin->end table
      const std::int32_t end_idx = end_for_begin[static_cast<std::size_t>(i)];
      if (end_idx != -1) {
        const std::uint32_t j = static_cast<std::uint32_t>(end_idx);
        append_resync(j + 2);
        i = j;
        continue;
      }

      const std::string_view L = line_view(i);
      const ParsedPragma pp = parse_pragma(L);

      if (pp.kind == PragmaKind::Inline) {
        const std::uint32_t nb_idx = get_block_index(pp.id);
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

  // -------- Pass 2: emit file (skip definitions; expand inlines) --------
  for (std::uint32_t i = 0; i < n_lines; ++i) {
    const std::int32_t end_idx = end_for_begin[static_cast<std::size_t>(i)];
    if (end_idx != -1) {
      const std::uint32_t j = static_cast<std::uint32_t>(end_idx);
      append_resync(j + 2);
      i = j;
      continue;
    }

    const std::string_view L = line_view(i);
    const ParsedPragma pp = parse_pragma(L);

    if (pp.kind == PragmaKind::Inline) {
      const std::uint32_t b_idx = get_block_index(pp.id);
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

} // namespace denox::compiler

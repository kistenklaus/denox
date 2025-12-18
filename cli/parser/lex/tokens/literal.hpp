#pragma once

#include "denox/common/types.hpp"
#include "io/Path.hpp"
#include "parser/lex/tokens/duration.hpp"
#include <cassert>
#include <cstdint>
#include <fmt/core.h>
#include <string>
#include <string_view>

class LiteralToken {
public:
  explicit LiteralToken(std::string value) : m_value(std::move(value)) {}

  const std::string &value() const noexcept { return m_value; }
  std::string_view view() const noexcept { return m_value; }
  bool is_unsigned_int() const noexcept { return parse_unsigned().has_value(); }
  bool is_signed_int() const noexcept { return parse_signed().has_value(); }
  std::uint64_t as_unsigned_int() const {
    auto v = parse_unsigned();
    assert(v);
    return *v;
  }
  std::int64_t as_signed_int() const {
    auto v = parse_signed();
    assert(v);
    return *v;
  }
  bool is_duration() const noexcept {
    return Duration::parse(view()).has_value();
  }
  Duration as_duration() const {
    auto d = Duration::parse(view());
    assert(d);
    return *d;
  }
  bool is_target_env() const noexcept { return parse_target_env().has_value(); }
  denox::VulkanApiVersion as_target_env() const {
    auto env = parse_target_env();
    assert(env);
    return *env;
  }
  bool is_layout() const noexcept { return parse_layout().has_value(); }
  denox::Layout as_layout() const {
    auto l = parse_layout();
    assert(l);
    return *l;
  }
  bool is_dtype() const noexcept { return parse_dtype().has_value(); }
  denox::DataType as_dtype() const {
    auto dt = parse_dtype();
    assert(dt);
    return *dt;
  }
  bool is_storage() const noexcept { return parse_storage().has_value(); }
  denox::Storage as_storage() const {
    auto s = parse_storage();
    assert(s);
    return *s;
  }

  bool is_path() const noexcept;
  Path as_path() const;

private:
  std::optional<std::uint64_t> parse_unsigned() const noexcept;

  std::optional<std::int64_t> parse_signed() const noexcept;

  std::optional<denox::VulkanApiVersion> parse_target_env() const noexcept;

  static char to_lower_ascii(char c) noexcept;
  std::optional<denox::Layout> parse_layout() const noexcept;
  std::optional<denox::DataType> parse_dtype() const noexcept;
  std::optional<denox::Storage> parse_storage() const noexcept;

private:
  std::string m_value;
};

template <> struct fmt::formatter<LiteralToken> : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const LiteralToken &lit, FormatContext &ctx) const {
    return fmt::formatter<std::string_view>::format(lit.view(), ctx);
  }
};

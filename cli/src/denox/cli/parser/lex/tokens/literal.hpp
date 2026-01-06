#pragma once

#include "denox/cli/parser/lex/tokens/duration.hpp"
#include "denox/common/TensorDataType.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/common/TensorStorage.hpp"
#include "denox/device_info/ApiVersion.hpp"
#include "denox/io/fs/Path.hpp"
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

  bool is_float() const noexcept { return parse_float().has_value(); }
  float as_float() const noexcept { return *parse_float(); }

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
  denox::ApiVersion as_target_env() const {
    auto env = parse_target_env();
    assert(env);
    return *env;
  }

  bool is_format() const noexcept { return parse_format().has_value(); }
  denox::TensorFormat as_format() const {
    auto l = parse_format();
    assert(l);
    return *l;
  }

  bool is_dtype() const noexcept { return parse_dtype().has_value(); }
  denox::TensorDataType as_dtype() const {
    auto dt = parse_dtype();
    assert(dt);
    return *dt;
  }
  bool is_storage() const noexcept { return parse_storage().has_value(); }
  denox::TensorStorage as_storage() const {
    auto s = parse_storage();
    assert(s);
    return *s;
  }

  bool is_path() const noexcept;
  denox::io::Path as_path() const;

  bool is_bool() const noexcept {
    auto sv = view();
    return sv == "true" || sv == "false" || sv == "1" || sv == "0" ||
           sv == "yes" || sv == "no";
  }

  bool as_bool() const {
    auto sv = view();
    if (sv == "true" || sv == "1" || sv == "yes")
      return true;
    if (sv == "false" || sv == "0" || sv == "no")
      return false;
    assert(false && "Literal::as_bool called on non-bool literal");
    return false;
  }

private:
  std::optional<std::uint64_t> parse_unsigned() const noexcept;

  std::optional<std::int64_t> parse_signed() const noexcept;
  std::optional<float> parse_float() const noexcept;

  std::optional<denox::ApiVersion> parse_target_env() const noexcept;

  static char to_lower_ascii(char c) noexcept;

  std::optional<denox::TensorFormat> parse_format() const noexcept;
  std::optional<denox::TensorDataType> parse_dtype() const noexcept;
  std::optional<denox::TensorStorage> parse_storage() const noexcept;

private:
  std::string m_value;
};

template <>
struct fmt::formatter<LiteralToken> : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const LiteralToken &lit, FormatContext &ctx) const {
    return fmt::formatter<std::string_view>::format(lit.view(), ctx);
  }
};

#pragma once

#include "denox/memory/container/string.hpp"
#include "denox/memory/container/string_view.hpp"
#include <cstdlib>
#include <filesystem>
#include <fmt/format.h>
#include <functional>
#include <system_error>

namespace denox::io {

class Path {
public:
  using size_type = std::uintmax_t;

  Path() = default;
  explicit Path(memory::string_view utf8) : m_native(from_utf8(utf8)) {}

  Path(const Path &) = default;
  Path(Path &&) noexcept = default;
  Path &operator=(const Path &) = default;
  Path &operator=(Path &&) noexcept = default;

  [[nodiscard]] memory::u8string utf8() const;
  [[nodiscard]] memory::string str() const;
  [[nodiscard]] const char *cstr() const;
  [[nodiscard]] bool empty() const noexcept { return m_native.empty(); }
  [[nodiscard]] bool is_absolute() const noexcept;
  [[nodiscard]] bool is_relative() const noexcept;
  [[nodiscard]] memory::string filename() const;
  [[nodiscard]] memory::string stem() const;
  [[nodiscard]] memory::string extension() const;
  [[nodiscard]] Path parent() const;
  [[nodiscard]] Path with_extension(memory::string_view new_ext) const;
  [[nodiscard]] Path with_filename(memory::string_view new_name) const;
  friend Path operator/(const Path &a, const Path &b);
  friend Path operator/(const Path &a, memory::string_view b_utf8);
  Path &operator/=(const Path &rhs);
  Path &operator/=(memory::string_view rhs_utf8);
  static Path join(std::initializer_list<memory::string_view> parts);
  [[nodiscard]] Path normalized() const;
  [[nodiscard]] Path absolute() const;
  [[nodiscard]] Path relative_to(const Path &base) const;
  [[nodiscard]] bool exists() const;
  [[nodiscard]] bool is_file() const;
  [[nodiscard]] bool is_dir() const;
  static Path assets();
  friend bool operator==(const Path &a, const Path &b) noexcept;
  friend bool operator!=(const Path &a, const Path &b) noexcept;
  friend bool operator<(const Path &a, const Path &b) noexcept;

private:
  [[nodiscard]] bool exists(std::error_code &ec) const;
  [[nodiscard]] Path relative_to(const Path &base, std::error_code &ec) const;
  [[nodiscard]] bool is_file(std::error_code &ec) const;
  [[nodiscard]] bool is_dir(std::error_code &ec) const;
  [[nodiscard]] Path absolute(std::error_code &ec) const;
  static Path from_native(const std::filesystem::path &native);
  static std::filesystem::path from_utf8(memory::string_view s);

private:
  std::filesystem::path m_native;
};

} // namespace denox::io

// Optional: hash support for unordered_map/unordered_set
template <> struct std::hash<denox::io::Path> {
  size_t operator()(const denox::io::Path &p) const noexcept {
    return std::hash<std::u8string>{}(p.utf8());
  }
};

template <>
struct fmt::formatter<denox::io::Path> : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const denox::io::Path &p, FormatContext &ctx) const {
    return fmt::formatter<std::string_view>::format(p.str(), ctx);
  }
};

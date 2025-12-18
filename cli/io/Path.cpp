#include "io/Path.hpp"

std::u8string Path::utf8() const {
  auto u8 = m_native.u8string();
  return std::u8string(u8.begin(), u8.end());
}

std::string Path::str() const {
  auto u8 = m_native.u8string();
  return std::string(u8.begin(), u8.end());
}

std::string Path::filename() const {
  return Path::from_native(m_native.filename()).str();
}

std::string Path::stem() const {
  return Path::from_native(m_native.stem()).str();
}

std::string Path::extension() const {
  return Path::from_native(m_native.extension()).str();
}

Path Path::with_extension(std::string_view new_ext) const {
  auto tmp = m_native;
  tmp.replace_extension(from_utf8(new_ext));
  return from_native(tmp);
}

Path Path::with_filename(std::string_view new_name) const {
  auto tmp = m_native;
  tmp.replace_filename(from_utf8(new_name));
  return from_native(tmp);
}

Path operator/(const Path &a, const Path &b) {
  return Path::from_native(a.m_native / b.m_native);
}

Path operator/(const Path &a, std::string_view b_utf8) {
  return Path::from_native(a.m_native / Path::from_utf8(b_utf8));
}

Path &Path::operator/=(const Path &rhs) {
  m_native /= rhs.m_native;
  return *this;
}

Path &Path::operator/=(std::string_view rhs_utf8) {
  m_native /= from_utf8(rhs_utf8);
  return *this;
}

Path Path::join(std::initializer_list<std::string_view> parts) {
  std::filesystem::path acc;
  for (auto s : parts)
    acc /= from_utf8(s);
  return from_native(acc);
}

Path Path::normalized() const {
  return from_native(m_native.lexically_normal());
}

Path Path::absolute() const {
  std::error_code ec;
  auto r = absolute(ec);
  if (ec)
    throw std::filesystem::filesystem_error("absolute()", m_native, ec);
  return r;
}

Path Path::relative_to(const Path &base) const {
  std::error_code ec;
  auto r = relative_to(base, ec);
  if (ec)
    throw std::filesystem::filesystem_error("relative_to()", m_native, ec);
  return r;
}

bool Path::exists() const {
  std::error_code ec;
  return std::filesystem::exists(m_native, ec);
}

bool Path::is_file() const {
  std::error_code ec;
  return std::filesystem::is_regular_file(m_native, ec);
}

bool Path::is_dir() const {
  std::error_code ec;
  return std::filesystem::is_directory(m_native, ec);
}

Path Path::cwd() {
  std::error_code ec;
  auto p = std::filesystem::current_path(ec);
  if (ec)
    return {};
  return from_native(p);
}

bool Path::exists(std::error_code &ec) const {
  return std::filesystem::exists(m_native, ec);
}

Path Path::relative_to(const Path &base, std::error_code &ec) const {
  return from_native(std::filesystem::relative(m_native, base.m_native, ec)
                         .lexically_normal());
}

bool Path::is_file(std::error_code &ec) const {
  return std::filesystem::is_regular_file(m_native, ec);
}

bool Path::is_dir(std::error_code &ec) const {
  return std::filesystem::is_directory(m_native, ec);
}

Path Path::absolute(std::error_code &ec) const {
  auto a = std::filesystem::absolute(m_native, ec);
  return from_native(a.lexically_normal());
}

Path Path::from_native(const std::filesystem::path &native) {
  Path r;
  r.m_native = native;
  return r;
}

std::filesystem::path Path::from_utf8(std::string_view s) {
#if defined(_WIN32)
  // Ensure UTF-8 correctness: go through u8path using char8_t
  std::u8string u8;
  u8.reserve(s.size());
  for (unsigned char c : s)
    u8.push_back(static_cast<char8_t>(c));
  return std::filesystem::u8path(u8);
#else
  // POSIX: narrow is UTF-8 by convention; use iterator range (no null
  // required)
  return std::filesystem::path(s.begin(), s.end());
#endif
}

bool Path::is_absolute() const noexcept { return m_native.is_absolute(); }

bool Path::is_relative() const noexcept { return m_native.is_relative(); }

Path Path::parent() const { return from_native(m_native.parent_path()); }

bool operator==(const Path &a, const Path &b) noexcept {
  return a.m_native == b.m_native;
}

bool operator!=(const Path &a, const Path &b) noexcept { return !(a == b); }

bool operator<(const Path &a, const Path &b) noexcept {
  return a.m_native < b.m_native;
}

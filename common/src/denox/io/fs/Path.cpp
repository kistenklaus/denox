#include "denox/io/fs/Path.hpp"

#if defined(__linux__)
#include <limits.h>
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#if defined(_WIN32)
#include <windows.h>
#endif

namespace denox::io {

memory::u8string Path::utf8() const {
  auto u8 = m_native.u8string();
  return memory::u8string(u8.begin(), u8.end());
}

memory::string Path::str() const {
  auto u8 = m_native.u8string();
  return memory::string(u8.begin(), u8.end());
}

const char *Path::cstr() const { return m_native.c_str(); }

memory::string Path::filename() const {
  return Path::from_native(m_native.filename()).str();
}

memory::string Path::stem() const {
  return Path::from_native(m_native.stem()).str();
}

memory::string Path::extension() const {
  return Path::from_native(m_native.extension()).str();
}

Path Path::with_extension(memory::string_view new_ext) const {
  auto tmp = m_native;
  tmp.replace_extension(from_utf8(new_ext));
  return from_native(tmp);
}

Path Path::with_filename(memory::string_view new_name) const {
  auto tmp = m_native;
  tmp.replace_filename(from_utf8(new_name));
  return from_native(tmp);
}

Path operator/(const Path &a, const Path &b) {
  return Path::from_native(a.m_native / b.m_native);
}

Path operator/(const Path &a, memory::string_view b_utf8) {
  return Path::from_native(a.m_native / Path::from_utf8(b_utf8));
}

Path &Path::operator/=(const Path &rhs) {
  m_native /= rhs.m_native;
  return *this;
}

Path &Path::operator/=(memory::string_view rhs_utf8) {
  m_native /= from_utf8(rhs_utf8);
  return *this;
}

Path Path::join(std::initializer_list<memory::string_view> parts) {
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

Path Path::assets() {
  if (const char *env = std::getenv("DENOX_HOME")) {
    Path p(env);
    if (!p.empty() && p.exists() && p.is_dir())
      return p.normalized();
  }

  std::error_code ec;
#if defined(__linux__)
  char buffer[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer));
  if (len <= 0) {
    ec = std::error_code(errno, std::generic_category());
    return {};
  }
  auto exe = Path::from_native(
      std::filesystem::path(std::string(buffer, static_cast<size_t>(len))));
#elif defined(__APPLE__)
  uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  std::string buffer(size, '\0');
  if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
    ec = std::make_error_code(std::errc::io_error);
    return {};
  }
  auto exe =
      Path::from_native(std::filesystem::path(buffer).lexically_normal());
#elif
  wchar_t buffer[MAX_PATH];
  DWORD len = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
  if (len == 0) {
    ec = std::error_code(GetLastError(), std::system_category());
    return {};
  }
  auto exe = Path::from_native(std::filesystem::path(buffer));
#endif

  if (!ec && !exe.empty()) {
    Path home = exe.parent().parent() / "share/denox";
    if (!home.empty() && home.exists() && home.is_dir())
      return home.normalized();
  }

  return {};
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

std::filesystem::path Path::from_utf8(memory::string_view s) {
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
} // namespace denox::io

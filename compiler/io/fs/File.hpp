#pragma once

#include "memory/container/string.hpp"
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <span>
#include <system_error>
#include <type_traits>

#if defined(_WIN32)
#include <io.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "io/fs/Path.hpp"

namespace denox::io {

class File {
public:
  using size_type = std::uint64_t;

  enum class OpenMode : unsigned {
    Read = 1u << 0,     // open existing for reading
    Write = 1u << 1,    // open for writing
    Append = 1u << 2,   // append at end
    Truncate = 1u << 3, // truncate on open
    Create = 1u << 4,   // create if not exist
    Readwrite = Read | Write
  };

  enum class SeekWhence : int { Set = 0, Cur = 1, End = 2 };

  File() = default;

  File(const File &) = delete;
  File &operator=(const File &) = delete;

  File(File &&other) noexcept;
  File &operator=(File &&other) noexcept;
  ~File() { close(); }
  static File open(const Path &p, OpenMode mode);
  void close() noexcept;
  [[nodiscard]] bool is_open() const noexcept { return m_file != nullptr; }
  [[nodiscard]] const Path &path() const noexcept { return m_path; }
  size_type read(std::span<std::byte> dst);
  void read_exact(std::span<std::byte> dst);
  size_type write(std::span<const std::byte> src);
  void write_exact(std::span<const std::byte> src);
  bool flush();
  void seek(std::int64_t offset, SeekWhence whence = SeekWhence::Set);
  [[nodiscard]] std::uint64_t tell() const;
  [[nodiscard]] size_type size() const;

private:
  bool open_ec(const Path &p, OpenMode mode, std::error_code &ec);
  size_type read_ec(std::span<std::byte> dst, std::error_code &ec) noexcept;
  size_type write_ec(std::span<const std::byte> src,
                     std::error_code &ec) noexcept;
  bool flush_ec(std::error_code &ec) noexcept;
  bool seek_ec(std::int64_t offset, SeekWhence whence,
               std::error_code &ec) noexcept;
  std::uint64_t tell_ec(std::error_code &ec) const noexcept;
  size_type size_ec(std::error_code &ec) const noexcept;
  static bool has(OpenMode m, OpenMode f);
  static const char *mode_cstr(OpenMode m);
#if defined(_WIN32)
  static const wchar_t *mode_wcstr(OpenMode m);
#endif
  void clear_error() const noexcept;

private:
  FILE *m_file = nullptr;
  Path m_path{};
};

inline File::OpenMode operator|(File::OpenMode a, File::OpenMode b) {
  using U = std::underlying_type_t<File::OpenMode>;
  return static_cast<File::OpenMode>(static_cast<U>(a) | static_cast<U>(b));
}
inline File::OpenMode &operator|=(File::OpenMode &a, File::OpenMode b) {
  a = a | b;
  return a;
}

} // namespace denox::io

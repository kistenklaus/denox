#include "denox/io/fs/File.hpp"
#include "denox/memory/container/span.hpp"

namespace denox::io {

File::File(File &&other) noexcept {
  m_file = other.m_file;
  other.m_file = nullptr;
  m_path = std::move(other.m_path);
}

File &File::operator=(File &&other) noexcept {
  if (this != &other) {
    close();
    m_file = other.m_file;
    other.m_file = nullptr;
    m_path = std::move(other.m_path);
  }
  return *this;
}

File File::open(const Path &p, OpenMode mode) {
  std::error_code ec;
  File f;
  if (!f.open_ec(p, mode, ec))
    throw std::system_error(ec, fmt::format("File::open(\"{}\")", p));
  return f;
}

void File::close() noexcept {
  if (m_file) {
    std::fclose(m_file);
    m_file = nullptr;
  }
}

File::size_type File::read(memory::span<std::byte> dst) {
  std::error_code ec;
  auto n = read_ec(dst, ec);
  if (ec)
    throw std::system_error(ec, "File::read");
  return n;
}

void File::read_exact(memory::span<std::byte> dst) {
  std::byte *it = dst.data();
  size_type remaining = dst.size();
  while (remaining) {
    std::error_code ec;
    size_type n = read_ec({it, static_cast<size_t>(remaining)}, ec);
    if (ec)
      throw std::system_error(ec, "File::read_exact");
    if (n == 0) {
      // Short read (likely EOF)
      throw std::system_error(
          std::make_error_code(std::errc::result_out_of_range),
          "File::read_exact: unexpected EOF");
    }
    it += n;
    remaining -= n;
  }
}

File::size_type File::write(memory::span<const std::byte> src) {
  std::error_code ec;
  auto n = write_ec(src, ec);
  if (ec)
    throw std::system_error(ec, "File::write");
  return n;
}

void File::write_exact(memory::span<const std::byte> src) {
  const std::byte *it = src.data();
  size_type remaining = src.size();
  while (remaining) {
    std::error_code ec;
    size_type n = write_ec({it, static_cast<size_t>(remaining)}, ec);
    if (ec)
      throw std::system_error(ec, "File::write_exact");
    if (n == 0) {
      throw std::system_error(std::make_error_code(std::errc::io_error),
                              "File::write_exact: short write");
    }
    it += n;
    remaining -= n;
  }
}

bool File::flush() {
  std::error_code ec;
  if (!flush_ec(ec))
    throw std::system_error(ec, "File::flush");
  return true;
}

void File::seek(std::int64_t offset, SeekWhence whence) {
  std::error_code ec;
  if (!seek_ec(offset, whence, ec))
    throw std::system_error(ec, "File::seek");
}

std::uint64_t File::tell() const {
  std::error_code ec;
  auto pos = tell_ec(ec);
  if (ec)
    throw std::system_error(ec, "File::tell");
  return pos;
}

File::size_type File::size() const {
  std::error_code ec;
  auto s = size_ec(ec);
  if (ec)
    throw std::system_error(ec, "File::size");
  return s;
}

bool File::open_ec(const Path &p, OpenMode mode, std::error_code &ec) {
  close();
  m_path = p;
#if defined(_WIN32)
  // Build wide path from UTF-8 via u8path (safe for UTF-8)
  std::u8string u8;
  {
    auto mem = p.utf8();
    u8.reserve(mem.size());
    for (unsigned char c : mem)
      u8.push_back(static_cast<char8_t>(c));
  }
  std::filesystem::path fsp = std::filesystem::u8path(u8);
  const wchar_t *wpath = fsp.c_str();
  const wchar_t *wmode = mode_wcstr(mode);
  FILE *f = _wfopen(wpath, wmode);
#else
  memory::string s = p.str();
  const char *cmode = mode_cstr(mode);
  FILE *f = std::fopen(s.c_str(), cmode);
#endif
  if (!f) {
    ec = std::error_code(errno, std::generic_category());
    m_path = {};
    return false;
  }
  m_file = f;
  ec.clear();
  return true;
}

File::size_type File::read_ec(memory::span<std::byte> dst,
                              std::error_code &ec) noexcept {
  if (!m_file) {
    ec = std::make_error_code(std::errc::bad_file_descriptor);
    return 0;
  }
  size_t n = std::fread(dst.data(), 1, dst.size(), m_file);
  if (n < dst.size() && std::ferror(m_file)) {
    ec = std::error_code(errno, std::generic_category());
    clear_error();
  } else {
    ec.clear();
  }
  return static_cast<size_type>(n);
}

File::size_type File::write_ec(memory::span<const std::byte> src,
                               std::error_code &ec) noexcept {
  if (!m_file) {
    ec = std::make_error_code(std::errc::bad_file_descriptor);
    return 0;
  }
  size_t n = std::fwrite(src.data(), 1, src.size(), m_file);
  if (n < src.size()) {
    ec = std::error_code(errno, std::generic_category());
    clear_error();
  } else {
    ec.clear();
  }
  return static_cast<size_type>(n);
}

bool File::flush_ec(std::error_code &ec) noexcept {
  if (!m_file) {
    ec = std::make_error_code(std::errc::bad_file_descriptor);
    return false;
  }
  if (std::fflush(m_file) == 0) {
    ec.clear();
    return true;
  }
  ec = std::error_code(errno, std::generic_category());
  return false;
}

bool File::seek_ec(std::int64_t offset, SeekWhence whence,
                   std::error_code &ec) noexcept {
  if (!m_file) {
    ec = std::make_error_code(std::errc::bad_file_descriptor);
    return false;
  }
  int origin = (whence == SeekWhence::Set)   ? SEEK_SET
               : (whence == SeekWhence::Cur) ? SEEK_CUR
                                             : SEEK_END;
#if defined(_WIN32)
  if (_fseeki64(m_fp, static_cast<long long>(offset), origin) == 0) {
    ec.clear();
    return true;
  }
#else
  if (::fseeko(m_file, static_cast<off_t>(offset), origin) == 0) {
    ec.clear();
    return true;
  }
#endif
  ec = std::error_code(errno, std::generic_category());
  clear_error();
  return false;
}

std::uint64_t File::tell_ec(std::error_code &ec) const noexcept {
  if (!m_file) {
    ec = std::make_error_code(std::errc::bad_file_descriptor);
    return 0;
  }
#if defined(_WIN32)
  auto p = _ftelli64(m_fp);
  if (p >= 0) {
    ec.clear();
    return static_cast<std::uint64_t>(p);
  }
#else
  auto p = ::ftello(m_file);
  if (p >= 0) {
    ec.clear();
    return static_cast<std::uint64_t>(p);
  }
#endif
  ec = std::error_code(errno, std::generic_category());
  return 0;
}

File::size_type File::size_ec(std::error_code &ec) const noexcept {
  if (!m_file) {
    ec = std::make_error_code(std::errc::bad_file_descriptor);
    return 0;
  }

#if defined(_WIN32)
  // Use file descriptor -> HANDLE -> GetFileSizeEx (no heap allocations).
  int fd = _fileno(m_fp);
  if (fd == -1) {
    ec = std::error_code(errno, std::generic_category());
    return 0;
  }
  HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (h == INVALID_HANDLE_VALUE) {
    ec = std::error_code(errno, std::generic_category());
    return 0;
  }
  LARGE_INTEGER li{};
  if (GetFileSizeEx(h, &li)) {
    ec.clear();
    return static_cast<size_type>(li.QuadPart);
  }
  ec =
      std::error_code(static_cast<int>(GetLastError()), std::system_category());
  return 0;
#else
  // Use fstat on the FILE* (no heap allocations).
  int fd = fileno(m_file);
  if (fd == -1) {
    ec = std::error_code(errno, std::generic_category());
    return 0;
  }
  struct stat st{};
  if (fstat(fd, &st) == 0) {
    ec.clear();
    return static_cast<size_type>(st.st_size);
  }
  ec = std::error_code(errno, std::generic_category());
  return 0;
#endif
}

bool File::has(OpenMode m, OpenMode f) {
  using U = std::underlying_type_t<OpenMode>;
  return (static_cast<U>(m) & static_cast<U>(f)) != 0;
}

const char *File::mode_cstr(OpenMode m) {
  const bool rd = has(m, OpenMode::Read);
  const bool wr = has(m, OpenMode::Write);
  const bool ap = has(m, OpenMode::Append);
  const bool tr = has(m, OpenMode::Truncate);
  const bool cr = has(m, OpenMode::Create);

  if (ap)
    return (rd || wr) ? "a+b" : "ab";
  if (wr && (tr || cr))
    return rd ? "w+b" : "wb";
  if (wr && rd)
    return "r+b";
  if (wr)
    return "r+b";
  return "rb";
}

#if defined(_WIN32)
const wchar_t *File::mode_wcstr(OpenMode m) {
  const bool rd = has(m, OpenMode::Read);
  const bool wr = has(m, OpenMode::write);
  const bool ap = has(m, OpenMode::Append);
  const bool tr = has(m, OpenMode::Truncate);
  const bool cr = has(m, OpenMode::Create);

  if (ap)
    return (rd || wr) ? L"a+b" : L"ab";
  if (wr && (tr || cr))
    return rd ? L"w+b" : L"wb";
  if (wr && rd)
    return L"r+b";
  if (wr)
    return L"r+b"; // must exist
  return L"rb";
}
#endif

void File::clear_error() const noexcept {
  if (m_file)
    std::clearerr(m_file);
}
} // namespace denox::io

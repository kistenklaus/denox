#pragma once

#include "denox/io/fs/Path.hpp"
#include <system_error>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace denox::io {

class FileLock {
public:
  FileLock() noexcept = default;

  FileLock(const FileLock &) = delete;
  FileLock &operator=(const FileLock &) = delete;

  FileLock(FileLock &&other) noexcept { move_from(std::move(other)); }

  FileLock &operator=(FileLock &&other) noexcept {
    if (this != &other) {
      release();
      move_from(std::move(other));
    }
    return *this;
  }

  ~FileLock() { release(); }

  static FileLock lock(const Path &path) {
    FileLock l;
    l.acquire(path);
    return l;
  }

  [[nodiscard]] bool owns_lock() const noexcept {
#if defined(_WIN32)
    return m_handle != INVALID_HANDLE_VALUE;
#else
    return m_fd != -1;
#endif
  }

private:
  FileLock(const Path &) = delete;

  void move_from(FileLock &&other) noexcept {
#if defined(_WIN32)
    m_handle = std::exchange(other.m_handle, INVALID_HANDLE_VALUE);
#else
    m_fd = std::exchange(other.m_fd, -1);
#endif
  }

  void acquire(const Path &path) {
#if defined(_WIN32)

    std::filesystem::path fsp = std::filesystem::u8path(path.utf8());
    const wchar_t *wpath = fsp.c_str();

    HANDLE h =
        CreateFileW(wpath, GENERIC_READ | GENERIC_WRITE,
                    0, // no sharing -> exclusive
                    nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);

    if (h == INVALID_HANDLE_VALUE) {
      throw std::system_error(static_cast<int>(GetLastError()),
                              std::system_category(),
                              "FileLock: CreateFileW failed");
    }

    // Lock entire file (non-blocking)
    OVERLAPPED ov{};
    if (!LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0,
                    MAXDWORD, MAXDWORD, &ov)) {
      DWORD err = GetLastError();
      CloseHandle(h);
      throw std::system_error(static_cast<int>(err), std::system_category(),
                              "FileLock: LockFileEx failed (already locked?)");
    }

    m_handle = h;

#else

    auto s = path.str();

    int fd = ::open(s.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
      throw std::system_error(errno, std::generic_category(),
                              "FileLock: open failed");
    }

    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
      int err = errno;
      ::close(fd);
      throw std::system_error(err, std::generic_category(),
                              "FileLock: flock failed (already locked?)");
    }

    m_fd = fd;

#endif
  }

  void release() noexcept {
#if defined(_WIN32)
    if (m_handle != INVALID_HANDLE_VALUE) {
      OVERLAPPED ov{};
      UnlockFileEx(m_handle, 0, MAXDWORD, MAXDWORD, &ov);
      CloseHandle(m_handle);
      m_handle = INVALID_HANDLE_VALUE;
    }
#else
    if (m_fd != -1) {
      flock(m_fd, LOCK_UN);
      ::close(m_fd);
      m_fd = -1;
    }
#endif
  }

private:
#if defined(_WIN32)
  HANDLE m_handle = INVALID_HANDLE_VALUE;
#else
  int m_fd = -1;
#endif
};

} // namespace denox::io

#include "denox/cli/io/Pipe.hpp"

std::vector<std::byte> Pipe::read_all() {
  std::vector<std::byte> data;

  constexpr std::size_t chunk_size = 4096;
  std::byte buffer[chunk_size];

  while (true) {
    std::size_t n = std::fread(buffer, 1, chunk_size, stdin);
    if (n > 0) {
      data.insert(data.end(), buffer, buffer + n);
    }
    if (n < chunk_size) {
      if (std::feof(stdin)) {
        break;
      }
      if (std::ferror(stdin)) {
        throw std::runtime_error("failed to read from stdin");
      }
    }
  }

  return data;
}

size_t Pipe::read(denox::memory::span<std::byte> data) {
  if (data.empty())
    return 0;

  const std::size_t n = std::fread(data.data(), 1, data.size(), stdin);

  if (n == 0 && std::ferror(stdin)) {
    throw std::runtime_error("failed to read from stdin");
  }

  return n; // 0 means EOF
}
void Pipe::read_exact(denox::memory::span<std::byte> data) {
  std::byte *ptr = data.data();
  std::size_t remaining = data.size();

  while (remaining > 0) {
    std::size_t n = read({ptr, remaining});
    if (n == 0) {
      throw std::runtime_error("unexpected EOF on stdin");
    }
    ptr += n;
    remaining -= n;
  }
}
bool Pipe::eof() const noexcept { return std::feof(stdin); }

size_t Pipe::write(std::span<const std::byte> data) {
  if (data.empty())
    return 0;

  const std::size_t written = std::fwrite(data.data(), 1, data.size(), stdout);

  if (written == 0 && std::ferror(stdout)) {
    throw std::runtime_error("failed to write to stdout");
  }

  return written;
}

void Pipe::write_exact(std::span<const std::byte> data) {
  const std::byte *ptr = data.data();
  std::size_t remaining = data.size();

  while (remaining > 0) {
    const std::size_t written = std::fwrite(ptr, 1, remaining, stdout);

    if (written == 0) {
      if (std::ferror(stdout)) {
        throw std::runtime_error("failed to write to stdout");
      }
      // fwrite returning 0 without ferror is unexpected
      throw std::runtime_error("unexpected short write to stdout");
    }

    ptr += written;
    remaining -= written;
  }

  std::fflush(stdout);
}

void Pipe::flush() {
  if (std::fflush(stdout) != 0) {
    throw std::runtime_error("failed to flush stdout");
  }
}


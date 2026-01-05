#pragma once

#include <cstddef>
#include <cstdio>
#include <fmt/core.h>
#include <span>
#include <stdexcept>
#include <vector>

struct Pipe {
  // Read all bytes from stdin until EOF
  static std::vector<std::byte> read_all() {
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

  // Write all bytes to stdout
  static void write_all(std::span<const std::byte> data) {
    const std::byte *ptr = data.data();
    std::size_t remaining = data.size();

    while (remaining > 0) {
      std::size_t written = std::fwrite(ptr, 1, remaining, stdout);

      if (written == 0) {
        throw std::runtime_error("failed to write to stdout");
      }

      ptr += written;
      remaining -= written;
    }

    std::fflush(stdout);
  }
};

template <> struct fmt::formatter<Pipe> : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const Pipe &, FormatContext &ctx) const {
    return fmt::formatter<std::string_view>::format("-", ctx);
  }
};

#pragma once

#include "denox/memory/container/span.hpp"
#include <cstddef>
#include <cstdio>
#include <fmt/core.h>
#include <span>
#include <vector>

struct Pipe {
  size_t read(denox::memory::span<std::byte> data);
  void read_exact(denox::memory::span<std::byte> data);
  bool eof() const noexcept;
  std::vector<std::byte> read_all();
  size_t write(std::span<const std::byte> data);
  void write_exact(std::span<const std::byte> data);
  void flush();
};

template <> struct fmt::formatter<Pipe> : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const Pipe &, FormatContext &ctx) const {
    return fmt::formatter<std::string_view>::format("-", ctx);
  }
};

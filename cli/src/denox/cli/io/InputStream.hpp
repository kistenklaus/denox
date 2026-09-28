#pragma once

#include "denox/cli/io/IOEndpoint.hpp"
#include "denox/cli/io/Pipe.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/io/fs/File.hpp"
#include "denox/memory/container/vector.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <variant>

class InputStream {
public:
  explicit InputStream(IOEndpoint endpoint)
      : m_source([&]() -> std::variant<Pipe, denox::io::File> {
          switch (endpoint.kind()) {
          case IOEndpointKind::Path:
            return denox::io::File::open(endpoint.path(),
                                         denox::io::File::OpenMode::Read);
          case IOEndpointKind::Pipe:
            return Pipe{};
          }
          denox::diag::unreachable();
        }()) {}

  std::span<const std::byte> peek(std::size_t count) {
    if (count > m_buffer.size()) {
      throw std::invalid_argument("InputStream::peek supports at most 8 bytes");
    }

    while (m_buffer_end - m_buffer_begin < count && !m_source_eof) {
      const std::size_t available = m_buffer_end - m_buffer_begin;
      std::move(m_buffer.begin() + m_buffer_begin,
                m_buffer.begin() + m_buffer_end,
                m_buffer.begin());
      m_buffer_begin = 0;
      m_buffer_end = available;

      const std::size_t n = read_source(
          {m_buffer.data() + m_buffer_end, count - available});
      if (n == 0) {
        m_source_eof = true;
      } else {
        m_buffer_end += n;
      }
    }

    return {m_buffer.data() + m_buffer_begin,
            m_buffer_end - m_buffer_begin};
  }

  std::size_t read(std::span<std::byte> dst) {
    if (dst.empty()) return 0;

    const std::size_t buffered = m_buffer_end - m_buffer_begin;
    const std::size_t n = std::min(buffered, dst.size());

    std::copy_n(m_buffer.data() + m_buffer_begin, n, dst.data());
    m_buffer_begin += n;

    if (m_buffer_begin == m_buffer_end) {
      m_buffer_begin = 0;
      m_buffer_end = 0;
    }

    if (n != 0) return n;
    if (m_source_eof) return 0;

    const std::size_t from_source = read_source(dst);
    if (from_source == 0) m_source_eof = true;
    return from_source;
  }

  void read_exact(std::span<std::byte> dst) {
    while (!dst.empty()) {
      const std::size_t n = read(dst);
      if (n == 0) {
        throw std::runtime_error("Unexpected EOF");
      }
      dst = dst.subspan(n);
    }
  }

  denox::memory::vector<std::byte> read_all() {
    denox::memory::vector<std::byte> result;
    std::array<std::byte, 4096> chunk;

    while (const std::size_t n = read(chunk)) {
      result.insert(result.end(), chunk.begin(), chunk.begin() + n);
    }
    return result;
  }

  [[nodiscard]] bool eof() const noexcept {
    return m_buffer_begin == m_buffer_end &&
           (m_source_eof ||
            (std::holds_alternative<Pipe>(m_source) &&
             std::get<Pipe>(m_source).eof()));
  }

private:
  std::size_t read_source(std::span<std::byte> dst) {
    if (std::holds_alternative<Pipe>(m_source)) {
      return std::get<Pipe>(m_source).read(dst);
    }
    return std::get<denox::io::File>(m_source).read(dst);
  }

  std::variant<Pipe, denox::io::File> m_source;
  std::array<std::byte, 8> m_buffer{};
  std::size_t m_buffer_begin = 0;
  std::size_t m_buffer_end = 0;
  bool m_source_eof = false;
};

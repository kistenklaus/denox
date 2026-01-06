#pragma once

#include "denox/cli/io/IOEndpoint.hpp"
#include "denox/cli/io/Pipe.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/io/fs/File.hpp"

#include <span>
#include <variant>

class OutputStream {
public:
  explicit OutputStream(IOEndpoint endpoint)
      : m_source([&]() -> std::variant<Pipe, denox::io::File> {
          switch (endpoint.kind()) {
          case IOEndpointKind::Path:
            return denox::io::File::open(
                endpoint.path(), denox::io::File::OpenMode::Write |
                                     denox::io::File::OpenMode::Create |
                                     denox::io::File::OpenMode::Truncate);
          case IOEndpointKind::Pipe:
            return Pipe{};
          }
        }()) {}

  std::size_t write(std::span<const std::byte> src) {
    if (std::holds_alternative<Pipe>(m_source)) {
      return std::get<Pipe>(m_source).write(src);
    } else if (std::holds_alternative<denox::io::File>(m_source)) {
      return std::get<denox::io::File>(m_source).write(src);
    } else {
      denox::diag::unreachable();
    }
  }

  void write_exact(std::span<const std::byte> src) {
    if (std::holds_alternative<Pipe>(m_source)) {
      std::get<Pipe>(m_source).write_exact(src);
    } else if (std::holds_alternative<denox::io::File>(m_source)) {
      std::get<denox::io::File>(m_source).write_exact(src);
    } else {
      denox::diag::unreachable();
    }
  }

  void flush() {
    if (std::holds_alternative<Pipe>(m_source)) {
      std::get<Pipe>(m_source).flush();
    } else if (std::holds_alternative<denox::io::File>(m_source)) {
      std::get<denox::io::File>(m_source).flush();
    } else {
      denox::diag::unreachable();
    }
  }

private:
  std::variant<Pipe, denox::io::File> m_source;
};

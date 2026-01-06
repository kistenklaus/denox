#pragma once

#include "denox/cli/io/IOEndpoint.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/io/fs/File.hpp"
#include <variant>
class InputStream {
public:
  InputStream(IOEndpoint endpoint)
      : m_source([&]() -> std::variant<Pipe, denox::io::File> {
          switch (endpoint.kind()) {
          case IOEndpointKind::Path:
            return denox::io::File::open(endpoint.path(),
                                         denox::io::File::OpenMode::Read);
          case IOEndpointKind::Pipe:
            return Pipe{};
          }
        }()) {}
  std::size_t read(std::span<std::byte> dst) {
    if (std::holds_alternative<Pipe>(m_source)) {
      return std::get<Pipe>(m_source).read(dst);
    } else if (std::holds_alternative<denox::io::File>(m_source)) {
      return std::get<denox::io::File>(m_source).read(dst);
    } else {
      denox::diag::unreachable();
    }
  }
  void read_exact(std::span<std::byte> dst) {
    if (std::holds_alternative<Pipe>(m_source)) {
      std::get<Pipe>(m_source).read_exact(dst);
    } else if (std::holds_alternative<denox::io::File>(m_source)) {
      std::get<denox::io::File>(m_source).read_exact(dst);
    } else {
      denox::diag::unreachable();
    }
  }
  [[nodiscard]] bool eof() const noexcept {
    if (std::holds_alternative<Pipe>(m_source)) {
      return std::get<Pipe>(m_source).eof();
    } else if (std::holds_alternative<denox::io::File>(m_source)) {
      return false;
    } else {
      denox::diag::unreachable();
    }
  }

private:
  std::variant<Pipe, denox::io::File> m_source;
};

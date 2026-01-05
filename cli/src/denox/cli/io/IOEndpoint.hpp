#pragma once

#include "denox/cli/io/Pipe.hpp"
#include "denox/io/fs/Path.hpp"

#include <cassert>
#include <fmt/core.h>
#include <variant>

enum class IOEndpointKind {
  Path,
  Pipe,
};

class IOEndpoint {
public:
  IOEndpoint(denox::io::Path path) noexcept : m_value(std::move(path)) {}

  IOEndpoint(Pipe pipe) noexcept : m_value(pipe) {}

  IOEndpointKind kind() const noexcept {
    if (std::holds_alternative<denox::io::Path>(m_value))
      return IOEndpointKind::Path;
    if (std::holds_alternative<Pipe>(m_value))
      return IOEndpointKind::Pipe;

    std::abort();
  }

  const denox::io::Path &path() const noexcept {
    assert(kind() == IOEndpointKind::Path);
    return std::get<denox::io::Path>(m_value);
  }

  Pipe pipe() const noexcept {
    assert(kind() == IOEndpointKind::Pipe);
    return std::get<Pipe>(m_value);
  }

private:
  std::variant<denox::io::Path, Pipe> m_value;
};

template <>
struct fmt::formatter<IOEndpoint> : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const IOEndpoint &io, FormatContext &ctx) const {
    switch (io.kind()) {
    case IOEndpointKind::Path:
      return fmt::formatter<std::string_view>::format(io.path().str(), ctx);

    case IOEndpointKind::Pipe:
      return fmt::formatter<std::string_view>::format(io.pipe(), ctx);
    }

    std::abort();
  }
};

#pragma once

#include "io/Path.hpp"
#include "io/Pipe.hpp"

#include <fmt/core.h>
#include <cassert>
#include <variant>

enum class IOEndpointKind {
  Path,
  Pipe,
};

class IOEndpoint {
public:
  IOEndpoint(Path path) noexcept
    : m_value(std::move(path)) {}

  IOEndpoint(Pipe pipe) noexcept
    : m_value(pipe) {}

  IOEndpointKind kind() const noexcept {
    if (std::holds_alternative<Path>(m_value))
      return IOEndpointKind::Path;
    if (std::holds_alternative<Pipe>(m_value))
      return IOEndpointKind::Pipe;

    std::abort();
  }

  const Path& path() const noexcept {
    assert(kind() == IOEndpointKind::Path);
    return std::get<Path>(m_value);
  }

  Pipe pipe() const noexcept {
    assert(kind() == IOEndpointKind::Pipe);
    return std::get<Pipe>(m_value);
  }

private:
  std::variant<Path, Pipe> m_value;
};

template <>
struct fmt::formatter<IOEndpoint> : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(const IOEndpoint& io, FormatContext& ctx) const {
    switch (io.kind()) {
      case IOEndpointKind::Path:
        return fmt::formatter<std::string_view>::format(
          io.path().str(), ctx);

      case IOEndpointKind::Pipe:
        return fmt::formatter<std::string_view>::format(
          io.pipe(), ctx);
    }

    std::abort();
  }
};

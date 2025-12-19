#pragma once

#include <optional>
namespace denox::memory {

template <typename T> using optional = std::optional<T>;
static constexpr std::nullopt_t nullopt = std::nullopt;

} // namespace denox::memory

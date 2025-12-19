#pragma once

#include <memory>
namespace denox::memory {

template <typename T> using shared_ptr = std::shared_ptr<T>;

template <typename T> using poly_shared_ptr = std::shared_ptr<T>;
} // namespace denox::memory

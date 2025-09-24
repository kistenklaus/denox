#pragma once

#include <span>
namespace denox::memory {

template<typename T, std::size_t Extent = std::dynamic_extent>
using span = std::span<T, Extent>;

}

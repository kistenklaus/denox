#pragma once

#include <memory>
namespace denox::memory {

template <typename T> using weak_shared_ptr = std::shared_ptr<T>;

template<typename T>
using weak_ptr = std::weak_ptr<T>;

}

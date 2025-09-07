#pragma once

#include <variant>
namespace denox::memory {

template <typename... Types> using variant = std::variant<Types...>;

}

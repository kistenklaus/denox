#pragma once

#include "denox/cli/parser/lex/tokens/token.hpp"
#include <string>

struct ParseError : std::runtime_error {
  using std::runtime_error::runtime_error;
};



std::string describe_token(const Token &t);

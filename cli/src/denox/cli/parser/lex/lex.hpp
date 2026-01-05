#pragma once

#include <vector>
#include "denox/cli/parser/lex/tokens/token.hpp"

std::vector<Token> lex(int argc, char **argv);

#include "./indent.hpp"

void vkcnn::codegen::indent(unsigned int indentation, std::string &source) {
  for (unsigned int i = 0; i < indentation; ++i) {
    source.push_back(' ');
  }
}


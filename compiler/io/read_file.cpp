#include "io/read_file.hpp"
#include <fstream>

namespace denox::io {

std::string readFile(const std::string &path) {
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + path);
  }

  std::string content;
  file.seekg(0, std::ios::end);
  content.resize(static_cast<size_t>(file.tellg()));
  file.seekg(0, std::ios::beg);
  file.read(&content[0], static_cast<std::streamsize>(content.size()));
  return content;
}

} // namespace denox::io

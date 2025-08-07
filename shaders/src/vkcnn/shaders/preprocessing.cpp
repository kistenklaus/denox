#include "./preprocessing.hpp"
#include <algorithm>
#include <fmt/base.h>
#include <fmt/format.h>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace vkcnn::shaders {

struct Block {
  std::string id;
  std::string src;
};

std::string preprocess_shader_src_pragmas(std::string &src) {

  // Read string by line.
  std::vector<std::string> lines;
  lines.reserve(1000);
  std::istringstream ss(src);
  for (std::string line; std::getline(ss, line);) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    lines.push_back(line);
  }

  // Replace #pragma unroll with [[unroll]]
  // TODO: This should depend on wether GL_EXT_control_flow_attributes is
  //       available.


  std::regex unroll_syntax(R"(^\s*(?!\/)\s*#pragma\s+unroll)");
  for (auto &line : lines) {
    if (std::regex_match(line, unroll_syntax)) {
      line = "#ifdef GL_EXT_control_flow_attributes\n[[unroll]]\n#endif";
    }
    // std::string pragmaUnroll = "#pragma unroll";
    // std::string::size_type pos = line.find(pragmaUnroll);
    // if (pos != std::string::npos) {
    //   line.clear();
    //   for (std::size_t p = 0; p < pos; ++p) {
    //     line.push_back(' ');
    //   }
    //   line.append("#ifdef GL_EXT_control_flow_attributes\n");
    //
    //   for (std::size_t p = 0; p < pos; ++p) {
    //     line.push_back(' ');
    //   }
    //   line.append("[[unroll]]\n");
    //
    //   for (std::size_t p = 0; p < pos; ++p) {
    //     line.push_back(' ');
    //   }
    //   line.append("#endif\n");
    // }
  }

  // Search for #pragma begin_block(ID) and corresponding end_block(ID) and
  // erase them from the shader.

  std::vector<Block> blocks;

  std::regex begin_block_syntax(
      R"(^\s*(?!\/)\s*#pragma\s+begin_block\s*\(\s*([a-zA-Z,_]+)\s*\))");
  std::regex end_block_syntax(R"(^\s*(?!\/)\s*#pragma\s+end_block)");

  std::regex inline_block_syntax(
      R"(^\s*(?!\/)\s*#pragma\s+inline_block\s*\(\s*([a-zA-Z,_]+)\s*\))");

  std::size_t startOfBlock = -1;
  constexpr std::size_t npos = static_cast<std::size_t>(-1);
  std::string blockId;
  for (std::size_t i = 0; i < lines.size(); ++i) {
    if (startOfBlock == npos) {
      std::smatch m;
      if (std::regex_search(lines[i], m, begin_block_syntax)) {
        startOfBlock = i;
        blockId = m[1].str();

      } else if (std::regex_search(lines[i], m, inline_block_syntax)) {
        std::string blockId = m[1];
        auto it = std::ranges::find_if(
            blocks, [&](const Block &b) { return b.id == blockId; });
        if (it == blocks.end()) {
          throw std::runtime_error(fmt::format(
              "#pragma inline_block(...) references invalid block \"{}\".",
              blockId));
        } else {
          lines[i] = it->src;
        }
      }
    } else {
      std::smatch m;
      if (std::regex_search(lines[i], m, end_block_syntax)) {
        std::size_t endOfBlock = i;
        std::string blockSrc;
        for (std::size_t i = startOfBlock + 1; i < endOfBlock; ++i) {
          blockSrc.append(lines[i]);
          blockSrc.push_back('\n');
        }
        blocks.emplace_back(blockId, std::move(blockSrc));
        auto s = lines.begin() + startOfBlock;
        auto e = lines.begin() + endOfBlock + 1;
        lines.erase(s, e);
        i -= std::distance(s, e);
        startOfBlock = npos;
      } else if (std::regex_search(lines[i], m, begin_block_syntax)) {
        throw std::runtime_error("Recursive #pragma blocks are not supported");
      }
    }
  }

  std::string outsrc;
  for (const auto &line : lines) {
    outsrc.append(line);
    outsrc.push_back('\n');
  }

  std::ofstream ofStream;
  ofStream.open("./preprocessed.comp", std::ios::out | std::ios::binary);

  ofStream.write(outsrc.c_str(), outsrc.size());

  return outsrc;
}

} // namespace vkcnn::shaders

#pragma once

#include "merian/vk/context.hpp"
#include "vkcnn/host/shaderlang.hpp"
#include <cassert>
#include <memory>
#include <span>
#include <vector>
namespace vkcnn {

class ShaderSource {
public:
  ShaderSource(ShaderLang lang, std::span<const std::byte> src)
      : m_lang(lang), m_src(std::make_shared<std::vector<std::byte>>(
                          src.begin(), src.end())) {}

  static ShaderSource glsl_from_file(const merian::ContextHandle &context,
                                     std::string_view path) {
    std::string conv2dSrcStr = context->file_loader.load_file(path);

    return ShaderSource(
        ShaderLang::GLSL,
        {reinterpret_cast<const std::byte *>(conv2dSrcStr.begin().base()),
         reinterpret_cast<const std::byte *>(conv2dSrcStr.end().base())});
  }

  static ShaderSource glsl_from_string(std::string_view src) {
    return ShaderSource(ShaderLang::GLSL,
                        {reinterpret_cast<const std::byte *>(src.begin()),
                         reinterpret_cast<const std::byte *>(src.end())});
  }

  template <typename B = std::byte> std::span<const B> src() const {
    assert((m_src->size() % sizeof(B)) == 0);
    const B *begin = reinterpret_cast<const B *>(m_src->cbegin().base());
    const B *end = reinterpret_cast<const B *>(m_src->cend().base());
    return {begin, end};
  }

  ShaderLang lang() const { return m_lang; }

private:
  ShaderLang m_lang;
  std::shared_ptr<std::vector<std::byte>> m_src;
};

} // namespace vkcnn

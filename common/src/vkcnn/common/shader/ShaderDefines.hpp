#pragma once

#include <span>
#include <string>
#include <vector>
namespace vkcnn {

struct ShaderDefine {
  std::string name;
  std::string value;
};

class ShaderDefines {
public:
  ShaderDefines(std::vector<ShaderDefine> defines)
      : m_defs(std::move(defines)) {}

  ShaderDefines(std::span<const ShaderDefine> defines = {})
      : m_defs(defines.begin(), defines.end()) {}

  auto begin() { return m_defs.begin(); }
  auto end() { return m_defs.begin(); }
  auto begin() const { return m_defs.begin(); }
  auto end() const { return m_defs.begin(); }

private:
  std::vector<ShaderDefine> m_defs;
};

} // namespace vkcnn

#pragma once

#include "merian/vk/context.hpp"
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
namespace bench {

struct ConvBenchmark {
  struct Report {};

  struct ConvImpl {
    std::string name;
    glm::uvec3 mmaShape;
    glm::uvec2 kernelSize;
    glm::uvec2 tileSize;
  };

  ConvBenchmark(const merian::ContextHandle &context) : m_context(context) {}

  void addImpl() {}

  void run() {}

  const Report &report() const { return m_report; }

private:
  merian::ContextHandle m_context;
  Report m_report;
};

} // namespace bench

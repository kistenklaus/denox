#include "./entry.hpp"
#include <fmt/format.h>
#include <glm/ext/vector_uint3.hpp>

namespace vkcnn::codegen {

void begin_entry(glm::uvec3 workgroupSize, std::string_view entry,
                 std::string &source) {
  source.append(fmt::format(
      "layout(local_size_x = {}, local_size_y = {}, local_size_z = {}) in;\n",
      workgroupSize.x, workgroupSize.y, workgroupSize.z));
  source.append(fmt::format("void {}(void) {{\n", entry));
}
void end_entry(std::string &source) { source.append("}\n"); }

} // namespace vkcnn::codegen

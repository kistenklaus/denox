#include "parser/lex/tokens/literal.hpp"

std::optional<std::uint64_t> LiteralToken::parse_unsigned() const noexcept {
  auto sv = view();

  if (sv.empty() || sv[0] == '-') {
    return std::nullopt;
  }

  std::uint64_t tmp{};
  auto res = std::from_chars(sv.data(), sv.data() + sv.size(), tmp);
  if (res.ec != std::errc{} || res.ptr != sv.data() + sv.size()) {
    return std::nullopt;
  }

  return tmp;
}

std::optional<std::int64_t> LiteralToken::parse_signed() const noexcept {
  auto sv = view();

  if (sv.empty()) {
    return std::nullopt;
  }

  std::int64_t tmp{};
  auto res = std::from_chars(sv.data(), sv.data() + sv.size(), tmp);
  if (res.ec != std::errc{} || res.ptr != sv.data() + sv.size()) {
    return std::nullopt;
  }

  return tmp;
}

std::optional<denox::VulkanApiVersion>
LiteralToken::parse_target_env() const noexcept {
  std::string_view sv = view();
  // accept common CLI spellings
  if (sv == "vulkan1.0" || sv == "vulkan10") {
    return denox::VulkanApiVersion::Vulkan_1_0;
  }
  if (sv == "vulkan1.1" || sv == "vulkan11") {
    return denox::VulkanApiVersion::Vulkan_1_1;
  }
  if (sv == "vulkan1.2" || sv == "vulkan12") {
    return denox::VulkanApiVersion::Vulkan_1_2;
  }
  if (sv == "vulkan1.3" || sv == "vulkan13") {
    return denox::VulkanApiVersion::Vulkan_1_3;
  }
  if (sv == "vulkan1.4" || sv == "vulkan14") {
    return denox::VulkanApiVersion::Vulkan_1_4;
  }

  return std::nullopt;
}
char LiteralToken::to_lower_ascii(char c) noexcept {
  return (c >= 'A' && c <= 'Z') ? (c + ('a' - 'A')) : c;
}
std::optional<denox::Layout> LiteralToken::parse_layout() const noexcept {
  std::string_view sv = view();
  if (sv.empty()) {
    return std::nullopt;
  }

  char buf[8];
  if (sv.size() >= sizeof(buf)) {
    return std::nullopt;
  }

  for (std::size_t i = 0; i < sv.size(); ++i) {
    buf[i] = to_lower_ascii(sv[i]);
  }

  std::string_view v{buf, sv.size()};

  // tensor layouts
  if (v == "hwc") {
    return denox::Layout::HWC;
  }
  if (v == "chw") {
    return denox::Layout::CHW;
  }
  if (v == "chwc8") {
    return denox::Layout::CHWC8;
  }

  // image / channel layouts
  if (v == "rgba") {
    return denox::Layout::RGBA;
  }
  if (v == "rgb") {
    return denox::Layout::RGB;
  }
  if (v == "rg") {
    return denox::Layout::RG;
  }
  if (v == "r") {
    return denox::Layout::R;
  }

  return std::nullopt;
}

std::optional<denox::DataType> LiteralToken::parse_dtype() const noexcept {
  std::string_view sv = view();
  if (sv.empty()) {
    return std::nullopt;
  }

  char buf[16];
  if (sv.size() >= sizeof(buf)) {
    return std::nullopt;
  }

  for (std::size_t i = 0; i < sv.size(); ++i) {
    buf[i] = to_lower_ascii(sv[i]);
  }

  std::string_view v{buf, sv.size()};

  if (v == "f16" || v == "float16") {
    return denox::DataType::Float16;
  }
  if (v == "f32" || v == "float32") {
    return denox::DataType::Float32;
  }
  if (v == "u8" || v == "uint8") {
    return denox::DataType::Uint8;
  }
  if (v == "i8" || v == "int8") {
    return denox::DataType::Int8;
  }

  return std::nullopt;
}

std::optional<denox::Storage> LiteralToken::parse_storage() const noexcept {
  std::string_view sv = view();
  if (sv.empty()) {
    return std::nullopt;
  }

  char buf[32];
  if (sv.size() >= sizeof(buf)) {
    return std::nullopt;
  }

  for (std::size_t i = 0; i < sv.size(); ++i) {
    buf[i] = to_lower_ascii(sv[i]);
  }

  std::string_view v{buf, sv.size()};

  if (v == "storagebuffer" || v == "storage-buffer" || v == "buffer" ||
      v == "ssbo") {
    return denox::Storage::StorageBuffer;
  }

  if (v == "storageimage" || v == "storage-image" || v == "image") {
    return denox::Storage::StorageImage;
  }

  if (v == "sampler" || v == "sampledimage" || v == "sampled-image" ||
      v == "texture") {
    return denox::Storage::Sampler;
  }

  return std::nullopt;
}

bool LiteralToken::is_path() const noexcept {
  // Reject empty
  if (m_value.empty())
    return false;

  // Reject things that are clearly not paths
  if (is_duration())
    return false;
  if (is_signed_int() || is_unsigned_int())
    return false;

  // Reject NUL bytes and control newlines
  for (char c : m_value) {
    if (c == '\0')
      return false;
    if (c == '\n' || c == '\r')
      return false;
  }

  // Attempt to construct a Path (purely syntactic)
  try {
    Path p{view()};
    return !p.empty();
  } catch (...) {
    return false;
  }
}

Path LiteralToken::as_path() const {
  assert(is_path());
  return Path{view()};
}

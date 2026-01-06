#include "denox/cli/parser/lex/tokens/literal.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/device_info/ApiVersion.hpp"
#include <charconv>
#include <optional>

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

std::optional<denox::ApiVersion>
LiteralToken::parse_target_env() const noexcept {
  std::string_view sv = view();
  // accept common CLI spellings
  if (sv == "vulkan1.0" || sv == "vulkan10") {
    return denox::ApiVersion::VULKAN_1_0;
  }
  if (sv == "vulkan1.1" || sv == "vulkan11") {
    return denox::ApiVersion::VULKAN_1_1;
  }
  if (sv == "vulkan1.2" || sv == "vulkan12") {
    return denox::ApiVersion::VULKAN_1_2;
  }
  if (sv == "vulkan1.3" || sv == "vulkan13") {
    return denox::ApiVersion::VULKAN_1_3;
  }
  if (sv == "vulkan1.4" || sv == "vulkan14") {
    return denox::ApiVersion::VULKAN_1_4;
  }

  return std::nullopt;
}
char LiteralToken::to_lower_ascii(char c) noexcept {
  return (c >= 'A' && c <= 'Z') ? (c + ('a' - 'A')) : c;
}
std::optional<denox::TensorFormat> LiteralToken::parse_format() const noexcept {
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
    return denox::TensorFormat::SSBO_HWC;
  }
  if (v == "chw") {
    return denox::TensorFormat::SSBO_CHW;
  }
  if (v == "chwc8") {
    return denox::TensorFormat::SSBO_CHWC8;
  }

  // image / channel layouts
  if (v == "rgba") {
    return denox::TensorFormat::TEX_RGBA;
  }
  if (v == "rgb") {
    return denox::TensorFormat::TEX_RGB;
  }
  if (v == "rg") {
    return denox::TensorFormat::TEX_RG;
  }
  if (v == "r") {
    return denox::TensorFormat::TEX_R;
  }

  return std::nullopt;
}

std::optional<denox::TensorDataType>
LiteralToken::parse_dtype() const noexcept {
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
    return denox::TensorDataType::Float16;
  }
  if (v == "f32" || v == "float32") {
    return denox::TensorDataType::Float32;
  }
  if (v == "f64" || v == "float64") {
    return denox::TensorDataType::Float64;
  }

  return std::nullopt;
}

std::optional<denox::TensorStorage>
LiteralToken::parse_storage() const noexcept {
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
    return denox::TensorStorage::StorageBuffer;
  }

  if (v == "storageimage" || v == "storage-image" || v == "image") {
    return denox::TensorStorage::StorageImage;
  }

  if (v == "sampledstorageimage" || v == "ssi") {
    return denox::TensorStorage::SampledStorageImage;
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
    denox::io::Path p{view()};
    return !p.empty();
  } catch (...) {
    return false;
  }
}

denox::io::Path LiteralToken::as_path() const {
  assert(is_path());
  return denox::io::Path{view()};
}

std::optional<float> LiteralToken::parse_float() const noexcept {
  float value = 0.0f;
  const char *begin = m_value.data();
  const char *end = begin + m_value.size();
  auto [ptr, ec] =
      std::from_chars(begin, end, value, std::chars_format::general);
  if (ec != std::errc{} || ptr != end)
    return std::nullopt;
  return value;
}

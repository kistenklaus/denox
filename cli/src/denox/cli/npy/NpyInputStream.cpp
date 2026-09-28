#include "denox/cli/npy/NpyInputStream.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint32_t read_le_uint(InputStream &stream, unsigned byte_count) {
  std::byte bytes[4]{};
  stream.read_exact({bytes, byte_count});

  std::uint32_t value = 0;
  for (unsigned i = 0; i < byte_count; ++i) {
    value |= std::uint32_t(std::to_integer<unsigned char>(bytes[i])) << (8 * i);
  }
  return value;
}

std::size_t checked_size(std::size_t h, std::size_t w, std::size_t c) {
  if (h == 0 || w == 0 || c == 0 || h > std::numeric_limits<unsigned>::max() ||
      w > std::numeric_limits<unsigned>::max() ||
      c > std::numeric_limits<unsigned>::max() ||
      h > std::numeric_limits<std::size_t>::max() / w ||
      h * w > std::numeric_limits<std::size_t>::max() / c ||
      h * w * c > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::runtime_error("Invalid or too large NPY shape");
  }
  return h * w * c;
}

} // namespace

denox::memory::optional<denox::memory::ActivationTensor>
NpyInputStream::read_tensor() {
  using namespace denox::memory;

  static_assert(sizeof(f16) == 2);

  std::byte first;
  if (m_stream->read({&first, 1}) == 0) {
    return nullopt;
  }

  std::byte magic[6]{first};
  m_stream->read_exact({magic + 1, 5});

  constexpr std::byte expected[6]{
      std::byte{0x93}, std::byte{'N'}, std::byte{'U'},
      std::byte{'M'},  std::byte{'P'}, std::byte{'Y'},
  };
  if (std::memcmp(magic, expected, sizeof(magic)) != 0) {
    throw std::runtime_error("Invalid NPY signature");
  }

  std::byte version[2];
  m_stream->read_exact(version);

  const unsigned major = std::to_integer<unsigned>(version[0]);
  if (major < 1 || major > 3) {
    throw std::runtime_error("Unsupported NPY version");
  }

  const std::uint32_t header_size = read_le_uint(*m_stream, major == 1 ? 2 : 4);
  if (header_size > 1'000'000) {
    throw std::runtime_error("NPY header is too large");
  }

  std::string header(header_size, '\0');
  m_stream->read_exact(
      {reinterpret_cast<std::byte *>(header.data()), header.size()});

  // Denox convention: a C-contiguous array with shape (C, H, W).
  const bool file_f16 = std::regex_search(
      header, std::regex(R"(['"]descr['"]\s*:\s*['"]<f2['"])"));
  const bool file_f32 = std::regex_search(
      header, std::regex(R"(['"]descr['"]\s*:\s*['"]<f4['"])"));
  const bool c_order = std::regex_search(
      header, std::regex(R"(['"]fortran_order['"]\s*:\s*False\b)"));

  if ((!file_f16 && !file_f32) || !c_order) {
    throw std::runtime_error(
        "Expected C-order little-endian float16 or float32 NPY");
  }
  if (std::endian::native != std::endian::little) {
    throw std::runtime_error("NPY input requires a little-endian host");
  }

  std::smatch shape;
  if (!std::regex_search(
          header, shape,
          std::regex(
              R"(['"]shape['"]\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,?\s*\))"))) {
    throw std::runtime_error("Expected a three-dimensional CHW array");
  }

  const std::size_t channels = std::stoull(shape[1].str());
  const std::size_t height = std::stoull(shape[2].str());
  const std::size_t width = std::stoull(shape[3].str());
  checked_size(height, width, channels);

  const ActivationDescriptor desc{
      .shape = {static_cast<unsigned>(width), static_cast<unsigned>(height),
                static_cast<unsigned>(channels)},
      .layout = ActivationLayout::CHW,
      .type = file_f16 ? Dtype::F16 : Dtype::F32,
  };

  ActivationTensor tensor{desc};
  m_stream->read_exact(tensor.span());
  return tensor;
}

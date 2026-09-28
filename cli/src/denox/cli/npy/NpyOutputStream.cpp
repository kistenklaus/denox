#include "denox/cli/npy/NpyOutputStream.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

void NpyOutputStream::write_tensor(
    const denox::memory::ActivationTensor& tensor) {
  using namespace denox::memory;

  if constexpr (std::endian::native != std::endian::little) {
    throw std::runtime_error("NPY output requires a little-endian host");
  }

  const auto width = tensor.shape().w;
  const auto height = tensor.shape().h;
  const auto channels = tensor.shape().c;

  if (width == 0 || height == 0 || channels == 0) {
    throw std::runtime_error("Cannot write an empty tensor");
  }

  const ActivationDescriptor desc{
      .shape = {width, height, channels},
      .layout = ActivationLayout::CHW,
      .type = Dtype::F32,
  };
  const ActivationTensor chw{desc, tensor};

  std::string header =
      "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
      std::to_string(channels) + ", " +
      std::to_string(height) + ", " +
      std::to_string(width) + "), }";

  // NPY v1.0: 6-byte magic, 2-byte version, 2-byte header length.
  // Pad the header so that the data begins at a 64-byte boundary.
  constexpr std::size_t preamble_size = 10;
  const std::size_t padding =
      (64 - ((preamble_size + header.size() + 1) % 64)) % 64;
  header.append(padding, ' ');
  header.push_back('\n');

  if (header.size() > UINT16_MAX) {
    throw std::runtime_error("NPY header exceeds version 1 limit");
  }

  const auto length = static_cast<std::uint16_t>(header.size());
  const std::byte preamble[]{
      std::byte{0x93}, std::byte{'N'}, std::byte{'U'},
      std::byte{'M'},  std::byte{'P'}, std::byte{'Y'},
      std::byte{1},    std::byte{0},
      std::byte{static_cast<unsigned char>(length & 0xff)},
      std::byte{static_cast<unsigned char>(length >> 8)},
  };

  m_stream->write_exact(preamble);
  m_stream->write_exact({
      reinterpret_cast<const std::byte*>(header.data()), header.size()});
  m_stream->write_exact(chw.span());
}

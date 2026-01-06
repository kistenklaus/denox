#include "denox/cli/png/PngInputStream.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/vector.hpp"

#include <png.h>
#include <pngconf.h>
#include <stdexcept>

denox::memory::optional<denox::memory::ActivationTensor>
PngInputStream::read_image() {
  std::byte sig[8];
  try {
    m_stream->read_exact(sig);
  } catch (...) {
    return denox::memory::nullopt;
  }

  if (png_sig_cmp(reinterpret_cast<png_const_bytep>(sig), 0, 8)) {
    return denox::memory::nullopt;
  }
  png_structp png =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png) {
    throw std::runtime_error("Failed to create png read struct");
  }
  png_infop info = png_create_info_struct(png);
  if (!info) {
    png_destroy_read_struct(&png, nullptr, nullptr);
    throw std::runtime_error("Failed to create png info struct");
  }
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_read_struct(&png, &info, nullptr);
    throw std::runtime_error("libpng decode error");
  }

  png_set_read_fn(
      png, m_stream, [](png_structp png_ptr, png_bytep out, png_size_t count) {
        auto *stream = static_cast<InputStream *>(png_get_io_ptr(png_ptr));
        stream->read_exact({reinterpret_cast<std::byte *>(out), count});
      });
  png_set_sig_bytes(png, 8);

  png_read_info(png, info);

  const png_uint_32 width = png_get_image_width(png, info);
  const png_uint_32 height = png_get_image_height(png, info);
  const int color_type = png_get_color_type(png, info);
  const int bit_depth = png_get_bit_depth(png, info);

  if (bit_depth == 16) {
    png_set_strip_16(png);
  }
  if (color_type == PNG_COLOR_TYPE_PALETTE) {
    png_set_palette_to_rgb(png);
  }
  if (color_type == PNG_COLOR_TYPE_GRAY ||
      color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
    png_set_gray_to_rgb(png);
  }

  if (png_get_valid(png, info, PNG_INFO_tRNS)) {
    png_set_tRNS_to_alpha(png);
  }

  png_set_strip_alpha(png);

  png_read_update_info(png, info);

  const png_uint_32 channels = png_get_channels(png, info);
  if (channels != 3) {
    png_destroy_read_struct(&png, &info, nullptr);
    throw std::runtime_error("Expected RGB PNG after normalization");
  }

  denox::memory::ActivationDescriptor desc{
      .shape = {width, height, channels},
      .layout = denox::memory::ActivationLayout::HWC,
      .type = denox::memory::Dtype::F32,
  };
  denox::memory::ActivationTensor tensor{desc};

  denox::memory::vector<png_byte> row_u8(width * channels);
  float *dst = reinterpret_cast<float *>(tensor.data());

  constexpr float inv255 = 1.0f / 255.0f;

  for (png_uint_32 y = 0; y < height; ++y) {
    png_read_row(png, row_u8.data(), nullptr);
    float *row_dst = dst + y * width * channels;
    for (png_uint_32 i = 0; i < width * channels; ++i) {
      row_dst[i] = row_u8[i] * inv255;
    }
  }
  png_read_end(png, nullptr);
  png_destroy_read_struct(&png, &info, nullptr);
  return tensor;
}

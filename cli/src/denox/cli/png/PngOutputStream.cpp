#include "denox/cli/png/PngOutputStream.hpp"

#include <algorithm>
#include <png.h>
#include <stdexcept>
#include <vector>

void PngOutputStream::write_image(
    const denox::memory::ActivationTensor &image) {

  // ---- 1. Normalize input tensor to HWC, F32 -----------------------------
  denox::memory::ActivationDescriptor desc{
      .shape = {image.shape().w, image.shape().h, image.shape().c},
      .layout = denox::memory::ActivationLayout::HWC,
      .type = denox::memory::Dtype::F32,
  };

  denox::memory::ActivationTensor tensor{desc, image};

  const uint32_t width = tensor.shape().w;
  const uint32_t height = tensor.shape().h;
  const uint32_t channels = tensor.shape().c;

  if (channels < 1 || channels > 4) {
    throw std::runtime_error("PNG output supports 1–4 channels only");
  }

  const bool write_alpha = (channels == 4);

  // ---- 2. Create libpng write structs ------------------------------------
  png_structp png =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png)
    throw std::runtime_error("png_create_write_struct failed");

  png_infop info = png_create_info_struct(png);
  if (!info) {
    png_destroy_write_struct(&png, nullptr);
    throw std::runtime_error("png_create_info_struct failed");
  }

  if (setjmp(png_jmpbuf(png))) {
    png_destroy_write_struct(&png, &info);
    throw std::runtime_error("libpng write error");
  }

  // ---- 3. Install custom write callback ----------------------------------
  png_set_write_fn(
      png, m_stream,
      [](png_structp png_ptr, png_bytep data, png_size_t length) {
        auto *stream = static_cast<OutputStream *>(png_get_io_ptr(png_ptr));
        stream->write_exact(
            {reinterpret_cast<const std::byte *>(data), length});
      },
      [](png_structp png_ptr) {
        auto *stream = static_cast<OutputStream *>(png_get_io_ptr(png_ptr));
        stream->flush();
      });

  // ---- 4. Configure PNG header -------------------------------------------
  const int color_type = write_alpha ? PNG_COLOR_TYPE_RGBA : PNG_COLOR_TYPE_RGB;

  png_set_IHDR(png, info, width, height,
               8, // bit depth
               color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png, info);

  // ---- 5. Encode rows -----------------------------------------------------
  std::vector<png_byte> row_u8(width * (write_alpha ? 4 : 3));
  const float *src = reinterpret_cast<const float *>(tensor.data());

  for (uint32_t y = 0; y < height; ++y) {
    const float *row_src = src + y * width * channels;
    png_byte *row = row_u8.data();

    for (uint32_t x = 0; x < width; ++x) {
      const float *px = row_src + x * channels;

      const auto to_u8 = [](float v) -> png_byte {
        v = std::clamp(v, 0.0f, 1.0f);
        return static_cast<png_byte>(v * 255.0f + 0.5f);
      };

      // R
      row[0] = to_u8(px[0]);

      // G
      row[1] = (channels >= 2) ? to_u8(px[1]) : 0;

      // B
      row[2] = (channels >= 3) ? to_u8(px[2]) : 0;

      if (write_alpha) {
        row[3] = to_u8(px[3]);
        row += 4;
      } else {
        row += 3;
      }
    }

    png_write_row(png, row_u8.data());
  }

  // ---- 6. Finalize --------------------------------------------------------
  png_write_end(png, nullptr);
  png_destroy_write_struct(&png, &info);
}

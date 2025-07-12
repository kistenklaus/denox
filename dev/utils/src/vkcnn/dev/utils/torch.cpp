#include "./torch.hpp"

#include "ATen/core/ATen_fwd.h"
#include "c10/core/DeviceType.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/Exception.h"
#include "torch/cuda.h"
#include "torch/types.h"
#include "vkcnn/common/tensor/ActivationDescriptor.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
#include <stdexcept>

::torch::Tensor
vkcnn::torch::fromActivation(ActivationHostTensorConstView activation) {
  unsigned int W = activation.shape().w;
  unsigned int H = activation.shape().h;
  unsigned int C = activation.shape().c;

  ::torch::Dtype dtype;
  if (activation.type() == FloatType::F16) {
    dtype = ::torch::Dtype::Half;
  } else if (activation.type() == FloatType::F32) {
    dtype = ::torch::Dtype::Float;
  } else if (activation.type() == FloatType::F64) {
    dtype = ::torch::Dtype::Double;
  } else {
    throw std::runtime_error("Unsupported FloatType");
  }

  ::torch::IntArrayRef sizes;
  ::torch::IntArrayRef strides;
  int64_t tmp_sz[4];
  int64_t tmp_st[4];

  if (activation.layout() == ActivationLayout::CHW) {
    tmp_sz[0] = C;
    tmp_sz[1] = H;
    tmp_sz[2] = W;

    tmp_st[0] = H * W;
    tmp_st[1] = W;
    tmp_st[2] = 1;

    sizes = {tmp_sz, 3};
    strides = {tmp_st, 3};
  } else if (activation.layout() == ActivationLayout::HWC) {
    tmp_sz[0] = H;
    tmp_sz[1] = W;
    tmp_sz[2] = C;

    tmp_st[0] = W * C;
    tmp_st[1] = C;
    tmp_st[2] = 1;

    sizes = {tmp_sz, 3};
    strides = {tmp_st, 3};
  } else if (activation.layout() == ActivationLayout::CHWC8) {
    const unsigned int blk = 8;
    TORCH_CHECK(C % blk == 0, "C must be a multiple of 8");
    tmp_sz[0] = C / blk;
    tmp_sz[1] = H;
    tmp_sz[2] = W;
    tmp_sz[3] = blk;

    tmp_st[0] = H * W * blk;
    tmp_st[1] = W * blk;
    tmp_st[2] = blk;
    tmp_st[3] = 1;

    sizes = {tmp_sz, 4};
    strides = {tmp_st, 4};
  } else if (activation.layout() == ActivationLayout::CHWC16) {
    const unsigned int blk = 16;
    TORCH_CHECK(C % blk == 0, "C must be a multiple of 16");
    tmp_sz[0] = C / blk;
    tmp_sz[1] = H;
    tmp_sz[2] = W;
    tmp_sz[3] = blk;

    tmp_st[0] = H * W * blk;
    tmp_st[1] = W * blk;
    tmp_st[2] = blk;
    tmp_st[3] = 1;

    sizes = {tmp_sz, 4};
    strides = {tmp_st, 4};
  } else {
    throw std::runtime_error("Unsupported layout");
  }
  auto opts = ::torch::TensorOptions().dtype(dtype).device(::torch::kCPU);

  auto no_del = [](void *) {};
  ::torch::Tensor t = ::torch::from_blob(
      const_cast<std::byte *>(activation.data()), sizes, strides, no_del, opts);
  t = t.clone();

  if (activation.layout() == ActivationLayout::HWC) {
    t = t.permute({2, 0, 1}); // HWC -> CHW
  } else if (activation.layout() == ActivationLayout::CHWC8 ||
             activation.layout() == ActivationLayout::CHWC16) {
    t = t.permute({0, 3, 1, 2}).reshape({C, H, W}); // CHWCX -> CHW
  }
  t = t.unsqueeze(0); // add batch dim.
  t = t.contiguous();
  if (::torch::cuda::is_available()) {
    t = t.to(::torch::kCUDA, true);
  }
  return t;
}

vkcnn::ActivationHostTensor vkcnn::torch::toActivation(::torch::Tensor tensor,
                                                       ActivationLayout layout,
                                                       FloatType type) {
  // --------- 0. device & batch handling ------------------------------------

  TORCH_CHECK(tensor.dim() == 3 || tensor.dim() == 4,
              "expected 3-D (C,H,W) or 4-D (N,C,H,W) tensor");

  if (tensor.dim() == 4) {
    TORCH_CHECK(tensor.size(0) == 1,
                "batch > 1 not supported by ActivationHostTensor");
    tensor = tensor.squeeze(0); // (1,C,H,W) → (C,H,W)
  }

  // From here on tensor is 3-D.
  const int64_t C_in = tensor.size(0);
  const int64_t H = tensor.size(1);
  const int64_t W = tensor.size(2);

  // --------- 1. bring tensor into the requested memory order ---------------
  if (layout == ActivationLayout::CHW) {
    if (!tensor.is_contiguous())
      tensor = tensor.contiguous();
  } else if (layout == ActivationLayout::HWC) {
    if (!(tensor.is_contiguous(::torch::MemoryFormat::ChannelsLast) &&
          tensor.dim() == 3)) {
      tensor = tensor.permute({1, 2, 0}); // H,W,C
    }
    tensor = tensor.contiguous();
  } else if (layout == ActivationLayout::CHWC8 ||
             layout == ActivationLayout::CHWC16) {
    const int blk = (layout == ActivationLayout::CHWC8) ? 8 : 16;
    // 1. Reorder to HWC so we can pack easily: (H,W,C)
    if (!(tensor.is_contiguous(::torch::MemoryFormat::ChannelsLast) &&
          tensor.dim() == 3))
      tensor = tensor.permute({1, 2, 0}); // H,W,C

    // 2. Pad channel dim up to multiple of blk
    const int64_t C_pad = (C_in + blk - 1) / blk * blk;
    if (C_pad != C_in) {
      auto padSz = tensor.sizes().vec();
      padSz.back() = C_pad - C_in; // extra channels
      auto pad = ::torch::zeros(padSz, tensor.options());
      tensor = ::torch::cat({tensor, pad}, -1);
    }

    // 3. Reshape to (H,W,C/blk,blk) then permute to (C/blk,H,W,blk)
    tensor = tensor.view({H, W, C_pad / blk, blk})
                 .permute({2, 0, 1, 3}) // Cb, H, W, blk
                 .contiguous();
  } else {
    TORCH_CHECK(false, "Unsupported ActivationLayout");
  }

  tensor.to(fromType(type));

  if (!tensor.device().is_cpu())
    tensor = tensor.cpu();

  // --------- 3. dtype -------------------------------------------------------

  ActivationDescriptor desc{
      .shape = {static_cast<unsigned int>(W), static_cast<unsigned int>(H),
                static_cast<unsigned int>(C_in)},
      .layout = layout,
      .type = type,
  };

  const std::size_t nBytes = desc.byteSize();
  return ActivationHostTensor{
      desc, std::span(reinterpret_cast<const std::byte *>(tensor.data_ptr()),
                      nBytes)};
}

//

::torch::Tensor vkcnn::torch::fromFilter(FilterHostTensorConstView filter) {
  const auto [R, S, C, K] = filter.shape(); // (r, s, c, k)

  // ---------- dtype -----------------------------------------------------
  ::torch::Dtype dtype;
  if (filter.type() == FloatType::F16) {
    dtype = ::torch::Dtype::Half;
  } else if (filter.type() == FloatType::F32) {
    dtype = ::torch::Dtype::Float;
  } else if (filter.type() == FloatType::F64) {
    dtype = ::torch::Dtype::Double;
  } else {
    throw std::runtime_error("Unsupported FloatType");
  }

  // ---------- sizes + strides tables ------------------------------------
  std::array<int64_t, 5> sz{}; // longest case = 5-D blocked layout
  std::array<int64_t, 5> st{};
  ::torch::IntArrayRef sizes;
  ::torch::IntArrayRef strides;

  const FilterLayout layout = filter.layout();

  if (layout == FilterLayout::KCRS) { // (K,C,R,S)
    sz = {K, C, R, S};
    st = {C * R * S, R * S, S, 1};
    sizes = {sz.data(), 4};
    strides = {st.data(), 4};
  } else if (layout == FilterLayout::KRSC) { // (K,R,S,C)
    sz = {K, R, S, C};
    st = {R * S * C, S * C, C, 1};
    sizes = {sz.data(), 4};
    strides = {st.data(), 4};
  } else if (layout == FilterLayout::RSCK) { // (R,S,C,K)
    sz = {R, S, C, K};
    st = {S * C * K, C * K, K, 1};
    sizes = {sz.data(), 4};
    strides = {st.data(), 4};
  } else if (layout == FilterLayout::RSKC) { // (R,S,K,C)
    sz = {R, S, K, C};
    st = {S * K * C, K * C, C, 1};
    sizes = {sz.data(), 4};
    strides = {st.data(), 4};
  } else if (layout == FilterLayout::RSCKC8 ||
             layout == FilterLayout::RSCKC16) {
    const int blk = (layout == FilterLayout::RSCKC16) ? 16 : 8;

    const int64_t C_pad = ((C + blk - 1) / blk) * blk; // <-- use padded C!

    sz = {R, S, C_pad / blk, K, blk};
    st = {S * C_pad * K, // stride R
          C_pad * K,     // stride S
          K * blk,       // stride C_block
          blk,           // stride K
          1};            // stride C_inner
    sizes = {sz.data(), 5};
    strides = {st.data(), 5};
  } else if (layout == FilterLayout::RCSKC8 ||
             layout == FilterLayout::RCSKC16) {
    const int blk = (layout == FilterLayout::RCSKC16) ? 16 : 8;
    const int64_t C_pad = ((C + blk - 1) / blk) * blk;

    sz = {R, C_pad / blk, S, K, blk};
    st = {C_pad * S * K, // stride R
          S * K * blk,   // stride C_block
          K * blk,       // stride S
          blk,           // stride K
          1};            // stride C_inner
    sizes = {sz.data(), 5};
    strides = {st.data(), 5};
  } else {
    throw std::runtime_error("unhandled filter layout");
  }

  // ---------- build view & clone ----------------------------------------
  auto opts = ::torch::TensorOptions().dtype(dtype).device(::torch::kCPU);

  static auto no_del = [](void *) {};
  auto t = ::torch::from_blob(
      const_cast<void *>(static_cast<const void *>(filter.data())), sizes,
      strides, no_del, opts);
  t = t.clone();

  if (layout == FilterLayout::KRSC) {
    t = t.permute({0, 3, 1, 2});
  } else if (layout == FilterLayout::RSCK) {
    t = t.permute({3, 2, 0, 1});
  } else if (layout == FilterLayout::RSKC) {
    t = t.permute({2, 3, 0, 1});
  } else if (layout == FilterLayout::RSCKC8 ||
             layout == FilterLayout::RSCKC16) {
    t = t.permute({3, 2, 0, 1, 4}).reshape({K, C, R, S});
  } else if (layout == FilterLayout::RCSKC8 ||
             layout == FilterLayout::RCSKC16) {
    t = t.permute({3, 1, 0, 2, 4}).reshape({K, C, R, S});
  }

  t = t.contiguous();

  if (::torch::cuda::is_available()) {
    t = t.to(::torch::kCUDA, true);
  }
  return t;
}

vkcnn::FilterHostTensor vkcnn::torch::toFilter(::torch::Tensor tensor,
                                               FilterLayout layout,
                                               FloatType type) {
  TORCH_CHECK(tensor.dim() == 4,
              "Expected 4-D weight tensor of shape (K,C,R,S)");

  const int64_t K = tensor.size(0);
  const int64_t C = tensor.size(1);
  const int64_t R = tensor.size(2);
  const int64_t S = tensor.size(3);

  auto dtype = fromType(type);
  if (tensor.dtype() != dtype)
    tensor = tensor.to(dtype); // convert on device

  if (layout == FilterLayout::KCRS) {
    if (!tensor.is_contiguous())
      tensor = tensor.contiguous();

  } else if (layout == FilterLayout::KRSC) {
    tensor = tensor.permute({0, 2, 3, 1}).contiguous();

  } else if (layout == FilterLayout::RSCK) {
    tensor = tensor.permute({2, 3, 1, 0}).contiguous();

  } else if (layout == FilterLayout::RSKC) {
    tensor = tensor.permute({2, 3, 0, 1}).contiguous();

  } else if (layout == FilterLayout::RSCKC8 ||
             layout == FilterLayout::RSCKC16) {

    int blk = layout == FilterLayout::RSCKC8 ? 8 : 16;
    int64_t C_pad = ((C + blk - 1) / blk) * blk;

    tensor = tensor.permute({2, 3, 1, 0}); // R, S, C, K

    if (C_pad != C) {
      auto pad = ::torch::zeros({R, S, C_pad - C, K}, tensor.options());
      tensor = ::torch::cat({tensor, pad}, 2); // pad C
    }

    tensor = tensor.view({R, S, C_pad / blk, blk, K})
                 .permute({0, 1, 2, 4, 3}) // R,S,Cb,K,Ci
                 .contiguous();

  } else if (layout == FilterLayout::RCSKC8 ||
             layout == FilterLayout::RCSKC16) {
    int blk = layout == FilterLayout::RCSKC8 ? 8 : 16;
    int64_t C_pad = ((C + blk - 1) / blk) * blk;

    tensor = tensor.permute({2, 1, 3, 0}); // R, C, S, K

    if (C_pad != C) {
      auto pad = ::torch::zeros({R, C_pad - C, S, K}, tensor.options());
      tensor = ::torch::cat({tensor, pad}, 1); // pad C
    }

    tensor = tensor.view({R, C_pad / blk, blk, S, K})
                 .permute({0, 1, 3, 4, 2}) // R,Cb,S,K,Ci
                 .contiguous();

  } else {
    TORCH_CHECK(false, "Unsupported FilterLayout");
  }

  if (!tensor.device().is_cpu())
    tensor = tensor.to(::torch::kCPU);

  tensor = tensor.contiguous(); // make sure it's dense

  FilterDescriptor desc{
      .shape = {static_cast<unsigned>(R), static_cast<unsigned>(S),
                static_cast<unsigned>(C), static_cast<unsigned>(K)},
      .layout = layout,
      .type = type,
  };

  const std::size_t nBytes = desc.byteSize();
  return FilterHostTensor{
      desc, std::span(reinterpret_cast<const std::byte *>(tensor.data_ptr()),
                      nBytes)};
}

::torch::Dtype vkcnn::torch::fromType(FloatType type) {
  if (type == FloatType::F16) {
    return ::torch::Dtype::Half;
  } else if (type == FloatType::F32) {
    return ::torch::Dtype::Float;
  } else if (type == FloatType::F64) {
    return ::torch::Dtype::Double;
  } else {
    throw std::runtime_error("Invalid float type");
  }
}
vkcnn::FloatType vkcnn::torch::toType(::torch::Dtype dtype) {
  if (dtype == ::torch::Dtype::Half) {
    return FloatType::F16;
  } else if (dtype == ::torch::Dtype::Float) {
    return FloatType::F32;
  } else if (dtype == ::torch::Dtype::Double) {
    return FloatType::F64;
  } else {
    throw std::runtime_error("Unsupported dtype");
  }
}

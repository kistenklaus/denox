#include "./torch.hpp"

#include "ATen/core/ATen_fwd.h"
#include "c10/core/DeviceType.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/Exception.h"
#include "torch/cuda.h"
#include "torch/types.h"
#include "vkcnn/common/tensor/ActivationDescriptor.hpp"
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









vkcnn::ActivationHostTensor vkcnn::torch::toActivation(::torch::Tensor tensor, ActivationLayout layout) {
  if (!tensor.device().is_cpu()) {
    tensor = tensor.cpu();
  }
  TORCH_CHECK(tensor.dim() == 3 || tensor.dim() == 4,
              "expected 3-D (C,H,W) or 4-D (N,C,H,W) tensor");

  if (tensor.dim() == 4) {
    TORCH_CHECK(tensor.size(0) == 1,
                "batch > 1 not supported by ActivationHostTensor");
    tensor = tensor.squeeze(0);
  }

  if (tensor.is_contiguous(::torch::MemoryFormat::ChannelsLast)) {
    // 4-D (N,H,W,C)  or  3-D (H,W,C)
    if (tensor.dim() == 4) {
      tensor = tensor.permute({0, 3, 1, 2}); // NHWC → NCHW
      tensor = tensor.squeeze(0);            // drop N = 1
    } else {                                 // (H,W,C)
      tensor = tensor.permute({2, 0, 1});    // HWC  → CHW
    }
    tensor = tensor.contiguous();       // real copy, row-major
  } else if (!tensor.is_contiguous()) { // any other exotic strides
    tensor = tensor.contiguous();       // simple row-major copy
  }

  tensor = tensor.contiguous();
  tensor = tensor.clone();

  FloatType dtype = FloatType::F16;

  if (tensor.dtype() == ::torch::Dtype::Half) {
    dtype = FloatType::F16;
  } else if (tensor.dtype() == ::torch::Dtype::Float) {
    dtype = FloatType::F32;
  } else if (tensor.dtype() == ::torch::Dtype::Double) {
    dtype = FloatType::F64;
  } else {
    throw std::runtime_error("Unsupported type");
  }

  // ---------- fill the descriptor -----------------------------------
  const int64_t C = tensor.size(0);
  const int64_t H = tensor.size(1);
  const int64_t W = tensor.size(2);
  ActivationDescriptor desc{
      .shape = {static_cast<unsigned int>(W), static_cast<unsigned int>(H),
                static_cast<unsigned int>(C)},
      .layout = ActivationLayout::CHW,
      .type = dtype,
  };
  const std::size_t nBytes = desc.byteSize();
  return ActivationHostTensor{
      desc, std::span(reinterpret_cast<const std::byte *>(tensor.data_ptr()),
                      nBytes)};
}
























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
    TORCH_CHECK(C % blk == 0, "C must be multiple of block size");

    sz = {R, S, C / blk, K, blk};
    st = {S * C * K, // stride R
          C * K,     // stride S
          K * blk,   // stride C/blk
          blk,       // stride K
          1};        // stride blk

    sizes = {sz.data(), 5};
    strides = {st.data(), 5};
  } else if (layout == FilterLayout::RCSKC8 ||
             layout == FilterLayout::RCSKC16) {
    const int blk = (layout == FilterLayout::RCSKC16) ? 16 : 8;
    TORCH_CHECK(C % blk == 0, "C must be multiple of block size");

    sz = {R, C / blk, S, K, blk};
    st = {C * S * K,   // stride R
          S * K * blk, // stride C/blk
          K * blk,     // stride S
          blk,         // stride K
          1};          // stride blk

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

vkcnn::FilterHostTensor vkcnn::torch::toFilter(::torch::Tensor tensor) {

  //---------------------------------------------------------------------
  // 1) Normalise device and shape
  //---------------------------------------------------------------------
  if (!tensor.device().is_cpu())
    tensor = tensor.cpu();

  TORCH_CHECK(tensor.dim() == 4,
              "Expected 4-D weight tensor (K,C,R,S or permutation thereof)");

  int64_t K = tensor.size(0);
  int64_t C = tensor.size(1);
  int64_t R = tensor.size(2);
  int64_t S = tensor.size(3);

  //---------------------------------------------------------------------
  // 2) Convert to row-major **K,C,R,S** (one copy at most)
  //---------------------------------------------------------------------
  if (!tensor.is_contiguous()) {
    // Detect a few common permutations quickly; fall back to .contiguous()
    //  (K,R,S,C)  →  (K,C,R,S)
    if (tensor.stride(1) == 1) { // cheap heuristic for KRSC
      tensor = tensor.permute({0, 3, 1, 2}).contiguous();
    }
    //  (R,S,C,K) or (R,S,K,C)  →  (K,C,R,S)
    else if (tensor.stride(0) == tensor.stride(1) * tensor.size(1)) {
      // RSCK
      tensor = tensor.permute({3, 2, 0, 1}).contiguous();
    } else {
      // generic, handles ChannelsLast3D etc.
      tensor = tensor.contiguous();
    }
  }

  //---------------------------------------------------------------------
  // 3) Fill descriptor (KCRS layout)
  //---------------------------------------------------------------------
  FloatType type = FloatType::F16;
  if (tensor.dtype() == ::torch::Dtype::Half) {
    type = FloatType::F16;
  } else if (tensor.dtype() == ::torch::Dtype::Float) {
    type = FloatType::F32;
  } else if (tensor.dtype() == ::torch::Dtype::Double) {
    type = FloatType::F64;
  } else {
    throw std::runtime_error("Unsupported torch::Dtype");
  }

  FilterDescriptor desc{
      .shape = {static_cast<unsigned>(R), static_cast<unsigned>(S),
                static_cast<unsigned>(C), static_cast<unsigned>(K)},
      .layout = FilterLayout::KCRS, // <-- change here if needed
      .type = type,
  };

  //---------------------------------------------------------------------
  // 4) Copy data into engine buffer
  //---------------------------------------------------------------------
  const std::size_t nBytes = desc.byteSize();
  std::vector<std::byte> buffer(nBytes);

  std::memcpy(buffer.data(), tensor.data_ptr(), nBytes);

  return FilterHostTensor(
      desc, std::span(reinterpret_cast<const std::byte *>(tensor.data_ptr()),
                      nBytes));
}

#include "frontend/onnx/details/ops/ops.hpp"

#include <fmt/format.h>
#include <stdexcept>

namespace denox::onnx::details::ops {

memory::vector<Tensor> pow(
    [[maybe_unused]] ImportState &state,
    memory::span<const memory::optional<Tensor>> inputs, std::size_t outputCount,
    [[maybe_unused]] const memory::hash_map<memory::string, Attribute>
        &attributes,
    [[maybe_unused]] opset_version version, memory::string_view nodeName) {
  // ---- arity ----
  if (inputs.size() != 2 || !inputs[0].has_value() || !inputs[1].has_value())
    throw std::runtime_error(fmt::format(
        "vkcnn: Pow \"{}\" expects exactly 2 inputs.", nodeName));
  if (outputCount != 1)
    throw std::runtime_error(fmt::format(
        "vkcnn: Pow \"{}\" must have exactly 1 output.", nodeName));

  const Tensor &A = *inputs[0];
  const Tensor &B = *inputs[1];

  // Host-only for now
  if (!A.isHost() || !B.isHost())
    throw std::runtime_error(
        fmt::format("vkcnn: Pow \"{}\": only HostTensor inputs are supported.",
                    nodeName));

  const HostTensor &Ah = A.host();
  const HostTensor &Bh = B.host();

  // Require constant shape/view so we can index deterministically
  if (!Ah.isConstant() || !Bh.isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Pow \"{}\": inputs must have constant shapes.", nodeName));
  if (!Ah.view().isConstant() || !Bh.view().isConstant())
    throw std::runtime_error(fmt::format(
        "vkcnn: Pow \"{}\": inputs must have constant views.", nodeName));

  // Dtype checks (float only). Allow F32/F64; upcast to F64 if mixed.
  const Dtype adt = Ah.type();
  const Dtype bdt = Bh.type();
  const bool aF32 = (adt == Dtype::Float32);
  const bool aF64 = (adt == Dtype::Float64);
  const bool bF32 = (bdt == Dtype::Float32);
  const bool bF64 = (bdt == Dtype::Float64);
  if ((!aF32 && !aF64) || (!bF32 && !bF64)) {
    throw std::runtime_error(
        fmt::format("vkcnn: Pow \"{}\": unsupported dtypes (A={}, B={}). "
                    "Supported: Float32, Float64.",
                    nodeName, adt.to_string(), bdt.to_string()));
  }
  const bool outIsF64 = (aF64 || bF64);

  // ---- broadcast shape ----
  const TensorShape Ashape = Ah.shape();
  const TensorShape Bshape = Bh.shape();
  TensorShape Oshape = TensorShape::broadcast(Ashape, Bshape);
  const auto O = Oshape.toU64();
  std::size_t N = 1;
  for (auto d : O)
    N *= static_cast<std::size_t>(d);

  // Precompute dims
  const auto Adims = Ashape.toU64();
  const auto Bdims = Bshape.toU64();
  const size_t rO = O.size();
  const size_t rA = Adims.size();
  const size_t rB = Bdims.size();

  // Allocate output storage
  auto make_out =
      [&](Dtype dt,
          std::size_t elemSize) -> std::shared_ptr<HostTensorStorage> {
    void *raw = std::malloc(N * elemSize);
    if (!raw)
      throw std::bad_alloc();
    return std::make_shared<HostTensorStorage>(
        HostTensorStorage::TakeOwnership(dt, raw, N * elemSize));
  };

  // Index buffers
  memory::vector<std::uint64_t> oIdx(rO, 0);
  memory::vector<std::uint64_t> aIdx(rA, 0);
  memory::vector<std::uint64_t> bIdx(rB, 0);

  auto step_o = [&]() -> bool {
    if (oIdx.empty())
      return false;
    for (size_t ax = rO; ax-- > 0;) {
      if (++oIdx[ax] < O[ax])
        return true;
      oIdx[ax] = 0;
    }
    return false;
  };

  // Helper: map output index -> A/B index with broadcasting
  auto map_to_a = [&]() {
    // align from the back
    for (size_t k = 0; k < rA; ++k) {
      size_t oa = rO - 1 - k;
      size_t ia = rA - 1 - k;
      if (k >= rO) {
        // leading axes (when A has more rank than O) — should not happen after
        // broadcast
        aIdx[ia] = 0;
      } else {
        aIdx[ia] = (Adims[ia] == 1) ? 0 : oIdx[oa];
      }
    }
  };
  auto map_to_b = [&]() {
    for (size_t k = 0; k < rB; ++k) {
      size_t ob = rO - 1 - k;
      size_t ib = rB - 1 - k;
      if (k >= rO) {
        bIdx[ib] = 0;
      } else {
        bIdx[ib] = (Bdims[ib] == 1) ? 0 : oIdx[ob];
      }
    }
  };

  if (outIsF64) {
    auto outStore = make_out(Dtype::Float64, sizeof(double));
    double *dst = reinterpret_cast<double *>(outStore->data());
    const auto AsrcF32 = aF32 ? Ah.storage()->f32() : memory::span<const float>{};
    const auto AsrcF64 = aF64 ? Ah.storage()->f64() : memory::span<const double>{};
    const auto BsrcF32 = bF32 ? Bh.storage()->f32() : memory::span<const float>{};
    const auto BsrcF64 = bF64 ? Bh.storage()->f64() : memory::span<const double>{};

    if (rO == 0) {
      // Scalar result
      const std::size_t ia = (rA == 0) ? 0 : Ah.view().constIndexOf({});
      const std::size_t ib = (rB == 0) ? 0 : Bh.view().constIndexOf({});
      const double av = aF64 ? AsrcF64[ia] : static_cast<double>(AsrcF32[ia]);
      const double bv = bF64 ? BsrcF64[ib] : static_cast<double>(BsrcF32[ib]);
      dst[0] = std::pow(av, bv);
    } else {
      for (std::size_t i = 0;; ++i) {
        map_to_a();
        map_to_b();
        const std::size_t ia =
            (rA == 0) ? 0 : Ah.view().constIndexOf({aIdx.data(), aIdx.size()});
        const std::size_t ib =
            (rB == 0) ? 0 : Bh.view().constIndexOf({bIdx.data(), bIdx.size()});
        const double av = aF64 ? AsrcF64[ia] : static_cast<double>(AsrcF32[ia]);
        const double bv = bF64 ? BsrcF64[ib] : static_cast<double>(BsrcF32[ib]);
        dst[i] = std::pow(av, bv);
        if (!step_o())
          break;
      }
    }

    HostTensor out(Oshape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  } else {
    auto outStore = make_out(Dtype::Float32, sizeof(float));
    float *dst = reinterpret_cast<float *>(outStore->data());
    const auto Asrc = Ah.storage()->f32();
    const auto Bsrc = Bh.storage()->f32();

    if (rO == 0) {
      const std::size_t ia = (rA == 0) ? 0 : Ah.view().constIndexOf({});
      const std::size_t ib = (rB == 0) ? 0 : Bh.view().constIndexOf({});
      dst[0] = std::pow(Asrc[ia], Bsrc[ib]);
    } else {
      for (std::size_t i = 0;; ++i) {
        map_to_a();
        map_to_b();
        const std::size_t ia =
            (rA == 0) ? 0 : Ah.view().constIndexOf({aIdx.data(), aIdx.size()});
        const std::size_t ib =
            (rB == 0) ? 0 : Bh.view().constIndexOf({bIdx.data(), bIdx.size()});
        dst[i] = std::pow(Asrc[ia], Bsrc[ib]);
        if (!step_o())
          break;
      }
    }

    HostTensor out(Oshape, std::move(outStore));
    return {Tensor::Host(std::move(out))};
  }
}

} // namespace vkcnn::details

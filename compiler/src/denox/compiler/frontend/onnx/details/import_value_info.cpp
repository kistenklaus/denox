#include "denox/compiler/frontend/onnx/details/import_value_info.hpp"
#include "denox/common/TensorDataType.hpp"
#include "denox/compiler/Options.hpp"
#include "denox/compiler/frontend/onnx/details/values/Tensor.hpp"
#include "denox/diag/invalid_argument.hpp"
#include "denox/diag/logging.hpp"
#include "denox/memory/container/optional.hpp"

#include <onnx.pb.h>

namespace denox::onnx::details {

void maybe_set_device_float_type_from_dtype(DeviceTensor &dev, Dtype onnxElem) {
  if (auto want = onnxElem.toTensorType()) {
    dev.handle().setType(*want);
  }
}

void import_value_info(ImportState &state,
                       const ::onnx::ValueInfoProto &valueInfo,
                       ValueInfoImportContext context,
                       const compiler::CompileOptions &options) {
  const memory::string &name = valueInfo.name();
  if (name.empty())
    throw std::runtime_error("vkcnn: \"\" is not a valid tensor name.");

  if (!valueInfo.has_type()) {
    if (context == ValueInfoImportContext::Input)
      throw std::runtime_error(fmt::format(
          "vkcnn: input tensor \"{}\" does not define a type.", name));
    if (context == ValueInfoImportContext::Output ||
        context == ValueInfoImportContext::Hint)
      return; // ignore missing type for Output/Hint
    throw std::logic_error("Unexpected ValueInfoImportContext.");
  }

  const auto &tp = valueInfo.type();
  if (tp.has_optional_type())
    throw std::runtime_error(fmt::format(
        "vkcnn: Tensor \"{}\" has unsupported optional_type", name));
  if (tp.has_sparse_tensor_type())
    throw std::runtime_error(fmt::format(
        "vkcnn: Tensor \"{}\" has unsupported sparse_tensor_type", name));
  if (tp.has_map_type())
    throw std::runtime_error(
        fmt::format("vkcnn: Tensor \"{}\" has unsupported map_type", name));
  if (tp.has_sequence_type())
    throw std::runtime_error(fmt::format(
        "vkcnn: Tensor \"{}\" has unsupported sequence_type", name));
  if (!tp.has_tensor_type()) {
    if (context == ValueInfoImportContext::Hint ||
        context == ValueInfoImportContext::Output)
      return; // ignore for Hint/Output
    throw std::runtime_error(
        fmt::format("vkcnn: Tensor \"{}\" missing tensor_type", name));
  }

  const auto &ttype = tp.tensor_type();

  // dtype is optional for Output/Hint (we only use it to set/verify float type)
  memory::optional<Dtype> dtypeOpt = Dtype::parse(ttype.elem_type());
  if (!dtypeOpt) {
    if (context == ValueInfoImportContext::Input) {
      throw std::runtime_error(
          fmt::format("vkcnn: Tensor \"{}\" has unsupported elem_type {}", name,
                      Dtype::parse_to_string(ttype.elem_type())));
    }
    // For Output/Hint we just skip dtype-based checks.
  }

  // ---------------- HINT ----------------
  if (context == ValueInfoImportContext::Hint) {
    // We don't enforce anything here.
    return;
  }

  // ---------------- INPUT ----------------
  if (context == ValueInfoImportContext::Input) {
    if (!ttype.has_shape()) {
      throw std::runtime_error(fmt::format(
          "vkcnn: tensor {} has unknown shape (dynamic rank unsupported)",
          name));
    }
    // TensorShape::parse
    const auto &shp = ttype.shape();
    SymGraph *symGraph = state.symGraph;
    memory::vector<Symbolic> dims;
    dims.reserve(static_cast<std::size_t>(shp.dim_size()));
    auto g = symGraph;
    if (!g) {
      throw std::runtime_error("vkcnn: symGraph is null");
    }

    auto interfaceDescriptor = std::ranges::find_if(
        options.interfaceDescriptors, [&](const auto &tensorDescriptor) {
          return tensorDescriptor.name == name;
        });

    for (int i = 0; i < shp.dim_size(); ++i) {
      const auto &d = shp.dim(i);
      if (d.has_dim_value()) {
        const int64_t v = d.dim_value();
        if (v < 0) {
          throw std::runtime_error(
              fmt::format("vkcnn: {} has negative dim at axis {}", name, i));
        }
        dims.emplace_back(g, Sym::Const(v));
        // NOTE: Check that it matches the input-shape option if set.

        if (interfaceDescriptor != options.interfaceDescriptors.end()) {
          int ri = shp.dim_size() == 4 ? i : (i + 1);

          if (ri == 1) {
            // channels
            if (interfaceDescriptor->channels.has_value() &&
                interfaceDescriptor->channels.value() != v) {
              DENOX_ERROR(
                  "Input {}, has non dynamic channel count {}, expected {}.",
                  name, v, interfaceDescriptor->channels.value());
              diag::invalid_argument();
            }
            if (interfaceDescriptor->channelValueName) {
              state.output.assignValueName(
                  interfaceDescriptor->channelValueName.value(), Sym::Const(v));
            }
          }
          if (ri == 2) {
            // height
            if (interfaceDescriptor->height.has_value() &&
                interfaceDescriptor->height.value() != v) {
              diag::invalid_argument();
            }

            if (interfaceDescriptor->heightValueName) {
              state.output.assignValueName(
                  interfaceDescriptor->heightValueName.value(), Sym::Const(v));
            }
          }

          if (ri == 3) {
            // width
            if (interfaceDescriptor->width.has_value() &&
                interfaceDescriptor->width.value() != v) {
              diag::invalid_argument();
            }
            if (interfaceDescriptor->widthValueName) {
              state.output.assignValueName(
                  interfaceDescriptor->widthValueName.value(), Sym::Const(v));
            }
          }
        }

      } else if (d.has_dim_param()) {
        assert(symGraph != nullptr);
        const std::string &label = d.dim_param();
        if (label.empty()) {
          throw std::runtime_error(
              fmt::format("vkcnn: {} has empty dim_param at axis {}", name, i));
        }
        Sym s;
        if (shp.dim_size() == 4 && i == 0) {
          s = Sym::Const(1);
        } else {

          int ri = shp.dim_size() == 4 ? i : (i + 1);
          if (ri == 1) {
            // channels.
            if (interfaceDescriptor != options.interfaceDescriptors.end()) {
              if (interfaceDescriptor->channels.has_value()) {
                // overwrite symbolic behavior of onnx model!
                s = Sym::Const(interfaceDescriptor->channels.value());
                // naming the constant!
              } else {
                if (interfaceDescriptor->channelValueName.has_value()) {
                  std::optional<Sym> attempt = state.output.getValueByName(
                      interfaceDescriptor->channelValueName.value());
                  if (attempt) {
                    s = *attempt;
                  } else {
                    s = state.output.requireValueOfName(label, true);
                  }
                } else {
                  s = state.output.requireValueOfName(label, true);
                }
              }
              if (interfaceDescriptor->channelValueName.has_value()) {
                state.output.assignValueName(
                    interfaceDescriptor->channelValueName.value(), s);
              }
            } else {
              // check if a symbol if a identical name exist, then use this one.
              s = state.output.requireValueOfName(label, true);
            }
          }
          if (ri == 2) {
            // height
            if (interfaceDescriptor != options.interfaceDescriptors.end()) {

              if (interfaceDescriptor->height.has_value()) {
                // overwrite symbolic behavior of onnx model!
                s = Sym::Const(interfaceDescriptor->height.value());
                // naming the constant!
              } else {
                if (interfaceDescriptor->heightValueName.has_value()) {
                  std::optional<Sym> attempt = state.output.getValueByName(
                      interfaceDescriptor->heightValueName.value());
                  if (attempt) {
                    s = *attempt;
                  } else {
                    s = state.output.requireValueOfName(label, true);
                  }
                } else {
                  s = state.output.requireValueOfName(label, true);
                }
              }
              if (interfaceDescriptor->heightValueName.has_value()) {
                state.output.assignValueName(
                    interfaceDescriptor->heightValueName.value(), s);
              }
            } else {
              // check if a symbol if a identical name exist, then use this one.
              s = state.output.requireValueOfName(label, true);
            }
          }
          if (ri == 3) {
            // width
            if (interfaceDescriptor != options.interfaceDescriptors.end()) {
              if (interfaceDescriptor->width.has_value()) {
                // overwrite symbolic behavior of onnx model!
                s = Sym::Const(interfaceDescriptor->width.value());
                // naming the constant!
              } else {
                if (interfaceDescriptor->widthValueName.has_value()) {
                  std::optional<Sym> attempt = state.output.getValueByName(
                      interfaceDescriptor->widthValueName.value());
                  if (attempt) {
                    s = *attempt;
                  } else {
                    s = state.output.requireValueOfName(label, true);
                  }
                } else {
                  s = state.output.requireValueOfName(label, true);
                }
              }
              if (interfaceDescriptor->widthValueName.has_value()) {
                state.output.assignValueName(
                    interfaceDescriptor->widthValueName.value(), s);
              }
            } else {
              // check if a symbol if a identical name exist, then use this one.
              s = state.output.requireValueOfName(label, true);
            }
          }

          state.output.assignValueName(label, s, true);
        }
        dims.emplace_back(g, s);
      } else {
        assert(symGraph != nullptr);
        Sym s = symGraph->var();
        dims.emplace_back(g, s);
      }
    }
    TensorShape tshape{state.symGraph, std::move(dims)};

    const size_t r = tshape.rank();

    if (r != 3 && r != 4) {
      throw std::runtime_error(fmt::format(
          "vkcnn: Input \"{}\" must be rank 3 (CHW) or 4 (NCHW); got {}", name,
          r));
    }

    const size_t axC = (r == 4) ? 1 : 0;
    const size_t axH = (r == 4) ? 2 : 1;
    const size_t axW = (r == 4) ? 3 : 2;

    if (r == 4) {
      // If N is constant, require 1; else force to 1.
      if (tshape[0].isConstant()) {
        if (tshape[0].constant() != 1) {
          throw std::runtime_error(fmt::format(
              "Input tensor (\"{}\") has fixed batch {} (unsupported).", name,
              tshape[0].constant()));
        }
      } else {
        tshape[0] = Symbolic{state.symGraph, Sym::Const(1)};
      }
    }

    // Height/Width may be symbolic
    const Sym Hs = static_cast<Sym>(*tshape[axH]);
    const Sym Ws = static_cast<Sym>(*tshape[axW]);
    const Sym Cs = static_cast<Sym>(*tshape[axC]);

    // Optional float-type hint from dtype
    memory::optional<TensorDataType> hint =
        dtypeOpt ? dtypeOpt->toTensorType() : memory::nullopt;
    TensorDataType dtype = TensorDataType::Float16; // <- default
    if (interfaceDescriptor != options.interfaceDescriptors.end() &&
        interfaceDescriptor->dtype != TensorDataType::Auto) {
      dtype = interfaceDescriptor->dtype;
    } else if (hint.has_value()) {
      dtype = *hint;
    }

    compiler::TensorHandle rt = state.output.input(name, Ws, Hs, Cs, dtype);

    DeviceTensor dev(r, std::move(rt));
    state.tensors.emplace(name, Tensor::Device(std::move(dev)));
    return;
  }

  // ---------------- OUTPUT ----------------
  // Must exist and be DeviceTensor
  auto it = state.tensors.find(name);
  if (it == state.tensors.end()) {
    throw std::runtime_error(
        fmt::format("Output \"{}\" was never produced by any node.", name));
  }
  if (!it->second.isDevice()) {
    throw std::runtime_error(
        fmt::format("vkcnn: Output (\"{}\") is not a runtime tensor (constant "
                    "outputs unsupported).",
                    name));
  }

  auto interfaceDescriptor = std::ranges::find_if(
      options.interfaceDescriptors, [&](const auto &tensorDescriptor) {
        return tensorDescriptor.name == name;
      });

  DeviceTensor dev = it->second.device();

  // (A) DType consistency: if ONNX dtype is supported → verify or set
  if (interfaceDescriptor != options.interfaceDescriptors.end() &&
      interfaceDescriptor->dtype != TensorDataType::Auto) {
    dev.handle().setType(interfaceDescriptor->dtype);
  } else if (dtypeOpt) {
    if (auto want = dtypeOpt->toTensorType()) {
      auto &h = dev.handle();
      if (h.type() == TensorDataType::Auto) {
        h.setType(*want);
      } else if (h.type() != *want) {
        throw std::runtime_error("vkcnn: Output tensor type mismatch.");
      }
    }
  }

  // (B) Channel-count check only (ignore all other dims/symbols)
  // We only do this if ONNX provided a tensor shape and a concrete channel dim.
  if (ttype.has_shape()) {
    const auto &oshape = ttype.shape();

    // Determine which axis is channels from the produced tensor rank (3/4).
    const size_t pr = dev.shape().rank();
    if (pr != 3 && pr != 4) {
      throw std::runtime_error(fmt::format(
          "vkcnn: Output tensor (\"{}\") must be rank 3 or 4; got {}.", name,
          pr));
    }
    // If ONNX rank doesn't match or is weird, we don't try to reconcile; we
    // simply skip the channel check.
    if (interfaceDescriptor != options.interfaceDescriptors.end()) {
      const size_t orank = static_cast<size_t>(oshape.dim_size());
      for (std::size_t d = 0; d < orank; ++d) {
        const auto &dim = oshape.dim()[static_cast<int>(d)];
        std::size_t rd = (orank == 4) ? d : (d + 1);
        if (rd == 1) {
          if (dim.has_dim_value() &&
              interfaceDescriptor->channels.has_value() &&
              dim.dim_value() != interfaceDescriptor->channels.value()) {
            DENOX_ERROR("Input \'{}\' has channel count {}, expected {}.", name,
                        dim.dim_value(), interfaceDescriptor->channels.value());
            diag::invalid_argument();
          }
          if (interfaceDescriptor->channelValueName.has_value()) {
            std::optional<Sym> lookup = state.output.getValueByName(
                *interfaceDescriptor->channelValueName);
            if (lookup && dev.handle().channels() != *lookup) {
              DENOX_ERROR("Input \'{}\' has invalid dynamic channel extent.",
                          name);
              diag::invalid_argument();
            }
            if (!lookup) {
              state.output.assignValueName(
                  *interfaceDescriptor->channelValueName,
                  dev.handle().channels());
            }
          }

        } else if (rd == 2) {
          if (dim.has_dim_value() && interfaceDescriptor->height.has_value() &&
              dim.dim_value() != interfaceDescriptor->height.value()) {
            DENOX_ERROR("Input \'{}\' has height {}, expected {}.", name,
                        dim.dim_value(), interfaceDescriptor->height.value());
            diag::invalid_argument();
          }

          if (interfaceDescriptor->heightValueName.has_value()) {
            std::optional<Sym> lookup = state.output.getValueByName(
                *interfaceDescriptor->heightValueName);
            if (lookup && dev.handle().height() != *lookup) {
              DENOX_ERROR("Input \'{}\' has invalid dynamic height extent.", // <- HERE
                          name);
              diag::invalid_argument();
            }
            if (!lookup) {
              state.output.assignValueName(
                  *interfaceDescriptor->heightValueName,
                  dev.handle().channels());
            }
          }
        } else if (rd == 3) {
          if (dim.has_dim_value() && interfaceDescriptor->width.has_value() &&
              dim.dim_value() != interfaceDescriptor->width.value()) {
            DENOX_ERROR("Input \'{}\' has width {}, expected {}.", name,
                        dim.dim_value(), interfaceDescriptor->width.value());
            diag::invalid_argument();
          }
          if (interfaceDescriptor->widthValueName.has_value()) {
            std::optional<Sym> lookup = state.output.getValueByName(
                *interfaceDescriptor->widthValueName);
            if (lookup && dev.handle().width() != *lookup) {
              DENOX_ERROR("Input \'{}\' has invalid dynamic width extent.",
                          name);
              diag::invalid_argument();
            }
            if (!lookup) {
              state.output.assignValueName(*interfaceDescriptor->widthValueName,
                                           dev.handle().channels());
            }
          }
        }
      }
    }
  }

  // Finalize output in backend
  state.output.output(dev.handle(), name);
}

} // namespace denox::onnx::details

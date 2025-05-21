#include "../Tensor.hpp"
#include "pybind11/pytypes.h"
#include "src/host/CNNLayerDescription.hpp"
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ranges>
#include <stdexcept>
#include <string_view>

namespace py = pybind11;

py::dict result_ok(py::object value) {
  py::dict result;
  result["value"] = value;
  result["error"] = py::none();
  return result;
}

py::dict result_err(const std::string_view msg) {
  py::dict result;
  result["value"] = py::none();
  result["error"] = py::str(msg);
  return result;
}

static py::dict generatePipeline(py::object obj) {
  try {
    py::list reflectedLayer = obj;

    std::vector<pyvk::CNNLayerDescription> layers;
    for (auto obj : reflectedLayer) {
      py::dict layer = obj.cast<py::dict>();
      std::string name = layer["name"].cast<py::str>();
      std::string type = layer["type"].cast<py::str>();
      py::list input_shape = layer["input_shape"].cast<py::list>();
      if (input_shape.size() != 4) {
        throw std::runtime_error(
            "Failed to parse python obj. Invalid input shape.");
      }
      pyvk::TensorShape<3> inputShape;
      for (const auto &[i, d] :
           input_shape | std::views::drop(1) | std::views::enumerate) {
        inputShape[i] = d.cast<py::int_>();
      }

      py::list output_shape = layer["output_shape"].cast<py::list>();
      if (output_shape.size() != 4) {
        throw std::runtime_error(
            "Failed to parse python obj. Invalid output shape.");
      }
      pyvk::TensorShape<3> outputShape;
      for (const auto &[i, d] :
           output_shape | std::views::drop(1) | std::views::enumerate) {
        outputShape[i] = d.cast<py::int_>();
      }
      if (type == "input") {
        if (!std::ranges::equal(inputShape, outputShape)) {
          throw std::runtime_error(
              "Failed to parse python obj. Input and output shape have to "
              "match for input layers.");
        }
        layers.push_back(pyvk::CNNLayerDescription::input(name, inputShape));
      } else if (type == "conv2d") {
        py::dict parameters = layer["parameters"].cast<py::dict>();
        pyvk::TensorShape<2> kernelShape;
        {
          py::list kernel_size = parameters["kernel_size"];
          if (kernel_size.size() != 2) {
            throw std::runtime_error("Failed to parse python obj. Expecting "
                                     "kernel_size to have 2 dimensions.");
          }
          for (const auto &[i, d] : kernel_size | std::views::enumerate) {
            kernelShape[i] = d.cast<py::int_>();
          }
        }
        pyvk::TensorShape<2> strideShape;
        {
          py::list stride = parameters["stride"];
          if (stride.size() != 2) {
            throw std::runtime_error("Failed to parse python obj. Expecting "
                                     "stride to have 2 dimensions.");
          }
          for (const auto &[i, d] : stride | std::views::enumerate) {
            strideShape[i] = d.cast<py::int_>();
          }
        }
        pyvk::TensorShape<2> paddingShape;
        {
          py::list padding = parameters["padding"];
          if (padding.size() != 2) {
            throw std::runtime_error("Failed to parse python obj. Expecting "
                                     "padding to have 2 dimensions.");
          }
          for (const auto &[i, d] : padding | std::views::enumerate) {
            paddingShape[i] = d.cast<py::int_>();
          }
        }
        // --- gather the dimensions
        // -------------------------------------------------
        const std::size_t outChannels = outputShape[0];  // O
        const std::size_t inChannels = inputShape[0];    // I
        const std::size_t kernelWidth = kernelShape[0];  // H
        const std::size_t kernelHeight = kernelShape[1]; // W
        const std::size_t weightCount =
            outChannels * inChannels * kernelWidth * kernelHeight;
        std::vector<float> flatWeights =
            py::cast<std::vector<float>>(parameters["weights"]);
        if (flatWeights.size() != weightCount) {
          throw std::runtime_error("Failed to parse python obj. Invalid amount "
                                   "of weights fror a conv2d.");
        }
        pyvk::Tensor<float, pyvk::TensorFormat_OIHW> weights(
            {outChannels, inChannels, kernelHeight, kernelWidth}, flatWeights);

        // TODO bias.
        layers.push_back(pyvk::CNNLayerDescription::conv2d(
            name, inputShape, outputShape, kernelShape, strideShape,
            paddingShape, std::move(weights)));
      } else if (type == "activation-function") {
        py::dict parameters = layer["parameters"].cast<py::dict>();
        std::string funcName = py::cast<std::string>(parameters["func"]);
        if (funcName == "relu") {
          layers.push_back(pyvk::CNNLayerDescription::activation(
              name, inputShape, pyvk::ActivationFunction::Relu));
        } else {
          throw std::runtime_error(
              "Failed to parse python obj. Invalid activation function.");
        }
      } else if (type == "maxpool") {
        py::dict parameters = layer["parameters"].cast<py::dict>();
        pyvk::TensorShape<2> kernelShape;
        {
          py::list kernel_size = parameters["kernel_size"];
          if (kernel_size.size() != 2) {
            throw std::runtime_error("Failed to parse python obj. Expecting "
                                     "kernel_size to have 2 dimensions.");
          }
          for (const auto &[i, d] : kernel_size | std::views::enumerate) {
            kernelShape[i] = d.cast<py::int_>();
          }
        }
        pyvk::TensorShape<2> strideShape;
        {
          py::list stride = parameters["stride"];
          if (stride.size() != 2) {
            throw std::runtime_error("Failed to parse python obj. Expecting "
                                     "stride to have 2 dimensions.");
          }
          for (const auto &[i, d] : stride | std::views::enumerate) {
            strideShape[i] = d.cast<py::int_>();
          }
        }

        layers.push_back(pyvk::CNNLayerDescription::maxPool(
            name, inputShape, outputShape, kernelShape, strideShape));
      } else if (type == "upsample") {
        continue;
        py::dict parameters = layer["parameters"].cast<py::dict>();
        // py::list targetSize = parameters["target_size"].cast<py::list>();
        // if (targetSize.size() != 2) {
        //   throw std::runtime_error("Failed to parse python obj. Expecting "
        //                            "targetSize to have 2 dimensions.");
        // }
        // pyvk::TensorShape<2> targetHW;
        // for (const auto &[i, d] : targetSize | std::views::enumerate) {
        //   targetHW[i] = d.cast<py::int_>();
        // }
        // if (!std::ranges::equal(
        //         std::span(outputShape.begin() + 1, outputShape.end()),
        //         targetHW)) {
        //   throw std::runtime_error("Failed to parse python obj. Expecting "
        //                            "targetSize to match the outputShape");
        // }
        std::string mode = py::cast<std::string>(parameters["mode"]);
        pyvk::UpsampleFilterMode filterMode;
        if (mode == "nearest") {
          filterMode = pyvk::UpsampleFilterMode::Nearest;
        } else {
          throw std::runtime_error(
              "Failed to parse python obj. Invalid filter mode.");
        }

        layers.push_back(pyvk::CNNLayerDescription::upsample(
            name, inputShape, outputShape, filterMode));
      } else if (type == "concat") {

      } else if (type == "output") {

      } else {
        throw std::runtime_error(
            "Failed to parse python obj. Invalid layer type");
      }
    }

    return result_ok(py::str("everything worked"));
  } catch (const std::exception &e) {
    return result_err(e.what());
  } catch (...) {
    return result_err("Unknown error");
  }
}

PYBIND11_MODULE(_pyvk_cpp, m) {
  m.def("generatePipeline", &generatePipeline,
        "Build pipeline from reflected model");
}

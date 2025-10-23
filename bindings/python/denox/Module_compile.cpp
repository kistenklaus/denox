#include "Module.hpp"
#include <fmt/format.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace pydenox {

Module Module::compile(pybind11::object model,
                       // kwargs
                       pybind11::object device,                        //
                       pybind11::object target_env,                    //
                       pybind11::object input_shape,                   //
                       pybind11::object output_shape,                  //
                       pybind11::object input_type,                    //
                       pybind11::object output_type,                   //
                       pybind11::object input_layout,                  //
                       pybind11::object output_layout,                 //
                       pybind11::object input_storage,                 //
                       pybind11::object output_storage,                //
                       pybind11::object coopmat,                       //
                       pybind11::object fusion,                        //
                       pybind11::object memory_concat,                 //
                       pybind11::object spirv_debug_info,              //
                       pybind11::object spirv_non_semantic_debug_info, //
                       pybind11::object spirv_optimize                 //
) {
  throw std::runtime_error("Not implemented yet");
}

} // namespace pydenox

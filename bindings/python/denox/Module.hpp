#pragma once

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <stdexcept>

namespace pydenox {

class Module {
public:
  static Module load(pybind11::object pypath);
  static void define(pybind11::module_ &m);
  static Module compile(pybind11::object model,
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
  );

  explicit Module(void *dnxBuffer, std::size_t dnxBufferSize);
  ~Module();
  Module *enter();
  bool exit(pybind11::object exc_type, pybind11::object exc,
            pybind11::object tb);
  void save(pybind11::object pypath);
  pybind11::bytes bytes() const;
  std::string repr() const;
  std::size_t size() const;
  const void *get() const;

private:
  void release();

private:
  void *m_dnxBuffer;
  std::size_t m_dnxBufferSize;
};

} // namespace pydenox

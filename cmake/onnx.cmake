include_guard(GLOBAL)
include(cmake/protobuf.cmake)  

denox_add_proto_lib(denox::onnx ${CMAKE_SOURCE_DIR}/third_party/onnx/onnx.proto)

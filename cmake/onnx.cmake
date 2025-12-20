include_guard(GLOBAL)
include(${PROJECT_SOURCE_DIR}/cmake/protobuf.cmake)  

denox_add_proto_lib(denox::onnx ${PROJECT_SOURCE_DIR}/third_party/onnx/onnx.proto PIC BUILD_PIL)

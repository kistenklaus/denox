protobuf_dep = dependency('protobuf', required: true)
protoc = find_program('protoc', required: true)

onnx_proto_file = 'onnx.proto'

onnx_gen = custom_target('onnx_proto_gen',
  input  : files(onnx_proto_file),
  output : ['onnx.pb.cc', 'onnx.pb.h'],
  command: [
    protoc,
    '--proto_path=' + meson.current_source_dir(),
    '--cpp_out='    + meson.current_build_dir(),
    onnx_proto_file   # <--- bare name, just like your script
  ],
  build_by_default: true
)

inc = include_directories('.')

onnx_proto_lib = static_library('onnx_proto',
  onnx_gen,                       # Meson takes the .cc from custom_target outputs
  include_directories : inc,
  dependencies        : protobuf_dep,
  install             : false
)

onnx_proto = declare_dependency(
  link_with           : onnx_proto_lib,
  include_directories : inc,      # now you can: #include "onnx.pb.h"
  dependencies        : protobuf_dep
)

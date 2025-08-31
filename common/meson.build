
vkcnn_common_src = files(
  'src/vkcnn/common/io/read_file.cpp',
  'src/vkcnn/common/tensor/FilterHostTensor.cpp',
  'src/vkcnn/common/tensor/ActivationHostTensor.cpp',
  'src/vkcnn/common/tensor/BiasHostTensor.cpp',
  'src/vkcnn/common/tensor/FloatType.cpp',
  'src/vkcnn/common/symbolic/SymGraph_add_entry.cpp',
  'src/vkcnn/common/symbolic/SymGraph_affine.cpp',
  'src/vkcnn/common/symbolic/SymGraph_affine_symbol_cache.cpp',
  'src/vkcnn/common/symbolic/SymGraph_api.cpp',
  'src/vkcnn/common/symbolic/SymGraph_debug.cpp',
  'src/vkcnn/common/symbolic/SymGraph_div_entry.cpp',
  'src/vkcnn/common/symbolic/SymGraph_helpers.cpp',
  'src/vkcnn/common/symbolic/SymGraph_max_entry.cpp',
  'src/vkcnn/common/symbolic/SymGraph_min_entry.cpp',
  'src/vkcnn/common/symbolic/SymGraph_mod_entry.cpp',
  'src/vkcnn/common/symbolic/SymGraph_modsolver.cpp',
  'src/vkcnn/common/symbolic/SymGraph_modsolver_cache.cpp',
  'src/vkcnn/common/symbolic/SymGraph_mul_entry.cpp',
  'src/vkcnn/common/symbolic/SymGraph_nonaffine.cpp',
  'src/vkcnn/common/symbolic/SymGraph_nonaffine_symbol_cache.cpp',
  'src/vkcnn/common/symbolic/SymGraph_sub_entry.cpp',
  'src/vkcnn/common/model/Model_import.cpp',
)

vkcnn_common_inc = include_directories('src')

vkcnn_common_lib = static_library('vkcnn-common',
  vkcnn_common_src,
  dependencies: [onnx_proto],
  include_directories: vkcnn_common_inc,
  install: false,
  pic : true
)

vkcnn_common = declare_dependency(
  link_with: vkcnn_common_lib,
  include_directories: vkcnn_common_inc,
)


vkcnn_shaders_src = files(
  'src/vkcnn/shaders/conv/ConvTemplate.cpp',
  'src/vkcnn/shaders/conv/ConvGEMM.cpp',
  'src/vkcnn/shaders/copy/CopyTransform.cpp',
  'src/vkcnn/shaders/preprocessing.cpp',
)

vkcnn_shaders_inc = include_directories('src')

vkcnn_shaders_lib = static_library('vkcnn-shaders',
  vkcnn_shaders_src,
  dependencies: [vkcnn_common],
  include_directories: vkcnn_shaders_inc,
  install: false,
  pic : true
)

vkcnn_shaders = declare_dependency(
  link_with: vkcnn_shaders_lib,
  include_directories: vkcnn_shaders_inc,
  dependencies: [vkcnn_common],
)


vkcnn_shaders_src = files(
  'src/vkcnn/shaders/conv/Conv3x3mma16x16x16f16_CHWC16_RCSKC16_HR_P1.cpp',
  'src/vkcnn/shaders/conv/Conv3x3mma16x8x8_CHWC8_RCSKC8_HR_P2.cpp',
  'src/vkcnn/shaders/conv/Conv3x3mma16x8x8_CHWC8_RCSKC8_HR_P3.cpp',
  'src/vkcnn/shaders/conv/Conv3x3mma16x8x8_CHWC8_RSCKC8_NR_P2.cpp',
  'src/vkcnn/shaders/conv/ConvTemplate.cpp'
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

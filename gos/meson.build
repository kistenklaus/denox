
# Graph Optimization & Scheduling
vkcnn_gos_src = files(
  'src/vkcnn/gos/SuperGraph.cpp',
)

vkcnn_gos_inc = include_directories('src')

vkcnn_gos_lib = static_library('vkcnn-gos',
  vkcnn_gos_src,
  dependencies: [vkcnn_common],
  include_directories: vkcnn_gos_inc,
  install: false,
  pic : true
)

vkcnn_graph = declare_dependency(
  link_with: vkcnn_gos_lib,
  include_directories: vkcnn_gos_inc,
  dependencies : [vkcnn_common],
)

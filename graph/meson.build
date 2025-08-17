
vkcnn_graph_src = files(
  'src/vkcnn/graph/ComputeGraph.cpp',
)

vkcnn_graph_inc = include_directories('src')

vkcnn_graph_lib = static_library('vkcnn-graph',
  vkcnn_graph_src,
  dependencies: [vkcnn_common],
  include_directories: vkcnn_graph_inc,
  install: false,
  pic : true
)

vkcnn_graph = declare_dependency(
  link_with: vkcnn_graph_lib,
  include_directories: vkcnn_graph_inc,
  dependencies : [vkcnn_common],
)

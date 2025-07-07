project('vkcnn', 
    ['c', 'cpp'],
    version: '0.0.0',
    default_options : [
        'warning_level=3',
        'cpp_std=c++23',
        'b_ndebug=if-release',
        'merian:performance_profiling=true',
        'buildtype=release',
    ]
)

subdir('common')
subdir('shaders')
subdir('runtime')
subdir('dev')



# pybind11 = dependency('pybind11', required: true)
# merian_proj = subproject('merian')
# merian = merian_proj.get_variable('merian_dep')

# pyvk_lib_src = files(
#   'src/vkcnn/common/tensor/FilterHostTensor.cpp',
#   'src/vkcnn/common/tensor/ActivationHostTensor.cpp'
# )


# pyvk_lib_inc = include_directories('src')
# pyvk_lib = static_library('pyvk_lib',
#   pyvk_lib_src,
#   dependencies: [merian],
#   include_directories: pyvk_lib_inc,
#   install: false,
#   pic : true
# )

# pyvk = declare_dependency(
#   link_with: pyvk_lib,
#   include_directories: pyvk_lib_inc,
# )

# subdir('sandbox')
# subdir('modules/pyvk')
# subdir('modules/pyvk_cpp')

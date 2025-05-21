project('purelib-and-platlib', 
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

pybind11 = dependency('pybind11', required: true)

merian_proj = subproject('merian')
merian = merian_proj.get_variable('merian_dep')

subdir('modules/pyvk')
subdir('modules/pyvk_cpp')


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
subdir('gos')
subdir('runtime')

tests_enabled = get_option('dev')
if tests_enabled
  subdir('dev')
endif
